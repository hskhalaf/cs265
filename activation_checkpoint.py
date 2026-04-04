"""
Activation Checkpointing for CS265 Project — Phase 2 (Selection Algorithm)
and Phase 3 (Graph Rewriting).

Phase 2 implements the mu-TWO greedy selection algorithm (Purandare et al.,
MLSys 2023): given profiler data for each intermediate activation, decide
which to *recompute* during the backward pass and which to *retain*.

Phase 3 (to be implemented later) will use the selection output to extract
forward-pass subgraphs and insert recomputation nodes into the backward pass.

mu-TWO Algorithm (adapted for single-model activation checkpointing)
---------------------------------------------------------------------
The full mu-TWO algorithm considers both swap (CPU offloading) and recompute.
For single-model AC without CPU offloading, we use the recompute branch:

1.  Build a candidate set of all intermediate activations.
2.  Each iteration:
    a. Pick the candidate with the highest recompute_ratio
       (= memory_size / total_recomp_time).
    b. Mark it for recomputation; update the memory simulator.
    c. Propagate dependency changes: if the evicted tensor is a recompute
       source for another tensor, update that tensor's recomp_time.
3.  Stop when the memory simulator reports peak_memory <= mem_limit.
4.  Validate: ensure every recomputed node's inputs are available.
"""

import torch
import torch.nn as nn
import torch.fx as fx

from typing import Dict, List, Optional, Set, Tuple
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

from graph_tracer import SEPFunction
from graph_prof import GraphProfiler, IntermediateInfo, NodeType, OP


# ---------------------------------------------------------------------------
# Memory Simulator
# ---------------------------------------------------------------------------


def _simulate_peak_memory(
    profiler: GraphProfiler,
    evicted: Set[fx.Node],
) -> int:
    """Estimate peak memory (bytes) given a set of evicted intermediates.

    Walks the node timeline and computes live memory at each step.  An evicted
    intermediate is not counted during its idle period (between last_fwd_access
    and first_bwd_access), but IS counted during its active forward period
    (it must be computed in the forward pass even if it will be discarded
    afterwards) and during its recomputation point in the backward pass.

    For simplicity, we model recomputation as instantaneous: the evicted tensor
    is live only at the exact step of its first backward use (when it would be
    recomputed), not across any span.

    Parameters
    ----------
    profiler : GraphProfiler
        The profiler with completed static analysis.
    evicted : set of fx.Node
        Intermediate nodes that have been marked for recomputation.

    Returns
    -------
    int
        Estimated peak memory in bytes.
    """
    n_steps = len(profiler.node_list)
    timeline = [0] * n_steps

    for node in profiler.node_list:
        if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
            continue
        size = profiler.node_sizes.get(node.name, 0)
        if size == 0:
            continue

        produced_at = profiler.node_to_idx[node]
        user_indices = [
            profiler.node_to_idx[u]
            for u in node.users
            if u in profiler.node_to_idx
        ]
        last_use = max(user_indices) if user_indices else produced_at

        if node in evicted:
            # Evicted intermediate: live during forward (produced_at to
            # last_fwd_access), then freed, then briefly live at
            # first_bwd_access when it is recomputed.
            info = profiler.intermediate_info[node]
            # Forward period: produced_at to last_fwd_access.
            fwd_end = min(info.last_fwd_access + 1, n_steps)
            for t in range(produced_at, fwd_end):
                timeline[t] += size
            # Recomputation point: first_bwd_access only.
            if 0 <= info.first_bwd_access < n_steps:
                timeline[info.first_bwd_access] += size
        else:
            # Non-evicted tensor: live from produced_at to last_use.
            for t in range(produced_at, min(last_use + 1, n_steps)):
                timeline[t] += size

    return max(timeline) if timeline else 0


# ---------------------------------------------------------------------------
# Phase 2: mu-TWO Greedy Selection Algorithm
# ---------------------------------------------------------------------------


def select_activations_to_recompute(
    profiler: GraphProfiler,
    mem_limit: Optional[int] = None,
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Decide which intermediate activations to recompute vs. retain.

    Implements the mu-TWO greedy selection (Algorithm B from the paper):
    iteratively pick the candidate with the highest recompute_ratio, evict it,
    re-simulate peak memory, and stop when peak fits within mem_limit.

    Parameters
    ----------
    profiler : GraphProfiler
        A profiler that has been run and had ``aggregate_stats()`` called.
    mem_limit : int, optional
        Target peak memory in bytes.  If None, defaults to
        ``current_peak - 0.5 * total_activation_memory`` — i.e. we aim to
        reduce peak by half the activation footprint.

    Returns
    -------
    (nodes_to_recompute, nodes_to_retain) : tuple of two lists of fx.Node
    """
    if not profiler.intermediate_nodes:
        return [], []

    # Compute the baseline peak (no evictions).
    baseline_peak = _simulate_peak_memory(profiler, evicted=set())

    # Default mem_limit: reduce peak by half the total activation memory.
    if mem_limit is None:
        total_act_mem = sum(
            profiler.intermediate_info[n].memory_size
            for n in profiler.intermediate_nodes
        )
        mem_limit = baseline_peak - (total_act_mem // 2)

    # Build candidate list with recompute metadata.
    # recomp_cnt tracks how many times a tensor would need to be recomputed
    # during its lifetime.  For now, each intermediate is recomputed once
    # (at its first backward use).  If a retained tensor that serves as a
    # recompute source for another tensor is itself later evicted, the
    # dependent tensor's recomp_cnt increases (cascading recomputation).
    candidates: Dict[fx.Node, dict] = {}
    for node in profiler.intermediate_nodes:
        info = profiler.intermediate_info[node]
        candidates[node] = {
            "memory_size": info.memory_size,
            "recomp_time": info.recompute_cost_ms,
            "recomp_cnt": 1,
            "total_recomp_time": info.recompute_cost_ms,
            "recomp_ratio": (
                info.memory_size / (info.recompute_cost_ms + 1e-9)
            ),
            # Sources: the intermediate nodes needed to recompute this one.
            "recomp_srcs": {
                inp for inp in node.all_input_nodes
                if inp in candidates or inp in profiler.intermediate_info
            },
        }

    # Rebuild recomp_srcs now that all candidates are registered.
    intermediate_set = set(profiler.intermediate_nodes)
    for node in profiler.intermediate_nodes:
        candidates[node]["recomp_srcs"] = {
            inp for inp in node.all_input_nodes
            if inp in intermediate_set
        }

    evicted: Set[fx.Node] = set()
    eviction_order: List[fx.Node] = []
    remaining = set(profiler.intermediate_nodes)

    # Greedy loop (Algorithm B): each iteration evicts the best candidate.
    while remaining:
        # Check stopping criterion: simulate peak with current evictions.
        current_peak = _simulate_peak_memory(profiler, evicted)
        if current_peak <= mem_limit:
            break

        # Pick candidate with highest recompute_ratio.
        best_node = None
        best_ratio = -1.0
        for node in remaining:
            ratio = candidates[node]["recomp_ratio"]
            if ratio > best_ratio:
                best_ratio = ratio
                best_node = node

        if best_node is None:
            break

        # Evict the best candidate.
        evicted.add(best_node)
        eviction_order.append(best_node)
        remaining.discard(best_node)

        # --- Propagate dependency changes (Algorithm E/F from the paper). --
        # If best_node is a recomp_src of another candidate, that candidate
        # can no longer use best_node directly — it would need best_node to
        # be recomputed first, increasing its own total_recomp_time.
        for node in remaining:
            cand = candidates[node]
            if best_node in cand["recomp_srcs"]:
                # The evicted node's own sources become transitive sources.
                cand["recomp_srcs"].discard(best_node)
                cand["recomp_srcs"].update(candidates[best_node]["recomp_srcs"])
                # Recomputing node now also requires recomputing best_node.
                cand["recomp_cnt"] += candidates[best_node]["recomp_cnt"]
                cand["total_recomp_time"] = (
                    cand["recomp_time"] * cand["recomp_cnt"]
                )
                # Update ratio.
                cand["recomp_ratio"] = (
                    cand["memory_size"] / (cand["total_recomp_time"] + 1e-9)
                )

    # Everything not evicted is retained.
    to_recompute = eviction_order
    to_retain = [n for n in profiler.intermediate_nodes if n not in evicted]

    # Validation pass: ensure recomputed nodes' inputs are available.
    placeholder_set = {
        n for n in profiler.node_list if n.op == OP.PLACEHOLDER
    }
    to_recompute, to_retain = _validate_recompute_set(
        to_recompute, to_retain, placeholder_set
    )

    return to_recompute, to_retain


def _validate_recompute_set(
    to_recompute: List[fx.Node],
    to_retain: List[fx.Node],
    placeholder_set: Set[fx.Node],
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Ensure every recomputed node can actually be recomputed.

    A node can be recomputed only if all of its inputs are either:
      (a) placeholder nodes (parameters, optimizer states, batch data), or
      (b) retained intermediate activations.

    Nodes that fail this check are moved from ``to_recompute`` to ``to_retain``.
    The check is iterated until stable, because moving one node from recompute
    to retain may make another node's inputs valid.
    """
    recompute_set = set(to_recompute)
    retain_set = set(to_retain)

    changed = True
    while changed:
        changed = False
        valid_inputs = placeholder_set | retain_set
        for node in list(recompute_set):
            inputs_ok = all(
                inp in valid_inputs or inp not in recompute_set
                for inp in node.all_input_nodes
            )
            if not inputs_ok:
                retain_set.add(node)
                recompute_set.discard(node)
                changed = True

    # Preserve original ordering.
    recompute_ordered = [n for n in to_recompute if n in recompute_set]
    retain_ordered = [n for n in to_retain if n in retain_set]
    moved = [n for n in to_recompute if n in retain_set]
    retain_ordered.extend(moved)

    return recompute_ordered, retain_ordered


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_ac_decisions(
    profiler: GraphProfiler,
    to_recompute: List[fx.Node],
    to_retain: List[fx.Node],
) -> None:
    """Print a human-readable summary of the AC selection decisions."""
    total_saved = sum(
        profiler.intermediate_info[n].memory_size for n in to_recompute
    )
    total_cost = sum(
        profiler.intermediate_info[n].recompute_cost_ms for n in to_recompute
    )
    total_retained = sum(
        profiler.intermediate_info[n].memory_size for n in to_retain
    )

    baseline = _simulate_peak_memory(profiler, evicted=set())
    after_ac = _simulate_peak_memory(profiler, evicted=set(to_recompute))

    print("\n" + "=" * 80)
    print("ACTIVATION CHECKPOINTING DECISIONS (mu-TWO Greedy Selection)")
    print("=" * 80)

    print(f"\n  Intermediates to RECOMPUTE ({len(to_recompute)}):")
    print(f"  {'Name':<30} {'Size(KB)':>10} {'Recomp(ms)':>11} {'Ratio':>12}")
    print("  " + "-" * 70)
    for node in to_recompute:
        info = profiler.intermediate_info[node]
        ratio = info.memory_size / (info.recompute_cost_ms + 1e-9)
        print(
            f"  {node.name:<30} "
            f"{info.memory_size / 1024:>10.2f} "
            f"{info.recompute_cost_ms:>11.4f} "
            f"{ratio:>12.0f}"
        )

    print(f"\n  Intermediates to RETAIN ({len(to_retain)}):")
    print(f"  {'Name':<30} {'Size(KB)':>10}")
    print("  " + "-" * 42)
    for node in to_retain:
        info = profiler.intermediate_info[node]
        print(f"  {node.name:<30} {info.memory_size / 1024:>10.2f}")

    print(f"\n  Summary:")
    print(f"    Peak memory before AC:       {baseline / (1024 * 1024):.2f} MB")
    print(f"    Peak memory after AC:        {after_ac / (1024 * 1024):.2f} MB")
    print(f"    Peak reduction:              {(baseline - after_ac) / (1024 * 1024):.2f} MB")
    print(f"    Activation memory freed:     {total_saved / 1024:.2f} KB "
          f"({total_saved / (1024 * 1024):.2f} MB)")
    print(f"    Extra computation cost:      {total_cost:.4f} ms")
    print(f"    Activation memory retained:  {total_retained / 1024:.2f} KB "
          f"({total_retained / (1024 * 1024):.2f} MB)")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Utilities (from starter code, preserved for Phase 3)
# ---------------------------------------------------------------------------


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node,
) -> None:
    """Replace uses of ``old_node`` with ``new_node`` for all nodes that appear
    after ``new_node`` in the graph.  Forward-pass uses are left untouched."""
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove identity detach nodes inserted by autograd tracing."""
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    """Build a mapping from node name to node object."""
    return {node.name: node for node in gm.graph.nodes}


# ---------------------------------------------------------------------------
# Hardcoded example from the starter code (kept for reference / testing).
# ---------------------------------------------------------------------------


def activation_checkpointing_example(gm: fx.GraphModule) -> fx.GraphModule:
    """Hardcoded AC example from the starter code.

    Recomputes the first relu in a 2-layer linear+relu network.  This is NOT
    the general algorithm — it is kept here as a reference for Phase 3 graph
    rewriting.
    """
    name_to_node = get_name_to_node_map(gm)
    first_back_access = name_to_node["t"]
    node_to_recompute = [name_to_node["relu"]]
    node_to_recompute_names = ["relu"]
    nodes_required_to_recompute = [name_to_node["w1_1"], name_to_node["x_1"]]

    recompute_subgraph = _extract_graph_with_inputs_outputs(
        joint_graph=gm.graph,
        inputs=nodes_required_to_recompute,
        outputs=node_to_recompute,
    )
    print("Extracted recomputation sub-graph: ")
    recompute_subgraph.print_tabular()

    with gm.graph.inserting_before(first_back_access):
        for n in recompute_subgraph.nodes:
            if n.op == "placeholder" or n.op == "output":
                continue
            new_node = gm.graph.node_copy(
                n, arg_transform=lambda arg: name_to_node[arg.name]
            )
            if n.name in node_to_recompute_names:
                old_node = name_to_node[n.name]
                replace_subsequent_uses_of(
                    gm.graph, old_node=old_node, new_node=new_node,
                )
            name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()
    return gm


# ---------------------------------------------------------------------------
# Standalone demo (same as starter code's __main__)
# ---------------------------------------------------------------------------


def custom_fn(
    w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor,
) -> torch.Tensor:
    """Simple 2-layer linear+relu function for testing AC."""
    z = torch.mm(w1, x)
    z = nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = nn.functional.relu(z)
    z = z.sum()
    z = SEPFunction.apply(z)
    z.backward()
    return w1.grad, w2.grad


if __name__ == "__main__":
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    with torch.no_grad():
        old_grads = graph_module(w1, w2, x)

    new_graph_module = activation_checkpointing_example(graph_module)
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    with torch.no_grad():
        new_grads = new_graph_module(w1, w2, x)

    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))
