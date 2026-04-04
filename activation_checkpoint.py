"""
Activation Checkpointing for CS265 Project — Phase 2 (Selection Algorithm)
and Phase 3 (Graph Rewriting).

Phase 2 implements the μ-TWO greedy selection algorithm: given profiler data
for each intermediate activation (memory size, recomputation cost), decide
which activations to *recompute* during the backward pass and which to
*retain* (checkpoint) in GPU memory.

Phase 3 (to be implemented later) will use the selection output to extract
forward-pass subgraphs and insert recomputation nodes into the backward pass.

μ-TWO Algorithm Summary (adapted for single-model activation checkpointing)
---------------------------------------------------------------------------
The original μ-TWO paper (Purandare et al., MLSys 2023) uses a greedy,
iterative approach for multi-model training.  For single-model AC, the
relevant metric is the **recompute ratio**:

    recompute_ratio = memory_size / recompute_cost

This measures how many bytes of GPU memory we free per millisecond of extra
computation.  The algorithm greedily selects the intermediate with the highest
recompute ratio, marks it for recomputation, and repeats until either all
intermediates are processed or peak memory falls within the budget.

A validation step ensures that each recomputed node's inputs are available at
recomputation time — they must be either placeholder nodes (parameters, batch
data) or retained intermediates.
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
# Phase 2: μ-TWO Greedy Selection Algorithm
# ---------------------------------------------------------------------------


def select_activations_to_recompute(
    profiler: GraphProfiler,
    memory_budget: Optional[int] = None,
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Decide which intermediate activations to recompute vs. retain.

    Implements the μ-TWO greedy selection: rank intermediates by
    ``recompute_ratio = memory_size / recompute_cost`` (descending) and greedily
    mark them for recomputation until peak memory fits within the budget.

    Parameters
    ----------
    profiler : GraphProfiler
        A profiler that has been run and had ``aggregate_stats()`` called.
    memory_budget : int, optional
        Target peak memory in bytes.  If None, all *valid* intermediates are
        selected for recomputation (maximum memory savings).

    Returns
    -------
    (nodes_to_recompute, nodes_to_retain) : tuple of two lists of fx.Node
        The first list contains intermediates that should be discarded during
        the forward pass and recomputed during the backward pass.  The second
        list contains intermediates that should be kept in GPU memory.
    """
    if not profiler.intermediate_nodes:
        return [], []

    # Gather candidates with their profiling data.
    candidates = []
    for node in profiler.intermediate_nodes:
        info = profiler.intermediate_info[node]
        candidates.append({
            "node": node,
            "info": info,
            "memory_size": info.memory_size,
            "recompute_cost": info.recompute_cost_ms,
            # Ratio: bytes freed per ms of recomputation.
            # Add epsilon to avoid division by zero for near-instant ops.
            "recompute_ratio": (
                info.memory_size / (info.recompute_cost_ms + 1e-9)
            ),
        })

    # Sort by recompute_ratio descending (best bang for buck first).
    candidates.sort(key=lambda c: c["recompute_ratio"], reverse=True)

    # Greedily select candidates for recomputation.
    to_recompute: List[fx.Node] = []
    to_retain: List[fx.Node] = []
    memory_saved = 0

    for cand in candidates:
        if memory_budget is not None and memory_saved >= memory_budget:
            to_retain.append(cand["node"])
            continue
        to_recompute.append(cand["node"])
        memory_saved += cand["memory_size"]

    # Validation pass: ensure each recomputed node's inputs are either
    # placeholders or retained intermediates.  If not, move it to retained.
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
        still_recompute = []
        for node in list(recompute_set):
            inputs_ok = all(
                inp in valid_inputs or inp not in recompute_set
                for inp in node.all_input_nodes
            )
            if inputs_ok:
                still_recompute.append(node)
            else:
                retain_set.add(node)
                recompute_set.discard(node)
                changed = True

    # Preserve original ordering.
    recompute_ordered = [n for n in to_recompute if n in recompute_set]
    retain_ordered = [n for n in to_retain if n in retain_set]
    # Nodes moved from recompute to retain.
    moved = [n for n in to_recompute if n in retain_set]
    retain_ordered.extend(moved)

    return recompute_ordered, retain_ordered


def print_ac_decisions(
    profiler: GraphProfiler,
    to_recompute: List[fx.Node],
    to_retain: List[fx.Node],
) -> None:
    """Print a human-readable summary of the AC selection decisions.

    Parameters
    ----------
    profiler : GraphProfiler
        The profiler instance (for accessing IntermediateInfo).
    to_recompute : list of fx.Node
        Nodes selected for recomputation.
    to_retain : list of fx.Node
        Nodes selected for retention.
    """
    total_saved = sum(
        profiler.intermediate_info[n].memory_size for n in to_recompute
    )
    total_cost = sum(
        profiler.intermediate_info[n].recompute_cost_ms for n in to_recompute
    )
    total_retained = sum(
        profiler.intermediate_info[n].memory_size for n in to_retain
    )

    print("\n" + "=" * 80)
    print("ACTIVATION CHECKPOINTING DECISIONS (μ-TWO Greedy Selection)")
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
    print(f"    Memory saved by recomputation: {total_saved / 1024:.2f} KB "
          f"({total_saved / (1024 * 1024):.2f} MB)")
    print(f"    Extra computation cost:        {total_cost:.4f} ms")
    print(f"    Memory still retained:         {total_retained / 1024:.2f} KB "
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
