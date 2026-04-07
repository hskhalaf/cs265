"""
PHASE 2:  Decide which activations to recompute vs retain.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Optional, Set, Tuple
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from graph_tracer import SEPFunction
from graph_prof import GraphProfiler, IntermediateInfo, NodeType, OP


# Takes the profiler (which has all the graph data) and a set of nodes we've decided to evict. Returns the estimated peak memory in bytes. 
# This gets called repeatedly by the greedy algorithm (once after each eviction) to check if we've reduced peak enough.

def _simulate_peak_memory(profiler: GraphProfiler, evicted: Set[fx.Node]) -> int:
    n_steps = len(profiler.node_list)
    timeline = [0] * n_steps # timeline[t] will hold the total bytes of all tensors alive at step t

    for node in profiler.node_list:
        if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER): # skip output nodes since they do not produce tensors
            continue
        size = profiler.node_sizes.get(node.name, 0) # Get this node's tensor size
        if size == 0:
            continue

        produced_at = profiler.node_to_idx[node]
        user_indices = [profiler.node_to_idx[u] for u in node.users if u in profiler.node_to_idx]
        last_use = max(user_indices) if user_indices else produced_at # Find the latest step that consumes this tensor. 

        if node in evicted:
            # Even though we're evicting it, it must still be computed during the forward pass. We just won't keep the result
            info = profiler.intermediate_info[node]
            for t in range(produced_at, min(info.last_fwd_access + 1, n_steps)):
                timeline[t] += size
            #  During backward, we'd recompute it on the spot when we need it
            if 0 <= info.first_bwd_access < n_steps:
                timeline[info.first_bwd_access] += size 
        else:
            for t in range(produced_at, min(last_use + 1, n_steps)):
                timeline[t] += size

    return max(timeline) if timeline else 0


def select_activations_to_recompute(profiler: GraphProfiler, mem_limit: Optional[int] = None) -> Tuple[List[fx.Node], List[fx.Node]]:
    if not profiler.intermediate_nodes:
        return [], []

    baseline_peak = _simulate_peak_memory(profiler, evicted=set())

    # Default: aim to reduce peak by half the total activation memory.
    if mem_limit is None:
        total_act_mem = sum(profiler.intermediate_info[n].memory_size for n in profiler.intermediate_nodes)
        mem_limit = baseline_peak - (total_act_mem // 2)

    # Build candidate metadata: size, cost, ratio, recomputation sources.
    candidates: Dict[fx.Node, dict] = {}
    for node in profiler.intermediate_nodes:
        info = profiler.intermediate_info[node]
        candidates[node] = {
            "memory_size": info.memory_size,
            "recomp_time": info.recompute_cost_ms,
            "total_recomp_time": info.recompute_cost_ms,
            "recomp_ratio": info.memory_size / (info.recompute_cost_ms + 1e-9),
            "recomp_srcs": set(),
        }

    # For each intermediate, find which of its inputs are also intermediates. This matter for cascading.
    intermediate_set = set(profiler.intermediate_nodes)
    for node in profiler.intermediate_nodes:
        candidates[node]["recomp_srcs"] = {inp for inp in node.all_input_nodes if inp in intermediate_set}

    evicted: Set[fx.Node] = set()
    eviction_order: List[fx.Node] = []
    remaining = set(profiler.intermediate_nodes)
    prev_peak = baseline_peak

    # Quick check: if evicting ALL intermediates doesn't reduce peak, AC can't help.
    if _simulate_peak_memory(profiler, set(profiler.intermediate_nodes)) >= baseline_peak:
        return [], list(profiler.intermediate_nodes)

    # Greedy loop: evict best candidate, re-simulate, stop when target met.
    while remaining:
        current_peak = _simulate_peak_memory(profiler, evicted)
        if current_peak <= mem_limit: break
        if eviction_order and current_peak >= prev_peak:  break  # last eviction didn't help so peak is non-activation-dominated

        prev_peak = current_peak
        # Pick candidate with highest recompute_ratio.
        best_node = max(remaining, key=lambda n: candidates[n]["recomp_ratio"])
        evicted.add(best_node)
        eviction_order.append(best_node)
        remaining.discard(best_node)

        # Cascading recomputation (Algorithms E/F): if best_node is a recomp_src of another candidate, that candidate's cost increases.
        for node in remaining:
            cand = candidates[node]
            if best_node in cand["recomp_srcs"]:
                cand["recomp_srcs"].discard(best_node)
                cand["recomp_srcs"].update(candidates[best_node]["recomp_srcs"])
                cand["total_recomp_time"] += candidates[best_node]["total_recomp_time"]
                cand["recomp_ratio"] = cand["memory_size"] / (cand["total_recomp_time"] + 1e-9)

    to_recompute = eviction_order
    to_retain = [n for n in profiler.intermediate_nodes if n not in evicted]

    # Validation: ensure every evicted node's inputs are available.
    placeholder_set = {n for n in profiler.node_list if n.op == OP.PLACEHOLDER}
    return _validate_recompute_set(to_recompute, to_retain, placeholder_set)


def _validate_recompute_set(to_recompute: List[fx.Node], to_retain: List[fx.Node], placeholder_set: Set[fx.Node]) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Move evicted nodes back to retained if their inputs aren't available. Iterate until stable."""
    recompute_set = set(to_recompute)
    retain_set = set(to_retain)

    changed = True
    while changed:
        changed = False
        valid_inputs = placeholder_set | retain_set
        for node in list(recompute_set):
            if not all(inp in valid_inputs or inp not in recompute_set for inp in node.all_input_nodes):
                retain_set.add(node)
                recompute_set.discard(node)
                changed = True

    recompute_ordered = [n for n in to_recompute if n in recompute_set]
    retain_ordered = [n for n in to_retain if n in retain_set]
    retain_ordered.extend(n for n in to_recompute if n in retain_set)
    return recompute_ordered, retain_ordered


def print_ac_decisions(profiler: GraphProfiler, to_recompute: List[fx.Node], to_retain: List[fx.Node]) -> None:
    """Print recompute/retain decisions with before/after peak memory."""
    total_saved = sum(profiler.intermediate_info[n].memory_size for n in to_recompute)
    total_cost = sum(profiler.intermediate_info[n].recompute_cost_ms for n in to_recompute)
    total_retained = sum(profiler.intermediate_info[n].memory_size for n in to_retain)
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
        print(f"  {node.name:<30} {info.memory_size / 1024:>10.2f} {info.recompute_cost_ms:>11.4f} {ratio:>12.0f}")

    print(f"\n  Intermediates to RETAIN ({len(to_retain)}):")
    print(f"  {'Name':<30} {'Size(KB)':>10}")
    print("  " + "-" * 42)
    for node in to_retain:
        info = profiler.intermediate_info[node]
        print(f"  {node.name:<30} {info.memory_size / 1024:>10.2f}")

    print(f"\n  Summary:")
    print(f"    Peak memory before AC:       {baseline / (1024**2):.2f} MB")
    print(f"    Peak memory after AC:        {after_ac / (1024**2):.2f} MB")
    print(f"    Peak reduction:              {(baseline - after_ac) / (1024**2):.2f} MB")
    print(f"    Activation memory freed:     {total_saved / 1024:.2f} KB ({total_saved / (1024**2):.2f} MB)")
    print(f"    Extra computation cost:      {total_cost:.4f} ms")
    print(f"    Activation memory retained:  {total_retained / 1024:.2f} KB ({total_retained / (1024**2):.2f} MB)")
    print("=" * 80 + "\n")


def replace_subsequent_uses_of(graph: fx.Graph, old_node: fx.Node, new_node: fx.Node) -> None:
    """Replace uses of old_node with new_node for nodes after new_node in the graph."""
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
    return {node.name: node for node in gm.graph.nodes}

def activation_checkpointing_example(gm: fx.GraphModule) -> fx.GraphModule:
    name_to_node = get_name_to_node_map(gm)
    first_back_access = name_to_node["t"]
    node_to_recompute = [name_to_node["relu"]]
    node_to_recompute_names = ["relu"]
    nodes_required = [name_to_node["w1_1"], name_to_node["x_1"]]

    subgraph = _extract_graph_with_inputs_outputs(joint_graph=gm.graph, inputs=nodes_required, outputs=node_to_recompute)
    subgraph.print_tabular()

    with gm.graph.inserting_before(first_back_access):
        for n in subgraph.nodes:
            if n.op in ("placeholder", "output"):
                continue
            new_node = gm.graph.node_copy(n, arg_transform=lambda arg: name_to_node[arg.name])
            if n.name in node_to_recompute_names:
                replace_subsequent_uses_of(gm.graph, old_node=name_to_node[n.name], new_node=new_node)
            name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()
    return gm


def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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

    gm = make_fx(custom_fn)(w1, w2, x)
    gm = remove_detach_nodes(gm)
    print("Original graph:")
    gm.graph.print_tabular()

    with torch.no_grad():
        old_grads = gm(w1, w2, x)

    new_gm = activation_checkpointing_example(gm)
    print("After AC:")
    new_gm.graph.print_tabular()

    with torch.no_grad():
        new_grads = new_gm(w1, w2, x)

    print("Gradient verification:")
    for old, new in zip(old_grads, new_grads):
        print(torch.allclose(old, new))
