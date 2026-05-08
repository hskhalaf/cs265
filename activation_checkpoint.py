"""
Phases 2 and 3 of the µ-TWO activation-checkpointing pipeline.

Phase 2 — selection
-------------------
``select_activations`` runs the µ-TWO greedy algorithm over the profiler's
intermediates: at each step it evicts the activation with the highest
``size / recompute_time`` ratio (best memory return per millisecond of
recomputation), re-simulates the peak with the new eviction set, and stops
when peak <= the requested limit (default: shrink peak by half the activation
total) — or when an eviction round fails to lower the peak (the peak isn't
activation-dominated, so AC can't help).

The simulator (`_simulate_peak`) accounts for the fact that an evicted
activation still occupies memory briefly during the forward pass (when it is
produced) and during the backward pass (when it is recomputed on demand).

Cascading recomputation: evicting an upstream activation makes any later
eviction that depends on it more expensive.  We track this by maintaining a
per-candidate ``total_recomp_time`` that grows when an ancestor is evicted.

Phase 3 — graph rewriting
-------------------------
``rewrite_with_checkpointing`` takes the selection and edits the FX graph in
place: for each evicted activation we extract its forward-pass subgraph,
splice a copy of it into the backward pass right before the first backward
use, and redirect later backward consumers to the recomputed copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

from graph_prof import GraphProfiler, OP


# --------------------------------------------------------------------------- #
# Phase 2 — selection                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class SelectionResult:
    to_recompute: List[fx.Node]
    to_retain:    List[fx.Node]
    peak_before:  int
    peak_after:   int

    @property
    def freed_bytes(self) -> int:
        return self.peak_before - self.peak_after


def _simulate_peak(profiler: GraphProfiler, evicted: Set[fx.Node]) -> int:
    """Estimate peak live memory if every node in ``evicted`` is recomputed.

    For each node, sum its size into the timeline over the steps where it is
    live.  An evicted intermediate is live (a) from production through its
    last forward use, then (b) at the single backward step that recomputes
    and consumes it.
    """
    n = len(profiler.nodes)
    timeline = [0] * n

    # Look up by node for O(1) eviction info.
    inter_by_node = {i.node: i for i in profiler.intermediates}

    for node in profiler.nodes:
        size = profiler.node_size_bytes.get(node, 0)
        if size == 0:
            continue
        produced = profiler.idx[node]
        user_idx = [profiler.idx[u] for u in node.users if u in profiler.idx]
        last_use = max(user_idx, default=produced)

        if node in evicted:
            inter = inter_by_node[node]
            # Forward window: produced through last forward use.
            for t in range(produced, min(inter.last_fwd_idx + 1, n)):
                timeline[t] += size
            # Backward recomputation spike at first_bwd_idx.
            if 0 <= inter.first_bwd_idx < n:
                timeline[inter.first_bwd_idx] += size
        else:
            for t in range(produced, min(last_use + 1, n)):
                timeline[t] += size
    return max(timeline) if timeline else 0


def select_activations(profiler: GraphProfiler,
                       mem_limit: Optional[int] = None) -> SelectionResult:
    """µ-TWO greedy selection.  Returns a SelectionResult."""
    inters = profiler.intermediates
    peak_before = _simulate_peak(profiler, evicted=set())
    if not inters:
        return SelectionResult([], [], peak_before, peak_before)

    # Default target: cut peak by half the total activation memory.
    if mem_limit is None:
        total_act = sum(i.size_bytes for i in inters)
        mem_limit = peak_before - total_act // 2

    # Quick bail-out: if the peak isn't dominated by activations, AC can't help.
    if _simulate_peak(profiler, set(i.node for i in inters)) >= peak_before:
        return SelectionResult([], [i.node for i in inters],
                               peak_before, peak_before)

    # Per-candidate state: size, total recompute time (grows with cascade),
    # and the set of intermediate ancestors still alive.
    intermediate_set = {i.node for i in inters}
    state: Dict[fx.Node, Dict] = {}
    for i in inters:
        ancestors = {a for a in i.node.all_input_nodes if a in intermediate_set}
        state[i.node] = {
            "size": i.size_bytes,
            "cost": i.recompute_ms,
            "ancestors": ancestors,
            "ratio": i.size_bytes / (i.recompute_ms + 1e-9),
        }

    evicted: Set[fx.Node] = set()
    order:   List[fx.Node] = []
    remaining = set(intermediate_set)
    prev_peak = peak_before

    while remaining:
        peak = _simulate_peak(profiler, evicted)
        if peak <= mem_limit:
            break
        if order and peak >= prev_peak:
            break  # last eviction didn't help — peak is non-activation-dominated
        prev_peak = peak

        best = max(remaining, key=lambda n: state[n]["ratio"])
        evicted.add(best)
        order.append(best)
        remaining.discard(best)

        # Cascade: every remaining candidate that depends on `best` now pays
        # `best`'s recompute cost too.
        for n in remaining:
            cand = state[n]
            if best in cand["ancestors"]:
                cand["ancestors"].discard(best)
                cand["ancestors"].update(state[best]["ancestors"])
                cand["cost"]     += state[best]["cost"]
                cand["ratio"]     = cand["size"] / (cand["cost"] + 1e-9)

    to_recompute, to_retain = _validate_recompute_set(
        order, [i.node for i in inters if i.node not in evicted],
        placeholders={n for n in profiler.nodes if n.op == OP.PLACEHOLDER},
    )
    peak_after = _simulate_peak(profiler, set(to_recompute))
    return SelectionResult(to_recompute, to_retain, peak_before, peak_after)


def _validate_recompute_set(
    to_recompute: List[fx.Node],
    to_retain:    List[fx.Node],
    placeholders: Set[fx.Node],
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Move evicted nodes back to retained if their recomputation inputs
    aren't all reachable.  Iterate to a fixed point.

    An eviction is *valid* only when every input to the evicted node is
    either a placeholder or itself retained (or itself a recomputable node
    whose inputs are valid — handled by iteration).
    """
    recompute = set(to_recompute)
    retain    = set(to_retain)
    changed = True
    while changed:
        changed = False
        valid = placeholders | retain | recompute
        for n in list(recompute):
            if not all(inp in valid for inp in n.all_input_nodes):
                retain.add(n)
                recompute.discard(n)
                changed = True
    # Preserve original ordering.
    new_recompute = [n for n in to_recompute if n in recompute]
    new_retain    = [n for n in to_retain    if n in retain]
    new_retain.extend(n for n in to_recompute if n in retain)
    return new_recompute, new_retain


# --------------------------------------------------------------------------- #
# Phase 3 — graph rewriting                                                   #
# --------------------------------------------------------------------------- #


def rewrite_with_checkpointing(gm: fx.GraphModule,
                               profiler: GraphProfiler,
                               to_recompute: List[fx.Node]) -> fx.GraphModule:
    """For each evicted activation, splice a recomputed copy into the
    backward pass and redirect later backward consumers to it.

    The rewriter is intentionally simple:
    - We process activations in forward order.
    - Each one gets its own recomputation subgraph extracted via
      ``_extract_graph_with_inputs_outputs``.  The "inputs" are the boundary
      of nodes guaranteed alive in the backward pass (placeholders + the
      retained intermediates + any already-recomputed activations).
    - The subgraph is spliced in just before the activation's first
      backward use; later backward uses are redirected to the recomputed
      output.
    """
    if not to_recompute:
        return gm

    name_to_node = {n.name: n for n in gm.graph.nodes}
    placeholders = {n for n in gm.graph.nodes if n.op == OP.PLACEHOLDER}
    recompute_set = set(to_recompute)
    retain_set    = {i.node for i in profiler.intermediates
                     if i.node not in recompute_set}

    # Boundary grows as we recompute more activations: each recomputed
    # activation becomes an input candidate for later subgraph extractions.
    boundary: Set[fx.Node] = placeholders | retain_set
    recomputed: Dict[fx.Node, fx.Node] = {}

    for act in sorted(to_recompute, key=lambda n: profiler.idx[n]):
        first_bwd_use = _first_backward_user(act, profiler)
        if first_bwd_use is None:
            continue

        inputs = _gather_recomp_inputs(act, boundary)
        subgraph = _extract_graph_with_inputs_outputs(
            joint_graph=gm.graph, inputs=inputs, outputs=[act],
        )

        new_act = _splice_subgraph_before(
            gm, subgraph, before=first_bwd_use, name_to_node=name_to_node,
        )
        if new_act is None:
            continue

        # Redirect subsequent backward uses of `act` to the recomputed copy.
        for user in list(act.users):
            if user is new_act:
                continue
            if profiler.idx.get(user, -1) >= profiler.sep_bwd_idx:
                user.replace_input_with(act, new_act)

        recomputed[act] = new_act
        boundary.add(act)  # other recomputations may reuse this output

    # Remove now-unused detach nodes the autograd tracer left behind.
    for n in list(gm.graph.nodes):
        if n.target == torch.ops.aten.detach.default and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()
    return gm


def _first_backward_user(act: fx.Node, profiler: GraphProfiler) -> Optional[fx.Node]:
    bwd_users = [(profiler.idx[u], u) for u in act.users
                 if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd_users, default=(None, None))[1]


def _gather_recomp_inputs(act: fx.Node, boundary: Set[fx.Node]) -> List[fx.Node]:
    """Walk back from ``act`` and collect the nearest ancestors that lie on
    ``boundary`` (placeholders / retained / already-recomputed activations).
    These are the inputs to the extracted subgraph."""
    seen: Set[fx.Node] = set()
    inputs: List[fx.Node] = []
    stack = list(act.all_input_nodes)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in boundary:
            if n not in inputs:
                inputs.append(n)
        else:
            stack.extend(n.all_input_nodes)
    return inputs


def _splice_subgraph_before(
    gm: fx.GraphModule,
    subgraph: fx.Graph,
    before: fx.Node,
    name_to_node: Dict[str, fx.Node],
) -> Optional[fx.Node]:
    """Copy ``subgraph`` (minus its placeholders / output) into ``gm.graph``
    immediately before ``before``.  Returns the copy of the subgraph's output
    node, or ``None`` if the subgraph had no body to splice."""
    sub_to_main: Dict[str, fx.Node] = {}
    for sn in subgraph.nodes:
        if sn.op == "placeholder":
            sub_to_main[sn.name] = name_to_node[sn.name]

    output_target_name: Optional[str] = None
    for sn in subgraph.nodes:
        if sn.op == "output":
            arg = sn.args[0]
            if isinstance(arg, (tuple, list)):
                arg = arg[0]
            output_target_name = arg.name if isinstance(arg, fx.Node) else None
            break

    last_copy: Optional[fx.Node] = None
    with gm.graph.inserting_before(before):
        for sn in subgraph.nodes:
            if sn.op in ("placeholder", "output"):
                continue
            new = gm.graph.node_copy(
                sn, arg_transform=lambda a: sub_to_main[a.name],
            )
            sub_to_main[sn.name] = new
            name_to_node[new.name] = new
            last_copy = new

    if output_target_name is not None and output_target_name in sub_to_main:
        return sub_to_main[output_target_name]
    return last_copy


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def print_ac_decisions(profiler: GraphProfiler,
                       result: SelectionResult) -> None:
    inter_by_node = {i.node: i for i in profiler.intermediates}
    saved = sum(inter_by_node[n].size_bytes for n in result.to_recompute)
    extra_ms = sum(inter_by_node[n].recompute_ms for n in result.to_recompute)
    retained = sum(inter_by_node[n].size_bytes for n in result.to_retain)

    print("\n" + "=" * 80)
    print("ACTIVATION CHECKPOINTING DECISIONS  (µ-TWO greedy)")
    print("=" * 80)

    print(f"\n  RECOMPUTE ({len(result.to_recompute)}):")
    print(f"  {'Name':<30} {'Size(KB)':>10} {'Recomp(ms)':>11} {'Ratio':>12}")
    print("  " + "-" * 70)
    for n in result.to_recompute:
        i = inter_by_node[n]
        ratio = i.size_bytes / (i.recompute_ms + 1e-9)
        print(f"  {n.name:<30} {i.size_bytes / 1024:>10.2f}"
              f" {i.recompute_ms:>11.4f} {ratio:>12.0f}")

    print(f"\n  RETAIN ({len(result.to_retain)}):")
    for n in result.to_retain:
        i = inter_by_node[n]
        print(f"  {n.name:<30} {i.size_bytes / 1024:>10.2f} KB")

    print("\n  Summary:")
    print(f"    Peak before AC: {result.peak_before / (1024**2):>8.2f} MB")
    print(f"    Peak after  AC: {result.peak_after  / (1024**2):>8.2f} MB")
    print(f"    Freed:          {result.freed_bytes / (1024**2):>8.2f} MB")
    print(f"    Activation memory freed:    {saved   / (1024**2):>6.2f} MB")
    print(f"    Activation memory retained: {retained / (1024**2):>6.2f} MB")
    print(f"    Extra computation cost:     {extra_ms:>6.2f} ms")
    print("=" * 80 + "\n")
