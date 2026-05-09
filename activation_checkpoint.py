"""
Phases 2 and 3 of the µ-TWO activation-checkpointing pipeline.

Phase 2 — selection
-------------------
``select_activations`` runs a µ-TWO-style greedy pass over the profiler's
intermediates.  At each step it tests the remaining activations in the memory
simulator, keeps only choices that reduce the current peak, and picks the one
with the best ``size / recompute_time`` ratio.

The simulator accounts for the fact that an evicted
activation still occupies memory briefly during the forward pass (when it is
produced) and during the backward pass (when it is recomputed on demand).

Phase 3 — graph rewriting
-------------------------
``rewrite_with_checkpointing`` takes the selection and edits the FX graph in
place: for each evicted activation we extract its forward-pass subgraph,
splice a copy of it into the backward pass right before the first backward
use, and redirect later backward consumers to the recomputed copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.fx as fx

from graph_prof import GraphProfiler


# --------------------------------------------------------------------------- #
# Phase 2 — selection                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class SelectionResult:
    to_recompute: List[fx.Node]
    to_retain:    List[fx.Node]
    peak_before:  int
    peak_after:   int
    mem_limit:    Optional[int] = None
    estimated_recompute_ms: float = 0.0
    reason: str = ""

    @property
    def freed_bytes(self) -> int:
        return self.peak_before - self.peak_after


def simulated_memory_timeline_by_role(
    profiler: GraphProfiler,
    evicted: Iterable[fx.Node],
) -> Dict:
    """Return the static memory timeline after simulated checkpointing."""
    from graph_prof import NodeType

    n = len(profiler.nodes)
    evicted = set(evicted)
    diff = {role: [0] * (n + 1) for role in NodeType}
    inter_by_node = {i.node: i for i in profiler.intermediates}

    def add_range(role, lo: int, hi: int, size: int) -> None:
        if n == 0:
            return
        lo = max(0, lo)
        hi = min(hi, n - 1)
        if lo > hi:
            return
        diff[role][lo] += size
        diff[role][hi + 1] -= size

    for owner, aliases in profiler.aliases_by_owner.items():
        size = profiler.node_size_bytes.get(owner, 0)
        if size == 0:
            continue

        role = max(profiler.node_type[a] for a in aliases)
        if owner.op == "placeholder":
            add_range(role, 0, n - 1, size)
            continue

        produced = profiler.idx[owner]
        evicted_aliases = [a for a in aliases if a in evicted]
        retained_backward_use = any(
            a not in evicted
            and any(profiler.idx[u] >= profiler.sep_bwd_idx
                    for u in a.users if u in profiler.idx)
            for a in aliases
        )

        if evicted_aliases and not retained_backward_use:
            last_fwd = produced
            for alias in aliases:
                last_fwd = max(last_fwd, profiler.idx[alias])
                fwd_users = [
                    profiler.idx[u] for u in alias.users
                    if u in profiler.idx and profiler.idx[u] < profiler.sep_bwd_idx
                ]
                if fwd_users:
                    last_fwd = max(last_fwd, max(fwd_users))
            add_range(role, produced, last_fwd, size)

            spike_steps = {
                inter_by_node[a].first_bwd_idx
                for a in evicted_aliases
                if a in inter_by_node and 0 <= inter_by_node[a].first_bwd_idx < n
            }
            for t in spike_steps:
                add_range(role, t, t, size)
        else:
            last_use = produced
            for alias in aliases:
                last_use = max(last_use, profiler.idx[alias])
                user_idx = [
                    profiler.idx[u] for u in alias.users if u in profiler.idx
                ]
                if user_idx:
                    last_use = max(last_use, max(user_idx))
            add_range(role, produced, last_use, size)

    timeline = {role: [0] * n for role in NodeType}
    for role in NodeType:
        running = 0
        for t in range(n):
            running += diff[role][t]
            timeline[role][t] = running
    return timeline


def simulate_peak(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> int:
    """Estimate peak live memory for a proposed recomputation set."""
    from graph_prof import NodeType

    timeline = simulated_memory_timeline_by_role(profiler, evicted)
    n = len(profiler.nodes)
    if n == 0:
        return 0
    return max(sum(timeline[role][t] for role in NodeType) for t in range(n))


def _recompute_body(
    profiler: GraphProfiler,
    activation: fx.Node,
    retained_activations: Set[fx.Node],
) -> List[fx.Node]:
    """Forward nodes needed to rebuild ``activation`` from live tensors."""
    placeholders = {n for n in profiler.nodes if n.op == "placeholder"}
    boundary = placeholders | retained_activations
    needed: Set[fx.Node] = set()
    stack: List[fx.Node] = [activation]

    while stack:
        node = stack.pop()
        if node in needed or node in boundary:
            continue
        needed.add(node)
        stack.extend(inp for inp in node.all_input_nodes if inp in profiler.idx)
    return [n for n in profiler.nodes if n in needed]


def recompute_body_for_activation(
    profiler: GraphProfiler,
    activation: fx.Node,
    to_recompute: Iterable[fx.Node],
) -> List[fx.Node]:
    """Return the nodes Phase 3 would copy for one checkpointed activation."""
    all_activations = {i.node for i in profiler.intermediates}
    retained = all_activations - set(to_recompute)
    return _recompute_body(profiler, activation, retained)


def _body_runtime_ms(profiler: GraphProfiler, body: Iterable[fx.Node]) -> float:
    return sum(profiler.avg_runtime_ms.get(n.name, 0.0) for n in body)


def estimate_recompute_ms(
    profiler: GraphProfiler,
    to_recompute: Iterable[fx.Node],
) -> float:
    """Estimate the latency added by independent recomputation chains."""
    selected = set(to_recompute)
    if not selected:
        return 0.0

    all_activations = {i.node for i in profiler.intermediates}
    retained = all_activations - selected
    total = 0.0
    for node in sorted(selected, key=lambda n: profiler.idx[n]):
        total += _body_runtime_ms(profiler, _recompute_body(profiler, node, retained))
    return total


def _candidate_recompute_ms(
    profiler: GraphProfiler,
    candidate: fx.Node,
    evicted: Set[fx.Node],
    all_activations: Set[fx.Node],
    fallback_ms: float,
) -> float:
    retained = all_activations - evicted - {candidate}
    cost = _body_runtime_ms(profiler, _recompute_body(profiler, candidate, retained))
    return cost if cost > 0 else fallback_ms


def _validate_recompute_set(
    profiler: GraphProfiler,
    selected_order: List[fx.Node],
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Drop selections whose recomputation body is not a forward subgraph."""
    all_activations = {i.node for i in profiler.intermediates}
    selected = set(selected_order)
    changed = True

    while changed:
        changed = False
        retained = all_activations - selected
        for node in list(selected_order):
            if node not in selected:
                continue
            body = _recompute_body(profiler, node, retained)
            valid = bool(body) and all(
                n.op in {"call_function", "call_method", "call_module"}
                and profiler.idx[n] < profiler.sep_idx
                for n in body
            )
            if not valid:
                selected.discard(node)
                changed = True

    to_recompute = [n for n in selected_order if n in selected]
    to_retain = sorted(all_activations - selected, key=lambda n: profiler.idx[n])
    return to_recompute, to_retain


_simulate_peak = simulate_peak


def select_activations(profiler: GraphProfiler,
                       mem_limit: Optional[int] = None) -> SelectionResult:
    """Choose retained vs recomputed activations with the Phase 2 greedy pass."""
    inters = profiler.intermediates
    peak_before = simulate_peak(profiler, evicted=set())
    if not inters:
        return SelectionResult([], [], peak_before, peak_before, mem_limit,
                               reason="no activation candidates")

    if mem_limit is None:
        total_act = sum(i.size_bytes for i in inters)
        mem_limit = peak_before - total_act // 2

    all_activations = {i.node for i in inters}
    inter_by_node = {i.node: i for i in inters}
    evicted: Set[fx.Node] = set()
    order: List[fx.Node] = []
    remaining = set(all_activations)
    reason = "memory target reached"

    while remaining:
        current_peak = simulate_peak(profiler, evicted)
        if current_peak <= mem_limit:
            break

        scored = []
        for node in remaining:
            trial_peak = simulate_peak(profiler, evicted | {node})
            peak_drop = current_peak - trial_peak
            if peak_drop <= 0:
                continue
            cost = _candidate_recompute_ms(
                profiler,
                node,
                evicted,
                all_activations,
                inter_by_node[node].recompute_ms,
            )
            ratio = inter_by_node[node].size_bytes / (cost + 1e-9)
            scored.append((ratio, peak_drop, inter_by_node[node].size_bytes, node))

        if not scored:
            reason = "no remaining activation lowers the current peak"
            break

        _, _, _, best = max(scored, key=lambda item: item[:3])
        evicted.add(best)
        order.append(best)
        remaining.discard(best)

    if not remaining and simulate_peak(profiler, evicted) > mem_limit:
        reason = "all useful activations selected before target was reached"

    to_recompute, to_retain = _validate_recompute_set(profiler, order)
    peak_after = simulate_peak(profiler, to_recompute)
    if peak_after > mem_limit and reason == "memory target reached":
        reason = "validated recompute set does not reach the target"
    extra_ms = estimate_recompute_ms(profiler, to_recompute)
    return SelectionResult(to_recompute, to_retain, peak_before, peak_after,
                           mem_limit, extra_ms, reason)


# --------------------------------------------------------------------------- #
# Phase 3 — graph rewriting                                                   #
# --------------------------------------------------------------------------- #


def rewrite_with_checkpointing(gm: fx.GraphModule,
                               profiler: GraphProfiler,
                               to_recompute: List[fx.Node]) -> fx.GraphModule:
    """For each evicted activation, splice a recomputed copy into the
    backward pass and redirect later backward consumers to it.

    Algorithm, per evicted activation ``act``:

    1. Walk back from ``act`` until we hit the *boundary* — nodes guaranteed
       to be alive at backward time (placeholders + retained intermediates).
       Collect every node strictly between the boundary and ``act``; this is
       the recomputation **body** in topological order.
    2. ``node_copy`` each body node into the main graph just before
       ``act``'s first backward use, building a per-iteration map from
       original nodes to their copies.  The copy of ``act`` is the
       "recomputed" tensor.
    3. Replace ``act`` with the recomputed tensor in every later backward
       user.

    Each evicted activation gets its own independent body (no shared
    recomputation across evictions).  This matches the additive cost model
    that Phase 2 uses when it does the cascading-cost accounting.
    """
    if not to_recompute:
        return gm

    placeholders  = {n for n in gm.graph.nodes if n.op == "placeholder"}
    recompute_set = set(to_recompute)
    retain_set    = {i.node for i in profiler.intermediates
                     if i.node not in recompute_set}
    boundary: Set[fx.Node] = placeholders | retain_set

    for act in sorted(to_recompute, key=lambda n: profiler.idx[n]):
        first_bwd_use = _first_backward_user(act, profiler)
        if first_bwd_use is None:
            continue

        body = _gather_recomp_body(act, boundary)
        if not body:
            continue

        # Per-iteration map: original node → the just-inserted copy.
        # Boundary nodes aren't in this map, so arg_transform falls through
        # to the original (which is correct — boundary nodes are alive).
        copies: Dict[fx.Node, fx.Node] = {}
        with gm.graph.inserting_before(first_bwd_use):
            for orig in body:
                new = gm.graph.node_copy(
                    orig,
                    arg_transform=lambda a: copies.get(a, a),
                )
                copies[orig] = new

        new_act = copies[act]
        for user in list(act.users):
            if user is new_act:
                continue
            if profiler.idx.get(user, -1) >= profiler.sep_bwd_idx:
                user.replace_input_with(act, new_act)

    # Remove orphan detach nodes the autograd tracer leaves behind.
    for n in list(gm.graph.nodes):
        if n.target == torch.ops.aten.detach.default and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()
    return gm


def _first_backward_user(act: fx.Node,
                         profiler: GraphProfiler) -> Optional[fx.Node]:
    bwd_users = [(profiler.idx[u], u) for u in act.users
                 if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd_users, default=(None, None))[1]


def _gather_recomp_body(act: fx.Node,
                        boundary: Set[fx.Node]) -> List[fx.Node]:
    """Body nodes (in topological order) needed to recompute ``act`` from
    the boundary.  Excludes the boundary itself; includes ``act``."""
    needed: Set[fx.Node] = set()
    stack: List[fx.Node] = [act]
    while stack:
        n = stack.pop()
        if n in needed or n in boundary:
            continue
        needed.add(n)
        stack.extend(n.all_input_nodes)
    # The graph's nodes are already in topological order, so filter-by-
    # appearance gives us a valid build order.
    return [n for n in act.graph.nodes if n in needed]


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def print_ac_decisions(profiler: GraphProfiler,
                       result: SelectionResult) -> None:
    inter_by_node = {i.node: i for i in profiler.intermediates}
    saved = sum(inter_by_node[n].size_bytes for n in result.to_recompute)
    extra_ms = result.estimated_recompute_ms
    retained = sum(inter_by_node[n].size_bytes for n in result.to_retain)

    print("\n" + "=" * 80)
    print("ACTIVATION CHECKPOINTING DECISIONS  (µ-TWO greedy)")
    print("=" * 80)

    print(f"\n  RECOMPUTE ({len(result.to_recompute)}):")
    print(f"  {'Name':<30} {'Size(KB)':>10} {'Body(ms)':>10} {'Ratio':>12}")
    print("  " + "-" * 70)
    for n in result.to_recompute:
        i = inter_by_node[n]
        cost = _body_runtime_ms(
            profiler,
            recompute_body_for_activation(profiler, n, result.to_recompute),
        )
        ratio = i.size_bytes / (cost + 1e-9)
        print(f"  {n.name:<30} {i.size_bytes / 1024:>10.2f}"
              f" {cost:>10.4f} {ratio:>12.0f}")

    print(f"\n  RETAIN ({len(result.to_retain)}):")
    for n in result.to_retain:
        i = inter_by_node[n]
        print(f"  {n.name:<30} {i.size_bytes / 1024:>10.2f} KB")

    print("\n  Summary:")
    if result.mem_limit is not None:
        print(f"    Target peak:    {result.mem_limit / (1024**2):>8.2f} MB")
    print(f"    Peak before AC: {result.peak_before / (1024**2):>8.2f} MB")
    print(f"    Peak after  AC: {result.peak_after  / (1024**2):>8.2f} MB")
    print(f"    Freed:          {result.freed_bytes / (1024**2):>8.2f} MB")
    print(f"    Stop reason:    {result.reason}")
    print(f"    Activation memory freed:    {saved   / (1024**2):>6.2f} MB")
    print(f"    Activation memory retained: {retained / (1024**2):>6.2f} MB")
    print(f"    Extra computation cost:     {extra_ms:>6.2f} ms")
    print("=" * 80 + "\n")
