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

    for owner, aliases in profiler.aliases_by_owner.items():
        size = profiler.node_size_bytes.get(owner, 0)
        if size == 0:
            continue
        produced = 0 if owner.op == OP.PLACEHOLDER else profiler.idx[owner]
        evicted_aliases = [a for a in aliases if a in evicted]

        retained_backward_use = any(
            a not in evicted
            and any(profiler.idx[u] >= profiler.sep_bwd_idx
                    for u in a.users if u in profiler.idx)
            for a in aliases
        )

        if evicted_aliases and not retained_backward_use:
            # Forward window: the shared storage exists until its last
            # forward/loss use.  Backward gets a one-step recomputation spike.
            last_fwd = produced
            for alias in aliases:
                last_fwd = max(last_fwd, profiler.idx[alias])
                fwd_users = [
                    profiler.idx[u] for u in alias.users
                    if u in profiler.idx and profiler.idx[u] < profiler.sep_bwd_idx
                ]
                if fwd_users:
                    last_fwd = max(last_fwd, max(fwd_users))
            for t in range(produced, min(last_fwd + 1, n)):
                timeline[t] += size

            spike_steps = {
                inter_by_node[a].first_bwd_idx
                for a in evicted_aliases
                if a in inter_by_node and 0 <= inter_by_node[a].first_bwd_idx < n
            }
            for t in spike_steps:
                timeline[t] += size
        else:
            last_use = produced
            for alias in aliases:
                last_use = max(last_use, profiler.idx[alias])
                user_idx = [
                    profiler.idx[u] for u in alias.users if u in profiler.idx
                ]
                if user_idx:
                    last_use = max(last_use, max(user_idx))
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

    placeholders  = {n for n in gm.graph.nodes if n.op == OP.PLACEHOLDER}
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
