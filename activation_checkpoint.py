"""
Phase 2 — Activation checkpointing (selection) and Phase 3 (graph rewriting).

Phase 2 — selection
-------------------
``select_activations`` runs the µ-TWO greedy.  At each step it tests every
remaining intermediate activation in the static memory simulator, keeps only
the candidates that lower the current peak, and picks the one with the best
``size / recompute_time`` ratio.  The cascade is handled by re-evaluating
costs against the *current* eviction set: if a candidate's recompute body
overlaps with already-evicted ancestors, those ancestors' runtimes are added
to its cost on the fly.

The simulator (``simulate_peak``) is the same live-memory walk as Phase 1,
with one twist: an evicted activation is treated as live only on the forward
window (production → last forward use) and at a single backward step
(``first_bwd_idx`` — the recomputation spike).  Everything else uses the
ordinary lifetime.

Phase 3 — graph rewriting
-------------------------
``rewrite_with_checkpointing`` takes a selection and edits the FX graph in
place.  For each evicted activation it extracts the forward-pass subgraph
that produces it, splices a copy of that subgraph into the backward pass
right before the activation's first backward use, and redirects the later
backward consumers to the recomputed copy.

Each evicted activation gets its own independent recomputation chain — no
shared recomputation across evictions.  This matches the additive cost
model the selector uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.fx as fx

from graph_prof import GraphProfiler, NodeType


# --------------------------------------------------------------------------- #
# Selection result                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class SelectionResult:
    """The output of ``select_activations``: which activations to recompute,
    which to retain, and the static numbers describing the trade-off."""
    to_recompute:           List[fx.Node]
    to_retain:              List[fx.Node]
    peak_before:            int
    peak_after:             int
    mem_limit:              Optional[int] = None
    estimated_recompute_ms: float         = 0.0
    reason:                 str           = ""

    @property
    def freed_bytes(self) -> int:
        return self.peak_before - self.peak_after


# --------------------------------------------------------------------------- #
# Static helpers — used by both Phase 2 (selection) and Phase 3 (rewrite)     #
# --------------------------------------------------------------------------- #

def recompute_body(profiler: GraphProfiler,
                   activation: fx.Node,
                   retained: Set[fx.Node]) -> List[fx.Node]:
    """Forward subgraph needed to rebuild ``activation`` from live tensors.

    Walks back from ``activation`` through ``all_input_nodes``, stopping at
    nodes guaranteed to be alive at backward time: placeholders (params,
    optimizer state, batched inputs) plus ``retained`` activations.  Returns
    the body in topological order, including ``activation`` itself but
    excluding the boundary.
    """
    placeholders = {n for n in profiler.nodes if n.op == "placeholder"}
    boundary     = placeholders | retained

    needed: Set[fx.Node] = set()
    stack: List[fx.Node] = [activation]
    while stack:
        n = stack.pop()
        if n in needed or n in boundary:
            continue
        needed.add(n)
        stack.extend(n.all_input_nodes)
    return [n for n in activation.graph.nodes if n in needed]


def body_runtime_ms(profiler: GraphProfiler, body: Iterable[fx.Node]) -> float:
    """Sum the average per-node runtime over a recomputation body."""
    return sum(profiler.avg_runtime_ms.get(n.name, 0.0) for n in body)


def estimate_recompute_ms(profiler: GraphProfiler,
                          to_recompute: Iterable[fx.Node]) -> float:
    """Total ms added to the iteration if every node in ``to_recompute`` is
    rebuilt independently in the backward pass."""
    selected = set(to_recompute)
    if not selected:
        return 0.0
    all_acts = {i.node for i in profiler.intermediates}
    retained = all_acts - selected
    return sum(
        body_runtime_ms(profiler, recompute_body(profiler, n, retained))
        for n in sorted(selected, key=lambda n: profiler.idx[n])
    )


# --------------------------------------------------------------------------- #
# Memory simulator                                                            #
# --------------------------------------------------------------------------- #

def simulate_timeline_by_role(profiler: GraphProfiler,
                              evicted: Iterable[fx.Node]) -> Dict[NodeType, List[int]]:
    """Live bytes at each step, by role, *if* every node in ``evicted`` is
    recomputed instead of stored.

    Same shape as ``profiler.memory_timeline_by_role()`` but with one rule:
    if any alias of an owner is in ``evicted`` AND no other alias is used in
    backward (i.e. nothing keeps the storage alive across the boundary), the
    owner is live only on the forward window plus a single backward
    recomputation step.
    """
    n = len(profiler.nodes)
    evicted = set(evicted)
    timeline: Dict[NodeType, List[int]] = {role: [0] * n for role in NodeType}
    inter_by_node = {i.node: i for i in profiler.intermediates}

    for owner, aliases in profiler.aliases_by_owner.items():
        size = profiler.node_size_bytes.get(owner, 0)
        if size == 0:
            continue

        role = max(profiler.node_type[a] for a in aliases)
        if owner.op == "placeholder":
            for t in range(n):
                timeline[role][t] += size
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
            # Forward window only — live until the last forward use, then
            # gone until a one-step recomputation spike at the backward use.
            last_fwd = produced
            for alias in aliases:
                last_fwd = max(last_fwd, profiler.idx[alias])
                fwd_users = [profiler.idx[u] for u in alias.users
                             if u in profiler.idx
                             and profiler.idx[u] < profiler.sep_bwd_idx]
                if fwd_users:
                    last_fwd = max(last_fwd, max(fwd_users))
            for t in range(produced, min(last_fwd + 1, n)):
                timeline[role][t] += size

            spike_steps = {inter_by_node[a].first_bwd_idx
                           for a in evicted_aliases
                           if a in inter_by_node
                           and 0 <= inter_by_node[a].first_bwd_idx < n}
            for t in spike_steps:
                timeline[role][t] += size
        else:
            # Ordinary lifetime — produced through last (forward or backward) use.
            last_use = produced
            for alias in aliases:
                last_use = max(last_use, profiler.idx[alias])
                user_idx = [profiler.idx[u] for u in alias.users
                            if u in profiler.idx]
                if user_idx:
                    last_use = max(last_use, max(user_idx))
            for t in range(produced, min(last_use + 1, n)):
                timeline[role][t] += size
    return timeline


def simulate_peak(profiler: GraphProfiler,
                  evicted: Iterable[fx.Node]) -> int:
    """Peak bytes if every node in ``evicted`` is recomputed."""
    timeline = simulate_timeline_by_role(profiler, evicted)
    n = len(profiler.nodes)
    if n == 0:
        return 0
    return max(sum(timeline[role][t] for role in NodeType) for t in range(n))


# --------------------------------------------------------------------------- #
# Validation: drop selections that don't actually have a forward subgraph     #
# --------------------------------------------------------------------------- #

def validate_recompute_set(profiler: GraphProfiler,
                           selected_order: List[fx.Node],
                           ) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Iterate to a fixed point: drop any selection whose recomputation body
    isn't a pure forward call_function/call_method/call_module subgraph.

    Backward-region nodes can't be in a body (they don't exist yet at
    recompute time); placeholders are fine because they're alive throughout.
    """
    all_acts = {i.node for i in profiler.intermediates}
    selected = set(selected_order)
    changed  = True

    while changed:
        changed  = False
        retained = all_acts - selected
        for node in list(selected_order):
            if node not in selected:
                continue
            body = recompute_body(profiler, node, retained)
            valid = bool(body) and all(
                n.op in {"call_function", "call_method", "call_module"}
                and profiler.idx[n] < profiler.sep_idx
                for n in body
            )
            if not valid:
                selected.discard(node)
                changed = True

    to_recompute = [n for n in selected_order if n in selected]
    to_retain    = sorted(all_acts - selected, key=lambda n: profiler.idx[n])
    return to_recompute, to_retain


# --------------------------------------------------------------------------- #
# Phase 2 — greedy selection                                                  #
# --------------------------------------------------------------------------- #

def select_activations(profiler: GraphProfiler,
                       mem_limit: Optional[int] = None) -> SelectionResult:
    """µ-TWO greedy selection.

    Per iteration:
      1. ``simulate_peak`` once to check the current peak.
      2. If the target is reached, stop.
      3. Pick the highest ``size/cost`` ratio candidate.  Move it from
         ``remaining`` into ``evicted``.
      4. Cascade: every still-remaining candidate that depended on the
         just-evicted node now pays its cost on top.  Update its cost,
         recompute its ratio.
      5. If after picking the peak didn't go down, roll back and stop —
         the peak isn't activation-dominated (e.g. BERT bs=4/8, where it
         lives in the optimizer region).

    ``mem_limit`` defaults to ``peak_before − total_activation_bytes / 2``.
    Complexity is O(intermediates) ``simulate_peak`` calls and a per-step
    cascade update — fast enough for the largest model (~360 intermediates,
    ~9000 nodes).
    """
    inters = profiler.intermediates
    peak_before = simulate_peak(profiler, evicted=set())
    if not inters:
        return SelectionResult([], [], peak_before, peak_before, mem_limit,
                               reason="no activation candidates")

    if mem_limit is None:
        total_act = sum(i.size_bytes for i in inters)
        mem_limit = peak_before - total_act // 2

    # Per-candidate state for the cascading-cost model.
    intermediate_set = {i.node for i in inters}
    state: Dict[fx.Node, Dict] = {}
    for i in inters:
        ancestors = {a for a in i.node.all_input_nodes if a in intermediate_set}
        state[i.node] = {
            "size":      i.size_bytes,
            "cost":      i.recompute_ms,
            "ancestors": ancestors,
            "ratio":     i.size_bytes / (i.recompute_ms + 1e-9),
        }

    evicted:   Set[fx.Node]  = set()
    order:     List[fx.Node] = []
    remaining = set(intermediate_set)
    prev_peak = peak_before
    reason    = "memory target reached"

    while remaining:
        peak = simulate_peak(profiler, evicted)
        if peak <= mem_limit:
            break
        if order and peak >= prev_peak:
            # Last pick didn't help — peak isn't activation-dominated.
            # Roll back the no-op pick so the report doesn't include it.
            last = order.pop()
            evicted.discard(last)
            remaining.add(last)
            reason = "no remaining activation lowers the current peak"
            break
        prev_peak = peak

        best = max(remaining, key=lambda n: state[n]["ratio"])
        evicted.add(best)
        order.append(best)
        remaining.discard(best)

        # Cascade: any still-remaining candidate whose body included `best`
        # now pays best's cost too (and inherits best's own ancestors).
        best_state = state[best]
        for n in remaining:
            cand = state[n]
            if best in cand["ancestors"]:
                cand["ancestors"].discard(best)
                cand["ancestors"].update(best_state["ancestors"])
                cand["cost"]  += best_state["cost"]
                cand["ratio"]  = cand["size"] / (cand["cost"] + 1e-9)

    to_recompute, to_retain = validate_recompute_set(profiler, order)
    peak_after = simulate_peak(profiler, to_recompute)
    extra_ms   = estimate_recompute_ms(profiler, to_recompute)
    return SelectionResult(to_recompute, to_retain, peak_before, peak_after,
                           mem_limit, extra_ms, reason)


# --------------------------------------------------------------------------- #
# Phase 3 — graph rewriting                                                   #
# --------------------------------------------------------------------------- #

def first_backward_user(activation: fx.Node,
                        profiler: GraphProfiler) -> Optional[fx.Node]:
    """The earliest backward-region user of ``activation`` (the splice point
    for its recomputed copy)."""
    bwd_users = [(profiler.idx[u], u) for u in activation.users
                 if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd_users, default=(None, None))[1]


def rewrite_with_checkpointing(gm:           fx.GraphModule,
                               profiler:     GraphProfiler,
                               to_recompute: List[fx.Node]) -> fx.GraphModule:
    """For each evicted activation, splice a recomputed copy into the
    backward pass and redirect later backward consumers.

    Per evicted activation ``act``:

      1. Walk back from ``act`` to the boundary (placeholders + retained
         intermediates).  Collect everything between → the *body*.
      2. ``node_copy`` each body node into the main graph just before
         ``act``'s first backward use, building a per-iteration map
         original → copy.  The copy of ``act`` is the recomputed tensor.
      3. Replace ``act`` with the recomputed copy in every later backward
         user.

    Each evicted activation gets its own independent body — no sharing
    across evictions, matching the additive cost model the selector uses.
    """
    if not to_recompute:
        return gm

    recompute_set = set(to_recompute)
    retain_set    = {i.node for i in profiler.intermediates
                     if i.node not in recompute_set}

    for act in sorted(to_recompute, key=lambda n: profiler.idx[n]):
        first_bwd = first_backward_user(act, profiler)
        if first_bwd is None:
            continue

        body = recompute_body(profiler, act, retain_set)
        if not body:
            continue

        # Per-iteration map: original node → the just-inserted copy.
        # Boundary nodes aren't in this map, so arg_transform falls through
        # to the original (which is correct — boundary nodes are alive).
        copies: Dict[fx.Node, fx.Node] = {}
        with gm.graph.inserting_before(first_bwd):
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


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def print_ac_decisions(profiler: GraphProfiler,
                       result:   SelectionResult) -> None:
    """One-screen summary of a selection: per-node breakdown then totals."""
    inter_by_node = {i.node: i for i in profiler.intermediates}
    saved    = sum(inter_by_node[n].size_bytes for n in result.to_recompute)
    retained = sum(inter_by_node[n].size_bytes for n in result.to_retain)
    extra_ms = result.estimated_recompute_ms

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
    if result.mem_limit is not None:
        print(f"    Target peak:      {result.mem_limit / (1024**2):>8.2f} MB")
    print(f"    Peak before AC:   {result.peak_before  / (1024**2):>8.2f} MB")
    print(f"    Peak after  AC:   {result.peak_after   / (1024**2):>8.2f} MB")
    print(f"    Memory freed:     {result.freed_bytes  / (1024**2):>8.2f} MB")
    print(f"    Bytes recomputed: {saved    / (1024**2):>8.2f} MB")
    print(f"    Bytes retained:   {retained / (1024**2):>8.2f} MB")
    print(f"    Extra compute:    {extra_ms:>8.2f} ms")
    print(f"    Stop reason:      {result.reason}")
    print("=" * 80 + "\n")
