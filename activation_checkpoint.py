from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.fx as fx
from graph_prof import GraphProfiler, NodeType


@dataclass
class SelectionResult:
    to_recompute: List[fx.Node]
    to_retain: List[fx.Node]
    peak_before: int
    peak_after: int
    mem_limit: Optional[int] = None
    estimated_recompute_ms: float = 0.0
    reason: str = ""
    @property
    def freed_bytes(self) -> int: return self.peak_before - self.peak_after


def recompute_body(profiler: GraphProfiler, activation: fx.Node, retained: Set[fx.Node]) -> List[fx.Node]:
    """
    This gets the forward subgraph (in topological order) needed to rebuild activation from the live tensors.
    """
    placeholders = {n for n in profiler.nodes if n.op == "placeholder"} # params or optimizer states
    boundary = placeholders | retained

    needed: Set[fx.Node] = set()
    stack: List[fx.Node] = [activation]
    while stack:
        n = stack.pop()
        if n in needed or n in boundary: continue
        needed.add(n)
        stack.extend(n.all_input_nodes)
    return [n for n in activation.graph.nodes if n in needed]


def body_runtime_ms(profiler: GraphProfiler, body: Iterable[fx.Node]) -> float:
    return sum(profiler.avg_runtime_ms.get(n.name, 0.0) for n in body)


def estimate_recompute_ms(profiler: GraphProfiler, to_recompute: Iterable[fx.Node]) -> float:
    """Total time added by the backward recomputations (shared model).

    Each unique node appearing in any eviction's body is computed exactly
    once in the backward pass — so total cost is the runtime of the
    *union* of all bodies, not the sum across bodies.
    """
    selected = set(to_recompute)
    if not selected: return 0.0
    all_acts = {i.node for i in profiler.intermediates}
    retained = all_acts - selected
    union_body: Set[fx.Node] = set()
    for act in selected:
        union_body |= set(recompute_body(profiler, act, retained))
    return body_runtime_ms(profiler, union_body)


# --------------------------------------------------------------------------- #
# Memory simulator                                                            #
# --------------------------------------------------------------------------- #

def first_bwd_use_idx(act: fx.Node, profiler: GraphProfiler) -> int:
    """Earliest backward-region user of ``act``, as an index.  -1 if none."""
    bwd = [profiler.idx[u] for u in act.users
           if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd) if bwd else -1


def simulate_timeline_by_role(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> Dict[NodeType, List[int]]:
    """Live bytes at each step, by role, modeling the *shared* AC rewrite.

    Two contributions per step:

    (A) Original tensors.  An evicted activation's storage is freed after
        its last forward use (no backward consumer in the rewritten graph).
        Everything else uses its normal lifetime.

    (B) Recomputed copies.  Each unique body node X (across all evictions
        whose recomputation chain includes X) is materialized ONCE in the
        backward pass at the splice point of the earliest eviction needing
        it, and stays alive through the latest eviction needing it (plus
        the redirected backward users of X if X is itself an eviction).
    """
    n = len(profiler.nodes)
    evicted = set(evicted)
    timeline: Dict[NodeType, List[int]] = {role: [0] * n for role in NodeType}

    # ----- (A) Original tensors with possibly-truncated lifetime ---------- #
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
            # Truncated to forward window only.
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
        else:
            last_use = produced
            for alias in aliases:
                last_use = max(last_use, profiler.idx[alias])
                user_idx = [profiler.idx[u] for u in alias.users
                            if u in profiler.idx]
                if user_idx:
                    last_use = max(last_use, max(user_idx))
            for t in range(produced, min(last_use + 1, n)):
                timeline[role][t] += size

    # ----- (B) Recomputed copies (shared) --------------------------------- #
    if not evicted:
        return timeline

    all_acts = {i.node for i in profiler.intermediates}
    retained = all_acts - evicted

    # For each unique body node, find its lifetime window in backward.
    # Window = [earliest first-bwd-use of an eviction needing it,
    #           latest first-bwd-use of an eviction needing it].
    # For evictions that ARE the body node, also extend to their latest
    # redirected backward user.
    body_window: Dict[fx.Node, Tuple[int, int]] = {}
    for act in evicted:
        first_idx = first_bwd_use_idx(act, profiler)
        if first_idx < 0:
            continue
        body = recompute_body(profiler, act, retained)
        for x in body:
            lo, hi = body_window.get(x, (first_idx, first_idx))
            body_window[x] = (min(lo, first_idx), max(hi, first_idx))

    # Extend evicted activations' windows to their last redirected user.
    for act in evicted:
        if act not in body_window:
            continue
        bwd_users = [profiler.idx[u] for u in act.users
                     if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
        if bwd_users:
            lo, hi = body_window[act]
            body_window[act] = (lo, max(hi, max(bwd_users)))

    for x, (lo, hi) in body_window.items():
        size = profiler.node_size_bytes.get(x, 0)
        if size == 0:
            continue
        role = profiler.node_type.get(x, NodeType.OTHER)
        for t in range(lo, min(hi + 1, n)):
            timeline[role][t] += size

    return timeline


def simulate_peak(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> int:
    timeline = simulate_timeline_by_role(profiler, evicted)
    n = len(profiler.nodes)
    if n == 0:
        return 0
    return max(sum(timeline[role][t] for role in NodeType) for t in range(n))


def validate_recompute_set(profiler: GraphProfiler, selected_order: List[fx.Node],) -> Tuple[List[fx.Node], List[fx.Node]]:
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


def select_activations(profiler: GraphProfiler,
                       mem_limit: Optional[int] = None) -> SelectionResult:
    inters = profiler.intermediates
    peak_before = simulate_peak(profiler, evicted=set())
    if not inters:
        return SelectionResult([], [], peak_before, peak_before, mem_limit, reason="no activation candidates")

    if mem_limit is None:
        total_act = sum(i.size_bytes for i in inters)
        mem_limit = peak_before - total_act // 2

    intermediate_set = {i.node for i in inters}
    state: Dict[fx.Node, Dict] = {}
    for i in inters:
        ancestors = {a for a in i.node.all_input_nodes if a in intermediate_set}
        state[i.node] = {"size": i.size_bytes, "cost":i.recompute_ms, "ancestors": ancestors, "ratio": i.size_bytes / (i.recompute_ms + 1e-9),}

    evicted: Set[fx.Node]  = set()
    order: List[fx.Node] = []
    remaining = set(intermediate_set)
    prev_peak = peak_before
    reason = "memory target reached"

    while remaining:
        peak = simulate_peak(profiler, evicted)
        if peak <= mem_limit:
            break
        if order and peak >= prev_peak:
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
        best_state = state[best]
        for n in remaining:
            cand = state[n]
            if best in cand["ancestors"]:
                cand["ancestors"].discard(best)
                cand["ancestors"].update(best_state["ancestors"])
                cand["cost"] += best_state["cost"]
                cand["ratio"] = cand["size"] / (cand["cost"] + 1e-9)

    to_recompute, to_retain = validate_recompute_set(profiler, order)
    peak_after = simulate_peak(profiler, to_recompute)
    extra_ms   = estimate_recompute_ms(profiler, to_recompute)
    return SelectionResult(to_recompute, to_retain, peak_before, peak_after, mem_limit, extra_ms, reason)


def first_backward_user(activation: fx.Node,  profiler: GraphProfiler) -> Optional[fx.Node]:
    bwd_users = [(profiler.idx[u], u) for u in activation.users if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd_users, default=(None, None))[1]


def rewrite_with_checkpointing(gm: fx.GraphModule, profiler: GraphProfiler,
                               to_recompute: List[fx.Node]) -> fx.GraphModule:
    """Splice shared recomputation chains into the backward pass.

    For each evicted activation, the body of nodes needed to rebuild it is
    inserted before its first backward use.  Crucially, ``copies`` is a
    *single global map* across all evictions — if two evictions share an
    ancestor, that ancestor is materialized exactly once (at the earlier
    splice point).  Each evicted activation's backward consumers are
    redirected to the (possibly shared) copy.
    """
    if not to_recompute:
        return gm

    recompute_set = set(to_recompute)
    retain_set    = {i.node for i in profiler.intermediates
                     if i.node not in recompute_set}

    # Global map: original FX node -> its single recomputed copy.
    copies: Dict[fx.Node, fx.Node] = {}

    # Process evictions in order of their first backward use, so the
    # earliest need for a body node decides its insertion point.  Later
    # evictions that share that node simply reuse the copy.
    sorted_evicts = sorted(
        to_recompute,
        key=lambda n: (first_bwd_use_idx(n, profiler), profiler.idx[n]),
    )

    for act in sorted_evicts:
        first_bwd = first_backward_user(act, profiler)
        if first_bwd is None:
            continue
        body = recompute_body(profiler, act, retain_set)
        if not body:
            continue

        # Insert the body nodes that don't already have a copy, in topo
        # order, just before this activation's first backward use.
        # ``arg_transform`` rewires each new copy's inputs through the
        # global ``copies`` map (boundary nodes fall through to themselves).
        with gm.graph.inserting_before(first_bwd):
            for orig in body:
                if orig in copies:
                    continue  # shared with an earlier eviction's chain
                new = gm.graph.node_copy(
                    orig,
                    arg_transform=lambda a: copies.get(a, a),
                )
                copies[orig] = new

        # Redirect this act's backward consumers to the (possibly shared) copy.
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
    print(f"  Recomputed: {len(result.to_recompute)}    "
          f"Retained: {len(result.to_retain)}")
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
