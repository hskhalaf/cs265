from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.fx as fx
from graph_prof import GraphProfiler, NodeType, alias_in_schema


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
    """Total time added by the backward recomputations (independent model).

    Each evicted activation is recomputed in its own chain — bodies are
    NOT shared across evictions.  Cost is the sum of per-eviction body
    runtimes, matching what ``rewrite_with_checkpointing`` actually
    materializes.
    """
    selected = set(to_recompute)
    if not selected: return 0.0
    all_acts = {i.node for i in profiler.intermediates}
    retained = all_acts - selected
    return sum(
        body_runtime_ms(profiler, recompute_body(profiler, n, retained))
        for n in sorted(selected, key=lambda n: profiler.idx[n])
    )


# --------------------------------------------------------------------------- #
# Memory simulator                                                            #
# --------------------------------------------------------------------------- #

def first_bwd_use_idx(act: fx.Node, profiler: GraphProfiler) -> int:
    """Earliest backward-region user of ``act``, as an index.  -1 if none."""
    bwd = [profiler.idx[u] for u in act.users
           if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd) if bwd else -1


def simulate_timeline_by_role(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> Dict[NodeType, List[int]]:
    """Live bytes at each step, by role, modeling the *independent* AC rewrite.

    (A) Original tensors.  An evicted activation's storage is freed after
        its last forward use (no backward consumer in the rewritten graph).
        Everything else uses its normal lifetime.

    (B) Recomputed copies (per-eviction).  Each evicted activation's body
        is materialized as its own independent chain at the splice point.
        If two evictions share an ancestor X, X is materialized TWICE —
        matching what ``rewrite_with_checkpointing`` actually does (each
        evicted activation gets its own ``copies`` map).

        - The activation copy itself lives from its splice point through
          the last redirected backward user.
        - Other body intermediates are alive briefly during the chain;
          we approximate their contribution as a single-step spike at
          the splice point.
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

    # ----- (B) Recomputed copies (independent, per-eviction) -------------- #
    if not evicted:
        return timeline

    all_acts = {i.node for i in profiler.intermediates}
    retained = all_acts - evicted

    for act in evicted:
        first_idx = first_bwd_use_idx(act, profiler)
        if first_idx < 0:
            continue
        body = recompute_body(profiler, act, retained)
        # Activation copy lives from splice point through last redirected user.
        bwd_users = [profiler.idx[u] for u in act.users
                     if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
        last_bwd = max(bwd_users) if bwd_users else first_idx

        for x in body:
            size = profiler.node_size_bytes.get(x, 0)
            if size == 0:
                continue
            role = profiler.node_type.get(x, NodeType.OTHER)
            if x is act:
                # Recomputed activation: alive over redirected-user window.
                for t in range(first_idx, min(last_bwd + 1, n)):
                    timeline[role][t] += size
            else:
                # Body intermediate: alive briefly during the recomputation
                # chain — approximate as one spike at the splice point.
                if 0 <= first_idx < n:
                    timeline[role][first_idx] += size

    return timeline


def simulate_peak(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> int:
    timeline = simulate_timeline_by_role(profiler, evicted)
    n = len(profiler.nodes)
    if n == 0:
        return 0
    return max(sum(timeline[role][t] for role in NodeType) for t in range(n))


def validate_recompute_set(profiler: GraphProfiler, selected_order: List[fx.Node],) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Iterate to a fixed point: drop any selection whose body isn't safe to
    replay in backward.  Three conditions for body safety:

      * every body node is a forward call_function / call_method / call_module
        (so we can re-run it),
      * every body node sits in the forward region (idx < sep_idx),
      * **no body node is an in-place / view op** (alias_in_schema): replaying
        such an op during backward would mutate a retained-boundary tensor and
        corrupt the original forward state.  (Even idempotent ones like
        ``relu_`` are excluded for safety; demoting them costs little.)
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
                and not (n.op == "call_function" and alias_in_schema(n))
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
    """µ-TWO greedy selection with body-aware cost.

    The candidate's cost is the *full body runtime* needed to recompute it
    given the current retained set — not just the activation's own runtime.
    This matches what ``rewrite_with_checkpointing`` actually executes and
    what ``estimate_recompute_ms`` reports, so the three components agree.

    Per iteration:
      1. ``simulate_peak`` once to check the current peak.
      2. If target reached, stop.
      3. Score every remaining candidate by ``size / body_runtime_ms``,
         where the body is computed against the current retained set.
      4. Pick the highest-ratio candidate.
      5. If no candidate at all lowers the peak, the loop exits with
         the reason "no remaining activation lowers the current peak."

    The "walk down the ratio list until one helps" check guarantees
    every selected activation actually lowers the simulated peak — the
    selector never picks a candidate whose body window cancels its
    savings, and never bails after a single failed pick.

    Each iteration costs O(|remaining| × N) for the body computations
    plus up to O(|remaining| × |evicted| × N) for the simulate_peak
    trials in the worst case.  In practice the highest-ratio candidate
    helps most of the time and only one trial is needed.
    """
    inters = profiler.intermediates
    peak_before = simulate_peak(profiler, evicted=set())
    if not inters:
        return SelectionResult([], [], peak_before, peak_before, mem_limit, reason="no activation candidates")

    if mem_limit is None:
        total_act = sum(i.size_bytes for i in inters)
        mem_limit = peak_before - total_act // 2

    all_acts      = {i.node for i in inters}
    inter_by_node = {i.node: i for i in inters}
    evicted: Set[fx.Node]  = set()
    order:   List[fx.Node] = []
    remaining = set(all_acts)
    reason    = "memory target reached"

    while remaining:
        peak = simulate_peak(profiler, evicted)
        if peak <= mem_limit:
            break

        # Body-aware ratio for every remaining candidate, against the
        # current retained set.
        retained = all_acts - evicted
        scored: List[tuple[float, fx.Node]] = []
        for n in remaining:
            body = recompute_body(profiler, n, retained - {n})
            cost = body_runtime_ms(profiler, body)
            if cost <= 0:
                cost = inter_by_node[n].recompute_ms
                if cost <= 0:
                    continue
            scored.append((inter_by_node[n].size_bytes / cost, n))

        if not scored:
            reason = "no candidate has a positive recomputation cost"
            break

        # Walk highest-ratio first; the first candidate whose eviction
        # actually lowers the simulated peak wins.
        scored.sort(reverse=True)
        picked: Optional[fx.Node] = None
        for _, n in scored:
            if simulate_peak(profiler, evicted | {n}) < peak:
                picked = n
                break

        if picked is None:
            reason = "no remaining activation lowers the current peak"
            break

        evicted.add(picked)
        order.append(picked)
        remaining.discard(picked)

    to_recompute, to_retain = validate_recompute_set(profiler, order)
    peak_after = simulate_peak(profiler, to_recompute)
    extra_ms   = estimate_recompute_ms(profiler, to_recompute)
    return SelectionResult(to_recompute, to_retain, peak_before, peak_after, mem_limit, extra_ms, reason)


def first_backward_user(activation: fx.Node,  profiler: GraphProfiler) -> Optional[fx.Node]:
    bwd_users = [(profiler.idx[u], u) for u in activation.users if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd_users, default=(None, None))[1]


def rewrite_with_checkpointing(gm: fx.GraphModule, profiler: GraphProfiler,
                               to_recompute: List[fx.Node]) -> fx.GraphModule:
    """Splice independent recomputation chains into the backward pass.

    For each evicted activation, the body of nodes needed to rebuild it is
    extracted from the forward pass and a *fresh, independent copy* is
    inserted before the activation's first backward use.  No sharing
    across evictions — if two evictions share an ancestor, that ancestor
    is materialized once per evicted descendant.  This matches the
    deliverable's "extract subgraph and replicate" description and keeps
    the cost model (sum of per-eviction body runtimes) consistent.
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

        # Per-activation copies map (independent rewrite — fresh per `act`).
        # ``arg_transform`` rewires each new copy's inputs through this
        # map; boundary nodes fall through to the original (which is alive
        # at backward time because it's a placeholder or retained).
        copies: Dict[fx.Node, fx.Node] = {}
        with gm.graph.inserting_before(first_bwd):
            for orig in body:
                new = gm.graph.node_copy(
                    orig,
                    arg_transform=lambda a: copies.get(a, a),
                )
                copies[orig] = new

        # Redirect this act's backward consumers to its recomputed copy.
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
