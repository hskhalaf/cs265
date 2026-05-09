from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.fx as fx
from graph_prof import GraphProfiler, NodeType, mutates_args


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
    def freed_bytes(self) -> int:
        return self.peak_before - self.peak_after


def recompute_body(profiler: GraphProfiler, activation: fx.Node, retained: Set[fx.Node]) -> List[fx.Node]:
    # this constructs the forward subgraph (in topological order) needed to rebuild activation
    boundary = {n for n in profiler.nodes if n.op == "placeholder"} | retained
    needed: Set[fx.Node] = set()
    stack = [activation]
    while stack:
        n = stack.pop()
        if n in needed or n in boundary:
            continue
        needed.add(n)
        stack.extend(n.all_input_nodes)
    return [n for n in activation.graph.nodes if n in needed]


def body_runtime_ms(profiler: GraphProfiler, body: Iterable[fx.Node]) -> float:
    return sum(profiler.avg_runtime_ms.get(n.name, 0.0) for n in body)


def estimate_recompute_ms(profiler: GraphProfiler, to_recompute: Iterable[fx.Node]) -> float:
    selected = set(to_recompute)
    if not selected: return 0.0
    retained = {i.node for i in profiler.intermediates} - selected
    return sum( body_runtime_ms(profiler, recompute_body(profiler, n, retained)) for n in sorted(selected, key=lambda n: profiler.idx[n]))


def first_bwd_use_idx(act: fx.Node, profiler: GraphProfiler) -> int:
    bwd = [profiler.idx[u] for u in act.users if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd) if bwd else -1


def simulate_timeline_by_role(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> Dict[NodeType, List[int]]:
    n = len(profiler.nodes)
    evicted = set(evicted)
    timeline: Dict[NodeType, List[int]] = {role: [0] * n for role in NodeType}

    for owner, aliases in profiler.aliases_by_owner.items():
        size = profiler.node_size_bytes.get(owner, 0)
        if size == 0:
            continue
        role = max(profiler.node_type[a] for a in aliases)
        if owner.op == "placeholder":
            for t in range(n):
                timeline[role][t] += size
            continue
        # Drop the backward window only when every retained alias is forward-only.
        fwd_only = (any(a in evicted for a in aliases)  and all(a in evicted or all( profiler.idx[u] < profiler.sep_bwd_idx for u in a.users if u in profiler.idx) for a in aliases))

        last = profiler.idx[owner]
        for alias in aliases:
            last = max(last, profiler.idx[alias])
            for u in alias.users:
                ui = profiler.idx.get(u, -1)
                if ui < 0 or (fwd_only and ui >= profiler.sep_bwd_idx):
                    continue
                last = max(last, ui)
        for t in range(profiler.idx[owner], min(last + 1, n)):
            timeline[role][t] += size

    if not evicted:
        return timeline


    retained = {i.node for i in profiler.intermediates} - evicted
    for act in evicted:
        first_idx = first_bwd_use_idx(act, profiler)
        if first_idx < 0:
            continue
        bwd_users = [profiler.idx[u] for u in act.users if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
        last_bwd = max(bwd_users) if bwd_users else first_idx

        for x in recompute_body(profiler, act, retained):
            size = profiler.node_size_bytes.get(x, 0)
            if size == 0:
                continue
            role = profiler.node_type.get(x, NodeType.OTHER)
            if x is act:
                for t in range(first_idx, min(last_bwd + 1, n)):
                    timeline[role][t] += size
            elif 0 <= first_idx < n:
                timeline[role][first_idx] += size

    return timeline


def simulate_peak(profiler: GraphProfiler, evicted: Iterable[fx.Node]) -> int:
    timeline = simulate_timeline_by_role(profiler, evicted)
    n = len(profiler.nodes)
    return max((sum(timeline[r][t] for r in NodeType) for t in range(n)), default=0)


def validate_recompute_set(profiler: GraphProfiler, selected_order: List[fx.Node]) -> Tuple[List[fx.Node], List[fx.Node]]:
    all_acts = {i.node for i in profiler.intermediates}
    selected = set(selected_order)

    def mutates_boundary(n: fx.Node, body_set: Set[fx.Node]) -> bool:
        # In-place op is OK iff its mutated input is a body-internal copy;
        # writing to a placeholder/retained tensor is what corrupts state.
        schema = getattr(n.target, "_schema", None)
        if schema is None:
            return False
        for i, arg in enumerate(schema.arguments):
            if arg.alias_info is None or not arg.alias_info.is_write:
                continue
            operand = n.args[i] if i < len(n.args) else n.kwargs.get(arg.name)
            if isinstance(operand, fx.Node) and operand not in body_set:
                return True
        return False

    def safe(node: fx.Node, retained: Set[fx.Node]) -> bool:
        body = recompute_body(profiler, node, retained)
        if not body:
            return False
        body_set = set(body)
        return all(
            n.op in {"call_function", "call_method", "call_module"}
            and profiler.idx[n] < profiler.sep_idx
            and not (n.op == "call_function" and mutates_boundary(n, body_set))
            for n in body
        )

    changed = True
    while changed:
        changed = False
        retained = all_acts - selected
        for node in selected_order:
            if node in selected and not safe(node, retained):
                selected.discard(node)
                changed = True

    to_recompute = [n for n in selected_order if n in selected]
    to_retain = sorted(all_acts - selected, key=lambda n: profiler.idx[n])
    return to_recompute, to_retain


def select_activations(profiler: GraphProfiler,mem_limit: Optional[int] = None) -> SelectionResult:
    inters = profiler.intermediates
    peak_before = simulate_peak(profiler, evicted=set())
    if not inters:
        return SelectionResult([], [], peak_before, peak_before, mem_limit, reason="no activation candidates")

    if mem_limit is None:
        mem_limit = peak_before - sum(i.size_bytes for i in inters) // 2

    all_acts = {i.node for i in inters}
    inter_by_node = {i.node: i for i in inters}
    evicted: Set[fx.Node] = set()
    order: List[fx.Node] = []
    remaining = set(all_acts)
    reason = "memory target reached"

    while remaining:
        if simulate_peak(profiler, evicted) <= mem_limit:
            break

        retained = all_acts - evicted
        scored = []
        for n in remaining:
            cost = body_runtime_ms(profiler, recompute_body(profiler, n, retained - {n}))
            if cost <= 0:
                cost = inter_by_node[n].recompute_ms
            if cost > 0:
                scored.append((inter_by_node[n].size_bytes / cost, n))

        if not scored:
            reason = "no candidate has a positive recomputation cost"
            break

        _, picked = max(scored)
        evicted.add(picked)
        order.append(picked)
        remaining.discard(picked)
    else:
        reason = "all candidates exhausted"

    to_recompute, to_retain = validate_recompute_set(profiler, order)
    peak_after = simulate_peak(profiler, to_recompute)
    dropped = len(order) - len(to_recompute)
    if dropped > 0 and peak_after > mem_limit:
        reason = f"validator dropped {dropped} picks; target not reached"
    return SelectionResult(to_recompute, to_retain, peak_before, peak_after, mem_limit, estimate_recompute_ms(profiler, to_recompute), reason)


def first_backward_user(activation: fx.Node, profiler: GraphProfiler) -> Optional[fx.Node]:
    bwd_users = [(profiler.idx[u], u) for u in activation.users if profiler.idx.get(u, -1) >= profiler.sep_bwd_idx]
    return min(bwd_users, default=(None, None))[1]


def rewrite_with_checkpointing(gm: fx.GraphModule, profiler: GraphProfiler, to_recompute: List[fx.Node]) -> fx.GraphModule:
    if not to_recompute:
        return gm

    recompute_set = set(to_recompute)
    retain_set = {i.node for i in profiler.intermediates if i.node not in recompute_set}

    for act in sorted(to_recompute, key=lambda n: profiler.idx[n]):
        first_bwd = first_backward_user(act, profiler)
        if first_bwd is None:
            continue
        body = recompute_body(profiler, act, retain_set)
        if not body:
            continue

        copies: Dict[fx.Node, fx.Node] = {}
        with gm.graph.inserting_before(first_bwd):
            for orig in body:
                copies[orig] = gm.graph.node_copy(
                    orig, arg_transform=lambda a: copies.get(a, a))

        new_act = copies[act]
        for user in list(act.users):
            if user is not new_act and profiler.idx.get(user, -1) >= profiler.sep_bwd_idx:
                user.replace_input_with(act, new_act)

    for n in list(gm.graph.nodes):
        if n.target == torch.ops.aten.detach.default and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()
    return gm


def print_ac_decisions(profiler: GraphProfiler, result: SelectionResult) -> None:
    inter_by_node = {i.node: i for i in profiler.intermediates}
    saved    = sum(inter_by_node[n].size_bytes for n in result.to_recompute)
    retained = sum(inter_by_node[n].size_bytes for n in result.to_retain)

    print("\n" + "=" * 80)
    print("ACTIVATION CHECKPOINTING DECISIONS  (µ-TWO greedy)")
    print("=" * 80)
    print(f"  Recomputed: {len(result.to_recompute)}    Retained: {len(result.to_retain)}")
    if result.mem_limit is not None:
        print(f"    Target peak:      {result.mem_limit / (1024**2):>8.2f} MB")
    print(f"    Peak before AC:   {result.peak_before  / (1024**2):>8.2f} MB")
    print(f"    Peak after  AC:   {result.peak_after   / (1024**2):>8.2f} MB")
    print(f"    Memory freed:     {result.freed_bytes  / (1024**2):>8.2f} MB")
    print(f"    Bytes recomputed: {saved    / (1024**2):>8.2f} MB")
    print(f"    Bytes retained:   {retained / (1024**2):>8.2f} MB")
    print(f"    Extra compute:    {result.estimated_recompute_ms:>8.2f} ms")
    print(f"    Stop reason:      {result.reason}")
    print("=" * 80 + "\n")
