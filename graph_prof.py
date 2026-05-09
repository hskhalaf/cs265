"""
Phase 1 — Graph Profiler.

Subclasses ``fx.Interpreter`` to walk a traced training-step ``GraphModule``
and produce, for one iteration:

  per node, static (computed from the FX graph alone — no execution)
    - region: FORWARD / LOSS / BACKWARD / OPTIMIZER
    - role:   PARAM / ACT / GRAD / OTHER
    - logical bytes vs. bytes actually allocated
    - storage owner: the node that owns the underlying GPU memory

  per node, runtime (timed across measurement runs)
    - execution time (CUDA-event timed) — needed for AC's recompute-cost
      estimate in Phase 2.

  per iteration
    - the set of intermediate activations (forward outputs read in backward)
      and their lifetimes — what AC selects between in Phase 2
    - the live-memory timeline by role over the iteration

Storage ownership is the central trick.  Several distinct FX nodes can share
a single GPU allocation (in-place / view ops via schema ``alias_info``;
``operator.getitem`` unpacking multi-output ops; placeholders that persist
across iterations).  ``_compute_aliases`` collapses these into one *owner*
per node; ``memory_timeline_by_role`` walks once per owner so each tensor's
bytes are counted exactly once over the right interval.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
import torch.fx as fx


# --------------------------------------------------------------------------- #
# Enums and data classes                                                      #
# --------------------------------------------------------------------------- #

class OP(str, Enum):
    """Mirrors fx.Node.op values."""
    CALL_FUNCTION = "call_function"
    CALL_MODULE   = "call_module"
    CALL_METHOD   = "call_method"
    GET_ATTR      = "get_attr"
    OUTPUT        = "output"
    PLACEHOLDER   = "placeholder"


class NodeType(IntEnum):
    """Tensor role.  Values double as merge priority (PARAM > GRAD > ACT >
    OTHER): when an owner's aliases disagree on role, the highest one wins.
    OTHER is the catch-all (opt state, foreach scratch, activation
    gradients, runtime workspace residuals)."""
    OTHER = 0
    ACT   = 1
    GRAD  = 2
    PARAM = 3


class Region(Enum):
    FORWARD   = "forward"
    LOSS      = "loss"
    BACKWARD  = "backward"
    OPTIMIZER = "optimizer"


@dataclass
class Intermediate:
    """A forward-pass activation that's consumed in the backward pass."""
    node: fx.Node
    size_bytes: int = 0
    last_fwd_idx: int = -1
    first_bwd_idx: int = -1
    recompute_ms: float = 0.0


# Display order for the per-role table in print_summary.
_DISPLAY_ROLES = (NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER)


# --------------------------------------------------------------------------- #
# Schema / aliasing helpers                                                   #
# --------------------------------------------------------------------------- #

def _iter_tensors(val: Any) -> Iterable[torch.Tensor]:
    if isinstance(val, torch.Tensor):
        yield val
    elif isinstance(val, (list, tuple)):
        for v in val:
            yield from _iter_tensors(v)


def _alias_in_schema(node: fx.Node) -> bool:
    """True if any schema return aliases any argument (in-place / view ops:
    ``relu_``, ``view``, ``aten.split``, ``_foreach_add_``, ``_fused_adam``)."""
    schema = getattr(node.target, "_schema", None)
    return (schema is not None
            and any(r.alias_info is not None for r in schema.returns))


def _is_getitem(node: fx.Node) -> bool:
    return (node.op == OP.CALL_FUNCTION
            and (node.target is operator.getitem
                 or getattr(node.target, "__name__", None) == "getitem"))


def _is_container(node: fx.Node) -> bool:
    """Output is a list/tuple containing at least one tensor."""
    val = node.meta.get("val")
    return isinstance(val, (list, tuple)) and any(_iter_tensors(val))


def _is_getitem_of_container(node: fx.Node) -> bool:
    return (_is_getitem(node)
            and bool(node.all_input_nodes)
            and _is_container(node.all_input_nodes[0]))


def _output_bytes(node: fx.Node) -> int:
    """Logical bytes in this node's output, summed across all returned tensors."""
    if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
        return 0
    return sum(t.numel() * t.element_size()
               for t in _iter_tensors(node.meta.get("val")))


def _allocation_bytes(node: fx.Node) -> int:
    """Bytes of NEW storage allocated by this node.

    Three multi-output cases:
      - aliasing parent (split, foreach in-place): both parent and getitems
        contribute 0; the chain points back to the inputs.
      - independent + only-getitem users (conv_backward): parent contributes
        0, each getitem owns one element so per-element role attribution
        (GRAD vs OTHER) survives.
      - independent + mixed users: parent owns full bytes (rare).
    """
    if _is_getitem_of_container(node):
        parent = node.all_input_nodes[0]
        if parent.op == OP.CALL_FUNCTION and _alias_in_schema(parent):
            return 0
        return _output_bytes(node)
    if node.op != OP.CALL_FUNCTION:
        return _output_bytes(node)
    if _alias_in_schema(node):
        return 0
    if _is_container(node) and all(_is_getitem(u) for u in node.users):
        return 0
    return _output_bytes(node)


def _alias_target(parent: fx.Node, getitem: fx.Node) -> Optional[fx.Node]:
    """For ``getitem(parent, i)`` where parent's outputs alias parent's
    inputs, return the input whose storage backs ``output[i]``:
      - single-input multi-output (split, unbind): all outputs view that input
      - list-input multi-output in-place (foreach): output[i] aliases args[0][i]
    """
    if len(getitem.args) < 2 or not parent.args:
        return None
    first = parent.args[0]
    if isinstance(first, fx.Node):
        return first
    idx = getitem.args[1]
    if (isinstance(first, (list, tuple))
            and isinstance(idx, int)
            and 0 <= idx < len(first)
            and isinstance(first[idx], fx.Node)):
        return first[idx]
    return None


# --------------------------------------------------------------------------- #
# GraphProfiler                                                               #
# --------------------------------------------------------------------------- #

class GraphProfiler(fx.Interpreter):

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.nodes: List[fx.Node] = list(self.module.graph.nodes)
        self.idx:   Dict[fx.Node, int] = {n: i for i, n in enumerate(self.nodes)}

        # Static analysis (each step depends on the previous).
        self._find_separators()
        self._assign_regions()
        self._classify_tensors()
        self._find_intermediates()
        self._compute_sizes()
        self._compute_aliases()

        self.reset_stats()

    # ----- static analysis -------------------------------------------------- #

    def _find_separators(self) -> None:
        """Locate %sep, %sep_backward, and the start of the optimizer step."""
        self.sep_idx = self.sep_bwd_idx = self.opt_idx = -1
        first_foreach_after_bwd = -1
        for i, n in enumerate(self.nodes):
            if n.op != OP.CALL_FUNCTION:
                continue
            t = n.target
            if   t == torch.ops.separator.sep.default:           self.sep_idx     = i
            elif t == torch.ops.separator.sep_backward.default:  self.sep_bwd_idx = i
            elif t == torch.ops.aten._fused_adam.default:        self.opt_idx     = i
            elif (first_foreach_after_bwd < 0
                    and self.sep_bwd_idx >= 0
                    and "_foreach_" in str(t)):
                first_foreach_after_bwd = i
        if self.opt_idx < 0:
            self.opt_idx = first_foreach_after_bwd
        assert self.sep_idx >= 0,     "no %sep marker in graph"
        assert self.sep_bwd_idx >= 0, "no %sep_backward marker in graph"

    def _assign_regions(self) -> None:
        self.region: Dict[fx.Node, Region] = {}
        for i, n in enumerate(self.nodes):
            if   0 <= self.opt_idx <= i: self.region[n] = Region.OPTIMIZER
            elif i >= self.sep_bwd_idx:  self.region[n] = Region.BACKWARD
            elif i >  self.sep_idx:      self.region[n] = Region.LOSS
            else:                        self.region[n] = Region.FORWARD

    def _classify_tensors(self) -> None:
        """Assign every node one of {PARAM, GRAD, OTHER} (ACT comes later).

        Fused-Adam path: take params/grads directly from the op's args.
        Foreach path: PARAM = a placeholder used in both forward and
        optimizer; GRAD = a backward call_function whose output flows
        directly into an optimizer op.
        """
        self.params: Set[fx.Node] = set()
        self.grads:  Set[fx.Node] = set()

        fused = next((n for n in self.nodes
                      if n.op == OP.CALL_FUNCTION
                      and n.target == torch.ops.aten._fused_adam.default), None)
        if fused is not None:
            self.params = set(fused.args[0])
            self.grads  = set(fused.args[1])
        elif self.opt_idx >= 0:
            for n in self.nodes:
                if n.op == OP.PLACEHOLDER:
                    user_idxs = [self.idx[u] for u in n.users if u in self.idx]
                    if (any(j <  self.sep_idx for j in user_idxs)
                            and any(j >= self.opt_idx for j in user_idxs)):
                        self.params.add(n)
                elif (n.op == OP.CALL_FUNCTION
                        and self.sep_bwd_idx <= self.idx[n] < self.opt_idx
                        and any(self.idx[u] >= self.opt_idx
                                for u in n.users if u in self.idx)):
                    self.grads.add(n)

        def role(n: fx.Node) -> NodeType:
            if n in self.params: return NodeType.PARAM
            if n in self.grads:  return NodeType.GRAD
            return NodeType.OTHER
        self.node_type: Dict[fx.Node, NodeType] = {n: role(n) for n in self.nodes}

    def _find_intermediates(self) -> None:
        """Forward call_function outputs that are read in the backward pass."""
        self.intermediates: List[Intermediate] = []
        for n in self.nodes:
            i = self.idx[n]
            if i >= self.sep_idx or n.op != OP.CALL_FUNCTION or n in self.params:
                continue
            user_idxs = [self.idx.get(u, -1) for u in n.users]
            if not any(j >= self.sep_bwd_idx for j in user_idxs):
                continue
            self.intermediates.append(Intermediate(
                node=n,
                last_fwd_idx=max((j for j in user_idxs if j <  self.sep_idx), default=i),
                first_bwd_idx=min((j for j in user_idxs if j >= self.sep_bwd_idx), default=-1),
            ))
            self.node_type[n] = NodeType.ACT

    def _compute_sizes(self) -> None:
        self.node_logical_size_bytes: Dict[fx.Node, int] = {
            n: _output_bytes(n) for n in self.nodes
        }
        self.node_size_bytes: Dict[fx.Node, int] = {
            n: _allocation_bytes(n) for n in self.nodes
        }
        for inter in self.intermediates:
            inter.size_bytes = self.node_logical_size_bytes[inter.node]

    def _compute_aliases(self) -> None:
        """Map each node to the node that owns its underlying GPU storage."""
        self.storage_owner:    Dict[fx.Node, fx.Node]       = {}
        self.aliases_by_owner: Dict[fx.Node, List[fx.Node]] = defaultdict(list)

        for n in self.nodes:
            owner = n
            if _is_getitem_of_container(n):
                # Aliasing parent → chain to the right input element.
                # Independent parent → leave each getitem as its own owner so
                # per-element role attribution (GRAD vs OTHER) is preserved.
                parent = n.all_input_nodes[0]
                if parent.op == OP.CALL_FUNCTION and _alias_in_schema(parent):
                    target = _alias_target(parent, n)
                    if target is not None and self.node_logical_size_bytes.get(target, 0) > 0:
                        owner = self.storage_owner.get(target, target)
            elif n.op == OP.CALL_FUNCTION and _alias_in_schema(n):
                # In-place / view ops chain through their first sized input.
                # Works for single-output (relu_, view) and multi-output
                # (foreach in-place, _fused_adam) alike.
                for inp in n.all_input_nodes:
                    if self.node_logical_size_bytes.get(inp, 0) > 0:
                        owner = self.storage_owner.get(inp, inp)
                        break
            self.storage_owner[n] = owner
            self.aliases_by_owner[owner].append(n)

    # ----- runtime measurement --------------------------------------------- #

    def reset_stats(self) -> None:
        """Clear per-iteration timing state.  Called between warm-up and
        measurement runs, and at construction."""
        self._runtimes_ms:  Dict[str, List[float]] = defaultdict(list)
        self._latencies_ms: List[float]            = []
        self.avg_runtime_ms:      Dict[str, float] = {}
        self.avg_iter_latency_ms: float            = 0.0

    def run(self, *args, **kwargs) -> Any:
        """Time the whole iteration in addition to per-node timing."""
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self._latencies_ms.append(start.elapsed_time(end))
        return result

    def run_node(self, n: fx.Node) -> Any:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.synchronize()
        self._runtimes_ms[n.name].append(start.elapsed_time(end))
        return result

    def aggregate_stats(self) -> None:
        """Average per-node and per-iteration timings across runs."""
        self.avg_runtime_ms = {name: sum(vs) / len(vs)
                               for name, vs in self._runtimes_ms.items()}
        for inter in self.intermediates:
            inter.recompute_ms = self.avg_runtime_ms.get(inter.node.name, 0.0)
        if self._latencies_ms:
            self.avg_iter_latency_ms = sum(self._latencies_ms) / len(self._latencies_ms)

    # ----- queries --------------------------------------------------------- #

    def memory_timeline_by_role(self) -> Dict[NodeType, List[int]]:
        """Live bytes at each step, broken down by tensor role.

        A storage owner contributes its size to every step from its
        production index through its last-use index, inclusive.
        Placeholders are special-cased to be live for the whole iteration
        (they exist before step 0 and persist after step N-1).  The role
        is the highest-priority role across the owner and its aliases.
        """
        n = len(self.nodes)
        timeline: Dict[NodeType, List[int]] = {nt: [0] * n for nt in NodeType}

        for owner, aliases in self.aliases_by_owner.items():
            size = self.node_size_bytes.get(owner, 0)
            if size == 0:
                continue
            role = max(self.node_type[a] for a in aliases)
            if owner.op == OP.PLACEHOLDER:
                lo, hi = 0, n - 1
            else:
                lo = self.idx[owner]
                hi = lo
                for a in aliases:
                    hi = max(hi, self.idx[a],
                             *(self.idx[u] for u in a.users if u in self.idx))
            for t in range(lo, min(hi + 1, n)):
                timeline[role][t] += size
        return timeline

    def peak_memory_bytes(self) -> int:
        n = len(self.nodes)
        if n == 0:
            return 0
        roles = self.memory_timeline_by_role()
        return max(sum(roles[nt][t] for nt in NodeType) for t in range(n))

    def iteration_latency_ms(self) -> float:
        return self.avg_iter_latency_ms

    # ----- pretty printing ------------------------------------------------- #

    def print_summary(self) -> None:
        """One-screen summary: static peak by role + iteration latency."""
        timeline = self.memory_timeline_by_role()
        n = len(self.nodes)
        per_step = [sum(timeline[nt][t] for nt in NodeType) for t in range(n)]
        peak_step = max(range(n), key=per_step.__getitem__) if n else 0

        nodes_by_role:   Dict[NodeType, int] = defaultdict(int)
        nodes_by_region: Dict[Region, int]   = defaultdict(int)
        for nt in self.node_type.values():   nodes_by_role[nt]   += 1
        for r  in self.region.values():      nodes_by_region[r]  += 1

        print(f"  Nodes: {n}  "
              f"(forward {nodes_by_region[Region.FORWARD]}, "
              f"backward {nodes_by_region[Region.BACKWARD]}, "
              f"optimizer {nodes_by_region[Region.OPTIMIZER]})")
        print(f"  Intermediates: {len(self.intermediates)}\n")
        print(f"  Static peak (step {peak_step}):")
        print(f"    {'Role':<8} {'Nodes':>6} {'Bytes':>10}")
        for nt in _DISPLAY_ROLES:
            print(f"    {nt.name:<8} {nodes_by_role[nt]:>6}"
                  f" {timeline[nt][peak_step] / 1024**2:>7.2f} MB")
        print(f"    {'TOTAL':<8} {n:>6} {per_step[peak_step] / 1024**2:>7.2f} MB\n")
        print(f"  Iteration latency: {self.avg_iter_latency_ms:>7.2f} ms"
              f"  (avg of {len(self._latencies_ms)} runs)")
