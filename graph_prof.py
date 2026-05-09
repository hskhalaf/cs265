"""
Phase 1 — Graph Profiler.

Subclasses ``fx.Interpreter`` to walk a traced training-step ``GraphModule``
and produce, for one iteration:

  static, per node (no execution required)
    - region:        FORWARD / LOSS / BACKWARD / OPTIMIZER
    - role:          PARAM / ACT / GRAD / OPT_STATE / OTHER
    - logical bytes / bytes actually allocated
    - storage owner: the node that owns the underlying GPU memory
  static, per iteration
    - intermediate activations (forward outputs read in backward) and
      their lifetimes — what AC selects between in Phase 2
    - the live-memory timeline by role over the iteration

  runtime, per node
    - execution time, CUDA-event timed (Phase 2 needs this for cost)
    - memory allocated after the op
    - per-op allocation delta
    - per-op peak allocation while the op ran

The central trick is *storage ownership*.  Distinct FX nodes can share a
single GPU allocation: in-place / view ops via schema ``alias_info``,
``operator.getitem`` unpacking multi-output ops, and placeholders that
persist across iterations.  ``compute_aliases`` collapses each into a
single owner, and ``memory_timeline_by_role`` walks once per owner so
each tensor's bytes are counted exactly once over the right interval.
"""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
import torch.fx as fx


# --------------------------------------------------------------------------- #
# Enums and data classes                                                      #
# --------------------------------------------------------------------------- #

class NodeType(IntEnum):
    """Tensor role.  Values double as merge priority — when an owner's
    aliases disagree, the highest one wins
    (PARAM > GRAD > ACT > OPT_STATE > OTHER).  OTHER is the catch-all
    (inputs, foreach scratch, activation gradients, constants, anything
    not first-class)."""
    OTHER = 0
    OPT_STATE = 1
    ACT   = 2
    GRAD  = 3
    PARAM = 4


class Region(Enum):
    FORWARD   = "forward"
    LOSS      = "loss"
    BACKWARD  = "backward"
    OPTIMIZER = "optimizer"


@dataclass
class Intermediate:
    """A forward-pass activation consumed in the backward pass."""
    node:          fx.Node
    size_bytes:    int   = 0
    last_fwd_idx:  int   = -1
    first_bwd_idx: int   = -1
    recompute_ms:  float = 0.0


# Display order for the per-role table in print_summary.
DISPLAY_ROLES = (
    NodeType.PARAM,
    NodeType.OPT_STATE,
    NodeType.ACT,
    NodeType.GRAD,
    NodeType.OTHER,
)


# --------------------------------------------------------------------------- #
# Schema / aliasing helpers                                                   #
# --------------------------------------------------------------------------- #

def iter_tensors(val: Any) -> Iterable[torch.Tensor]:
    """Yield every Tensor inside ``val``, descending into list/tuple."""
    if isinstance(val, torch.Tensor):
        yield val
    elif isinstance(val, (list, tuple)):
        for v in val:
            yield from iter_tensors(v)


def alias_in_schema(node: fx.Node) -> bool:
    """True if any of the op's schema returns aliases an input.  This
    detects in-place / view ops (``relu_``, ``view``, ``aten.split``,
    ``_foreach_add_``, ``_fused_adam``)."""
    schema = getattr(node.target, "_schema", None)
    return schema is not None and any(r.alias_info is not None
                                      for r in schema.returns)


def is_getitem(node: fx.Node) -> bool:
    """True if this is an ``operator.getitem`` call."""
    return (node.op == "call_function"
            and (node.target is operator.getitem
                 or getattr(node.target, "__name__", None) == "getitem"))


def is_container(node: fx.Node) -> bool:
    """True if the node's output is a list/tuple containing >=1 Tensor.
    NOTE: uses ``next(...)`` instead of ``any(...)`` because ``any`` would
    coerce each tensor to a bool, which FakeTensor refuses to evaluate."""
    val = node.meta.get("val")
    return (isinstance(val, (list, tuple))
            and next(iter_tensors(val), None) is not None)


def is_getitem_of_container(node: fx.Node) -> bool:
    """``getitem(container, i)`` where container's output is a list/tuple
    of tensors (`_foreach_*` outputs unpacked, `aten.split` outputs, etc)."""
    return (is_getitem(node)
            and bool(node.all_input_nodes)
            and is_container(node.all_input_nodes[0]))


def output_bytes(node: fx.Node) -> int:
    """Sum of bytes across all tensors in this node's output (logical
    size, ignoring whether the bytes are newly allocated or aliased)."""
    if node.op not in ("call_function", "placeholder"):
        return 0
    return sum(t.numel() * t.element_size()
               for t in iter_tensors(node.meta.get("val")))


def allocation_bytes(node: fx.Node) -> int:
    """Bytes of *new* storage allocated by this node.  Returns 0 for any
    node whose bytes are owned by some other node:

      - in-place / view ops (alias_info on schema)
      - multi-output ops fully unpacked by getitems (bytes are on the
        getitems so each can carry its own role)
      - getitem of an aliasing parent (just a view of an input element)
    """
    if node.op == "call_function":
        if alias_in_schema(node):
            return 0
        if is_container(node) and all(is_getitem(u) for u in node.users):
            return 0
    if is_getitem_of_container(node):
        parent = node.all_input_nodes[0]
        if parent.op == "call_function" and alias_in_schema(parent):
            return 0
    return output_bytes(node)


def alias_target(parent: fx.Node, getitem: fx.Node) -> Optional[fx.Node]:
    """For ``getitem(parent, i)`` where parent's outputs alias parent's
    inputs, return the input whose storage backs ``output[i]``.  Two
    common patterns:

      single-input multi-output (``aten.split``, ``unbind``, ``chunk``):
          all outputs view the same input — return ``args[0]``.
      list-input multi-output in-place (``_foreach_mul_(a_list, ...)``):
          ``output[i]`` aliases ``args[0][i]``.
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

        # Static analysis (each pass depends on the previous).
        self.find_separators()
        self.assign_regions()
        self.classify_tensors()
        self.find_intermediates()
        self.compute_sizes()
        self.compute_aliases()

        self.reset_stats()

    # ----- static analysis -------------------------------------------------- #

    def find_separators(self) -> None:
        """Locate the indices of the sep / sep_backward markers and the
        first optimizer op (either fused-Adam or the first foreach op
        after sep_backward)."""
        self.sep_idx = self.sep_bwd_idx = self.opt_idx = -1
        first_foreach_after_bwd = -1
        for i, n in enumerate(self.nodes):
            if n.op != "call_function":
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
        assert self.sep_idx     >= 0, "no %sep marker in graph"
        assert self.sep_bwd_idx >= 0, "no %sep_backward marker in graph"

    def assign_regions(self) -> None:
        """Label every node by which phase of the iteration it belongs to."""
        self.region: Dict[fx.Node, Region] = {}
        for i, n in enumerate(self.nodes):
            if   0 <= self.opt_idx <= i: self.region[n] = Region.OPTIMIZER
            elif i >= self.sep_bwd_idx:  self.region[n] = Region.BACKWARD
            elif i >  self.sep_idx:      self.region[n] = Region.LOSS
            else:                        self.region[n] = Region.FORWARD

    def classify_tensors(self) -> None:
        """Assign every node one of {PARAM, GRAD, OPT_STATE, OTHER};
        ACT is filled in by ``find_intermediates`` next.

        Fused-Adam path: read params and grads directly from the op args.
        Foreach path: PARAM = a placeholder used in *both* forward and
        optimizer; GRAD = a backward call_function whose output flows
        directly into an optimizer op; OPT_STATE = a placeholder used by
        optimizer only (Adam moments and step counters).
        """
        self.params: Set[fx.Node] = set()
        self.grads:  Set[fx.Node] = set()
        self.opt_states: Set[fx.Node] = set()

        fused = next((n for n in self.nodes
                      if n.op == "call_function"
                      and n.target == torch.ops.aten._fused_adam.default), None)
        if fused is not None:
            self.params = set(fused.args[0])
            self.grads  = set(fused.args[1])
            for arg in fused.args[2:]:
                if isinstance(arg, (list, tuple)):
                    self.opt_states.update(x for x in arg
                                           if isinstance(x, fx.Node))
        elif self.opt_idx >= 0:
            for n in self.nodes:
                if n.op == "placeholder":
                    user_idxs = [self.idx[u] for u in n.users if u in self.idx]
                    in_fwd = any(j < self.sep_idx for j in user_idxs)
                    in_opt = any(j >= self.opt_idx for j in user_idxs)
                    if in_fwd and in_opt:
                        self.params.add(n)
                    elif in_opt:
                        self.opt_states.add(n)
                elif (n.op == "call_function"
                        and self.sep_bwd_idx <= self.idx[n] < self.opt_idx
                        and any(self.idx[u] >= self.opt_idx
                                for u in n.users if u in self.idx)):
                    self.grads.add(n)

        self.node_type: Dict[fx.Node, NodeType] = {
            n: (NodeType.PARAM if n in self.params else
                NodeType.GRAD  if n in self.grads  else
                NodeType.OPT_STATE if n in self.opt_states else
                NodeType.OTHER)
            for n in self.nodes
        }

    def find_intermediates(self) -> None:
        """Forward call_function outputs that are read in the backward
        pass — these are the activations Phase 2 chooses between."""
        self.intermediates: List[Intermediate] = []
        for n in self.nodes:
            i = self.idx[n]
            if i >= self.sep_idx or n.op != "call_function" or n in self.params:
                continue
            user_idxs = [self.idx.get(u, -1) for u in n.users]
            if not any(j >= self.sep_bwd_idx for j in user_idxs):
                continue
            self.intermediates.append(Intermediate(
                node=n,
                last_fwd_idx =max((j for j in user_idxs if j <  self.sep_idx),     default=i),
                first_bwd_idx=min((j for j in user_idxs if j >= self.sep_bwd_idx), default=-1),
            ))
            self.node_type[n] = NodeType.ACT

    def compute_sizes(self) -> None:
        """Per-node logical size (output bytes) and per-node allocation
        size (zero for aliasing nodes)."""
        self.node_logical_size_bytes: Dict[fx.Node, int] = {
            n: output_bytes(n) for n in self.nodes
        }
        self.node_size_bytes: Dict[fx.Node, int] = {
            n: allocation_bytes(n) for n in self.nodes
        }
        for inter in self.intermediates:
            inter.size_bytes = self.node_logical_size_bytes[inter.node]

    def compute_aliases(self) -> None:
        """Map every node to the node that owns its underlying GPU storage.

        getitem of an aliasing parent (split, foreach in-place) chains to
        the input element it views.  An in-place / view op chains to its
        first sized input.  Independent multi-output (conv_backward) and
        non-aliasing nodes are their own owners.
        """
        self.storage_owner:    Dict[fx.Node, fx.Node]       = {}
        self.aliases_by_owner: Dict[fx.Node, List[fx.Node]] = defaultdict(list)

        for n in self.nodes:
            owner = n
            if is_getitem_of_container(n):
                parent = n.all_input_nodes[0]
                if parent.op == "call_function" and alias_in_schema(parent):
                    target = alias_target(parent, n)
                    if target is not None and self.node_logical_size_bytes.get(target, 0) > 0:
                        owner = self.storage_owner.get(target, target)
            elif n.op == "call_function" and alias_in_schema(n):
                for inp in n.all_input_nodes:
                    if self.node_logical_size_bytes.get(inp, 0) > 0:
                        owner = self.storage_owner.get(inp, inp)
                        break
            self.storage_owner[n] = owner
            self.aliases_by_owner[owner].append(n)

    # ----- runtime measurement --------------------------------------------- #

    def reset_stats(self) -> None:
        """Clear per-iteration runtime state; called at construction and
        between warm-up and measurement runs."""
        self._runtimes_ms: Dict[str, List[float]] = defaultdict(list)
        self._memory_after_bytes: Dict[str, List[int]] = defaultdict(list)
        self._memory_delta_bytes: Dict[str, List[int]] = defaultdict(list)
        self._memory_peak_bytes: Dict[str, List[int]] = defaultdict(list)
        self._latencies_ms: List[float] = []

        self.avg_runtime_ms: Dict[str, float] = {}
        self.avg_memory_after_bytes: Dict[str, int] = {}
        self.avg_memory_delta_bytes: Dict[str, int] = {}
        self.avg_memory_peak_bytes: Dict[str, int] = {}
        self.avg_iter_latency_ms: float = 0.0

    def run(self, *args, **kwargs) -> Any:
        """Inherited interpreter loop, plus whole-iteration timing."""
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
        start.record()
        result = super().run(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self._latencies_ms.append(start.elapsed_time(end))
        return result

    def run_node(self, n: fx.Node) -> Any:
        """Inherited per-node dispatch, plus per-node timing and memory."""
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        self._runtimes_ms[n.name].append(start.elapsed_time(end))
        self._memory_after_bytes[n.name].append(after)
        self._memory_delta_bytes[n.name].append(after - before)
        self._memory_peak_bytes[n.name].append(torch.cuda.max_memory_allocated())
        return result

    def aggregate_stats(self) -> None:
        """Average per-node runtime/memory and iteration latency across
        whatever measurement runs were collected."""
        self.avg_runtime_ms = {name: sum(vs) / len(vs)
                               for name, vs in self._runtimes_ms.items()}
        self.avg_memory_after_bytes = {name: sum(vs) // len(vs)
                                       for name, vs in self._memory_after_bytes.items()}
        self.avg_memory_delta_bytes = {name: sum(vs) // len(vs)
                                       for name, vs in self._memory_delta_bytes.items()}
        self.avg_memory_peak_bytes = {name: sum(vs) // len(vs)
                                      for name, vs in self._memory_peak_bytes.items()}
        for inter in self.intermediates:
            inter.recompute_ms = self.avg_runtime_ms.get(inter.node.name, 0.0)
        if self._latencies_ms:
            self.avg_iter_latency_ms = (sum(self._latencies_ms)
                                        / len(self._latencies_ms))

    # ----- queries --------------------------------------------------------- #

    def memory_timeline_by_role(self) -> Dict[NodeType, List[int]]:
        """Live bytes at each step, broken down by tensor role.

        Each storage owner contributes its size to every step from its
        production index through its last-use index, inclusive.
        Placeholders are special-cased to be live for the whole iteration
        (they exist before step 0 and persist after step N-1).  The role
        of an owner is the highest-priority role across its aliases.
        """
        n = len(self.nodes)
        timeline: Dict[NodeType, List[int]] = {nt: [0] * n for nt in NodeType}

        for owner, aliases in self.aliases_by_owner.items():
            size = self.node_size_bytes.get(owner, 0)
            if size == 0:
                continue
            role = max(self.node_type[a] for a in aliases)
            if owner.op == "placeholder":
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
        timeline  = self.memory_timeline_by_role()
        n         = len(self.nodes)
        per_step  = [sum(timeline[nt][t] for nt in NodeType) for t in range(n)]
        peak_step = max(range(n), key=per_step.__getitem__) if n else 0

        roles   = Counter(self.node_type.values())
        regions = Counter(self.region.values())

        print(f"  Nodes: {n}  "
              f"(forward {regions[Region.FORWARD]}, "
              f"backward {regions[Region.BACKWARD]}, "
              f"optimizer {regions[Region.OPTIMIZER]})")
        print(f"  Intermediates: {len(self.intermediates)}\n")
        print(f"  Static peak (step {peak_step}):")
        print(f"    {'Role':<8} {'Nodes':>6} {'Bytes':>10}")
        for nt in DISPLAY_ROLES:
            print(f"    {nt.name:<8} {roles[nt]:>6}"
                  f" {timeline[nt][peak_step] / 1024**2:>7.2f} MB")
        print(f"    {'TOTAL':<8} {n:>6} {per_step[peak_step] / 1024**2:>7.2f} MB\n")
        print(f"  Iteration latency: {self.avg_iter_latency_ms:>7.2f} ms"
              f"  (avg of {len(self._latencies_ms)} runs)")

    def write_full_log(self, path: str) -> None:
        """Write per-node compute/memory statistics and static facts."""
        with open(path, "w") as f:
            print("OPERATION SUMMARY", file=f)
            print("=" * 120, file=f)
            print(f"{'#':<5} {'Node':<32} {'Region':<10} {'Role':<10}"
                  f" {'Time(ms)':>10} {'Out(B)':>12} {'Alloc(B)':>12}"
                  f" {'Delta(B)':>12} {'Peak(B)':>12}  Target", file=f)
            print("-" * 140, file=f)
            for i, n in enumerate(self.nodes):
                print(f"{i:<5} {n.name[:31]:<32}"
                      f" {self.region[n].value:<10}"
                      f" {self.node_type[n].name:<10}"
                      f" {self.avg_runtime_ms.get(n.name, 0.0):>10.3f}"
                      f" {self.node_logical_size_bytes.get(n, 0):>12}"
                      f" {self.node_size_bytes.get(n, 0):>12}"
                      f" {self.avg_memory_delta_bytes.get(n.name, 0):>12}"
                      f" {self.avg_memory_peak_bytes.get(n.name, 0):>12}"
                      f"  {str(getattr(n, 'target', n.op))[:70]}",
                      file=f)

            print("\nINTERMEDIATE ACTIVATIONS", file=f)
            print("=" * 120, file=f)
            print(f"{'Name':<32} {'Size(KB)':>10} {'LastFwd':>8}"
                  f" {'FirstBwd':>9} {'Lifetime':>9} {'Recomp(ms)':>11}",
                  file=f)
            print("-" * 90, file=f)
            for inter in self.intermediates:
                print(f"{inter.node.name[:31]:<32}"
                      f" {inter.size_bytes / 1024:>10.2f}"
                      f" {inter.last_fwd_idx:>8}"
                      f" {inter.first_bwd_idx:>9}"
                      f" {inter.first_bwd_idx - inter.last_fwd_idx:>9}"
                      f" {inter.recompute_ms:>11.4f}",
                      file=f)

    def write_json_log(self, path: str) -> None:
        """Machine-readable dump of static analysis and runtime stats."""
        import json

        timeline = self.memory_timeline_by_role()
        data = {
            "summary": {
                "n_nodes": len(self.nodes),
                "n_intermediates": len(self.intermediates),
                "peak_static_bytes": self.peak_memory_bytes(),
                "iteration_latency_ms": self.avg_iter_latency_ms,
                "sep_idx": self.sep_idx,
                "sep_bwd_idx": self.sep_bwd_idx,
                "opt_idx": self.opt_idx,
            },
            "nodes": [
                {
                    "idx": i,
                    "name": n.name,
                    "region": self.region[n].value,
                    "role": self.node_type[n].name,
                    "runtime_ms": self.avg_runtime_ms.get(n.name, 0.0),
                    "logical_size_bytes": self.node_logical_size_bytes.get(n, 0),
                    "allocation_size_bytes": self.node_size_bytes.get(n, 0),
                    "memory_delta_bytes": self.avg_memory_delta_bytes.get(n.name, 0),
                    "memory_peak_bytes": self.avg_memory_peak_bytes.get(n.name, 0),
                    "memory_after_bytes": self.avg_memory_after_bytes.get(n.name, 0),
                    "storage_owner": self.storage_owner.get(n, n).name,
                    "target": str(getattr(n, "target", n.op))[:120],
                }
                for i, n in enumerate(self.nodes)
            ],
            "intermediates": [
                {
                    "name": inter.node.name,
                    "size_bytes": inter.size_bytes,
                    "last_fwd_idx": inter.last_fwd_idx,
                    "first_bwd_idx": inter.first_bwd_idx,
                    "lifetime": inter.first_bwd_idx - inter.last_fwd_idx,
                    "recompute_ms": inter.recompute_ms,
                }
                for inter in self.intermediates
            ],
            "timeline_by_role_bytes": {
                role.name: timeline[role] for role in NodeType
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
