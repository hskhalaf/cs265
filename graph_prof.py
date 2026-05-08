"""
Phase 1: Graph Profiler.

Subclasses ``torch.fx.Interpreter`` to walk a traced training step node by
node and collect, per node:

* execution time (CUDA-event timed)
* memory delta (``torch.cuda.memory_allocated`` before/after)

and, statically:

* a region per node (FORWARD / LOSS / BACKWARD / OPTIMIZER)
* a tensor role per node (PARAM / ACT / GRAD / OPT_STATE / OPT_SCRATCH / OTHER)
* the set of *intermediate activations* (forward call_function nodes whose
  output is consumed in the backward pass) and their lifetimes

The class is read by ``activation_checkpoint.select_activations`` (Phase 2)
and ``visualizer.plot_memory_breakdown`` (Phase 1 plots).

Why the storage-aware sizing matters
------------------------------------
Adam with ``foreach=True`` decomposes the optimizer step into ``_foreach_*``
ops whose output list elements *alias* the input parameters in-place.  A
naive size walk that counts both the input params and the foreach output
double-counts the parameters and produces a fake "spike" in the memory chart
at the optimizer step.  We avoid this by checking each op's schema: if any
return value aliases any argument, the op contributes 0 new bytes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
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


class NodeType(Enum):
    PARAM        = "param"
    ACT          = "activation"
    GRAD         = "gradient"
    OPT_STATE    = "optimizer_state"
    OPT_SCRATCH  = "optimizer_scratch"
    OTHER        = "other"


class Region(Enum):
    FORWARD   = "forward"
    LOSS      = "loss"
    BACKWARD  = "backward"
    OPTIMIZER = "optimizer"


@dataclass
class Intermediate:
    """An activation produced in the forward pass and consumed in backward."""
    node: fx.Node
    size_bytes: int = 0
    last_fwd_idx: int = -1
    first_bwd_idx: int = -1
    recompute_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _iter_tensors(val: Any) -> Iterable[torch.Tensor]:
    if isinstance(val, torch.Tensor):
        yield val
    elif isinstance(val, (list, tuple)):
        for v in val:
            if isinstance(v, torch.Tensor):
                yield v


def _output_alias_targets_inputs(node: fx.Node) -> bool:
    """True if the op's schema declares that any return aliases any argument.

    Catches in-place ops (``_foreach_add_``, ``relu_``), view ops
    (``view``, ``t``, ``transpose``), and fused optimizer ops
    (``_fused_adam``).  These contribute zero new bytes to live memory.
    """
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is None:
        return False
    for ret in schema.returns:
        if ret.alias_info is not None:
            return True
    return False


def _node_output_bytes(node: fx.Node) -> int:
    """Bytes added to live memory by this node, accounting for aliasing."""
    if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
        return 0
    if node.op == OP.CALL_FUNCTION and _output_alias_targets_inputs(node):
        return 0
    val = node.meta.get("val")
    return sum(t.numel() * t.element_size() for t in _iter_tensors(val))


# --------------------------------------------------------------------------- #
# GraphProfiler                                                               #
# --------------------------------------------------------------------------- #


class GraphProfiler(fx.Interpreter):

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.nodes: List[fx.Node] = list(self.module.graph.nodes)
        self.idx:   Dict[fx.Node, int] = {n: i for i, n in enumerate(self.nodes)}

        # static analysis (order matters: each step depends on the previous)
        self._find_separators()
        self._assign_regions()
        self._classify_tensors()
        self._find_intermediates()
        self._compute_sizes()

        # runtime accumulators
        self._node_runtimes_ms: Dict[str, List[float]] = defaultdict(list)
        self._iter_latency_ms: List[float] = []
        self.avg_runtime_ms: Dict[str, float] = {}
        self.avg_iter_latency_ms: float = 0.0

    # --- static analysis ---------------------------------------------------- #

    def _find_separators(self) -> None:
        """Locate %sep, %sep_backward, and the start of the optimizer step."""
        self.sep_idx = self.sep_bwd_idx = self.opt_idx = -1
        for i, n in enumerate(self.nodes):
            if n.op != OP.CALL_FUNCTION:
                continue
            if n.target == torch.ops.separator.sep.default:
                self.sep_idx = i
            elif n.target == torch.ops.separator.sep_backward.default:
                self.sep_bwd_idx = i
        assert self.sep_idx >= 0,     "could not find %sep"
        assert self.sep_bwd_idx >= 0, "could not find %sep_backward"

        # Optimizer step starts at the first _fused_adam, or — for the
        # foreach optimizer — at the first _foreach_* op after sep_backward.
        for i, n in enumerate(self.nodes):
            if n.op != OP.CALL_FUNCTION:
                continue
            if n.target == torch.ops.aten._fused_adam.default:
                self.opt_idx = i
                return
        for i in range(self.sep_bwd_idx + 1, len(self.nodes)):
            n = self.nodes[i]
            if n.op == OP.CALL_FUNCTION and "_foreach_" in str(n.target):
                self.opt_idx = i
                return

    def _assign_regions(self) -> None:
        self.region: Dict[fx.Node, Region] = {}
        for i, n in enumerate(self.nodes):
            if 0 <= self.opt_idx <= i:    self.region[n] = Region.OPTIMIZER
            elif i >= self.sep_bwd_idx:   self.region[n] = Region.BACKWARD
            elif i > self.sep_idx:        self.region[n] = Region.LOSS
            else:                         self.region[n] = Region.FORWARD

    def _classify_tensors(self) -> None:
        """Assign every node a NodeType.

        - PARAM: an input to the (fused) optimizer.
        - GRAD : the gradient input list of the optimizer.
        - OPT_STATE: placeholder used only in the optimizer region.
        - OPT_SCRATCH: call_function in the optimizer region (foreach scratch).
        - ACT : tagged later in _find_intermediates.
        - OTHER: everything else (e.g. constants, scalars).
        """
        self.params: Set[fx.Node] = set()
        self.grads:  Set[fx.Node] = set()
        self.opt_states: Set[fx.Node] = set()

        # Find the optimizer call (fused or foreach) to read its argument lists.
        fused_node = next(
            (n for n in self.nodes
             if n.op == OP.CALL_FUNCTION
             and n.target == torch.ops.aten._fused_adam.default),
            None,
        )
        if fused_node is not None:
            self.params = set(fused_node.args[0])
            self.grads  = set(fused_node.args[1])
        elif self.opt_idx >= 0:
            # Foreach path: parameters are placeholders used in BOTH the
            # forward region (model reads its weight) AND the optimizer region
            # (Adam updates the weight).
            for n in self.nodes:
                if n.op != OP.PLACEHOLDER:
                    continue
                user_idx = [self.idx[u] for u in n.users if u in self.idx]
                in_fwd = any(i <  self.sep_idx for i in user_idx)
                in_opt = any(i >= self.opt_idx for i in user_idx)
                if in_fwd and in_opt:
                    self.params.add(n)

        # Optimizer state: placeholder used only in the optimizer region.
        if self.opt_idx >= 0:
            for n in self.nodes:
                if n.op != OP.PLACEHOLDER or n in self.params or n in self.grads:
                    continue
                user_idx = [self.idx[u] for u in n.users if u in self.idx]
                if user_idx and all(i >= self.opt_idx for i in user_idx):
                    self.opt_states.add(n)

        # Default classification.
        self.node_type: Dict[fx.Node, NodeType] = {}
        for n in self.nodes:
            if   n in self.params:     self.node_type[n] = NodeType.PARAM
            elif n in self.grads:      self.node_type[n] = NodeType.GRAD
            elif n in self.opt_states: self.node_type[n] = NodeType.OPT_STATE
            elif self.region[n] == Region.OPTIMIZER and n.op == OP.CALL_FUNCTION:
                self.node_type[n] = NodeType.OPT_SCRATCH
            else:
                self.node_type[n] = NodeType.OTHER

    def _find_intermediates(self) -> None:
        """Forward call_function nodes whose output is read in the backward pass."""
        self.intermediates: List[Intermediate] = []
        for n in self.nodes:
            i = self.idx[n]
            if i >= self.sep_idx or n.op != OP.CALL_FUNCTION or n in self.params:
                continue
            user_idx = [self.idx.get(u, -1) for u in n.users]
            if not any(j >= self.sep_bwd_idx for j in user_idx):
                continue
            last_fwd  = max((j for j in user_idx if j <  self.sep_idx),
                            default=i)
            first_bwd = min((j for j in user_idx if j >= self.sep_bwd_idx),
                            default=-1)
            self.intermediates.append(Intermediate(
                node=n, last_fwd_idx=last_fwd, first_bwd_idx=first_bwd,
            ))
            self.node_type[n] = NodeType.ACT

    def _compute_sizes(self) -> None:
        """Per-node bytes (storage-aware) and back-fill Intermediate.size_bytes."""
        self.node_size_bytes: Dict[fx.Node, int] = {
            n: _node_output_bytes(n) for n in self.nodes
        }
        for inter in self.intermediates:
            inter.size_bytes = self.node_size_bytes[inter.node]

    # --- runtime ------------------------------------------------------------ #

    def run(self, *args, **kwargs) -> Any:
        """Time the whole iteration in addition to per-node timing."""
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self._iter_latency_ms.append(start.elapsed_time(end))
        return result

    def run_node(self, n: fx.Node) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.synchronize()
        self._node_runtimes_ms[n.name].append(start.elapsed_time(end))
        return result

    def aggregate_stats(self) -> None:
        for name, samples in self._node_runtimes_ms.items():
            self.avg_runtime_ms[name] = sum(samples) / len(samples)
        for inter in self.intermediates:
            inter.recompute_ms = self.avg_runtime_ms.get(inter.node.name, 0.0)
        if self._iter_latency_ms:
            self.avg_iter_latency_ms = (
                sum(self._iter_latency_ms) / len(self._iter_latency_ms)
            )

    def reset_stats(self) -> None:
        self._node_runtimes_ms.clear()
        self._iter_latency_ms.clear()
        self.avg_runtime_ms.clear()
        self.avg_iter_latency_ms = 0.0

    # --- queries ------------------------------------------------------------ #

    def memory_timeline_by_role(self) -> Dict[NodeType, List[int]]:
        """Live bytes at each step, broken down by tensor role.

        A node's contribution is added to every step from its production index
        through its last-use index, inclusive.
        """
        n = len(self.nodes)
        timeline: Dict[NodeType, List[int]] = {nt: [0] * n for nt in NodeType}
        for node in self.nodes:
            size = self.node_size_bytes.get(node, 0)
            if size == 0:
                continue
            produced = self.idx[node]
            user_idx = [self.idx[u] for u in node.users if u in self.idx]
            last_use = max(user_idx, default=produced)
            role = self.node_type[node]
            for t in range(produced, min(last_use + 1, n)):
                timeline[role][t] += size
        return timeline

    def peak_memory_bytes(self) -> int:
        roles = self.memory_timeline_by_role()
        if not self.nodes:
            return 0
        return max(sum(roles[nt][t] for nt in NodeType)
                   for t in range(len(self.nodes)))

    def iteration_latency_ms(self) -> float:
        return self.avg_iter_latency_ms

    # --- pretty printing ---------------------------------------------------- #

    def print_stats(self) -> None:
        print("\n" + "=" * 90)
        print("OPERATION SUMMARY")
        print("=" * 90)
        print(f"{'#':<5} {'Node':<35} {'Region':<10} {'Time(ms)':>10} {'Size(B)':>12}")
        print("-" * 90)
        for i, n in enumerate(self.nodes):
            print(f"{i:<5} {n.name[:34]:<35} {self.region[n].value:<10}"
                  f" {self.avg_runtime_ms.get(n.name, 0.0):>10.3f}"
                  f" {self.node_size_bytes.get(n, 0):>12}")

        print("\n" + "=" * 90)
        print("TENSOR CATEGORIZATION")
        print("=" * 90)
        counts: Dict[NodeType, int] = defaultdict(int)
        bytes_: Dict[NodeType, int] = defaultdict(int)
        for n, nt in self.node_type.items():
            counts[nt] += 1
            bytes_[nt] += self.node_size_bytes.get(n, 0)
        for nt in NodeType:
            print(f"  {nt.name:<18} count={counts[nt]:<6}"
                  f" total={bytes_[nt] / 1024:>10.2f} KB")

        print("\n" + "=" * 90)
        print("INTERMEDIATE ACTIVATIONS")
        print("=" * 90)
        print(f"{'Name':<30} {'Size(KB)':>10} {'LastFwd':>8} {'FirstBwd':>9}"
              f" {'Lifetime':>9} {'Recomp(ms)':>11}")
        print("-" * 80)
        for inter in self.intermediates:
            lt = inter.first_bwd_idx - inter.last_fwd_idx
            print(f"{inter.node.name[:29]:<30} {inter.size_bytes / 1024:>10.2f}"
                  f" {inter.last_fwd_idx:>8} {inter.first_bwd_idx:>9}"
                  f" {lt:>9} {inter.recompute_ms:>11.4f}")
        total_act = sum(i.size_bytes for i in self.intermediates)
        print(f"\nTotal intermediates: {len(self.intermediates)}"
              f"  ({total_act / (1024**2):.2f} MB)")

        print("\n" + "=" * 90)
        print("PEAK MEMORY")
        print("=" * 90)
        peak = self.peak_memory_bytes()
        print(f"  Estimated peak: {peak / (1024**2):.2f} MB")
        print(f"  Iteration latency (avg of {len(self._iter_latency_ms)} runs):"
              f" {self.avg_iter_latency_ms:.2f} ms")
        print("=" * 90 + "\n")
