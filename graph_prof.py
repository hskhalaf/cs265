"""
Phase 1: Graph Profiler.

Subclasses ``torch.fx.Interpreter`` to walk a traced training step node by
node and collect, per node:

* execution time (CUDA-event timed)
* memory delta (``torch.cuda.memory_allocated`` before/after)
* per-node CUDA peak allocation, which catches transient workspace/scratch
  allocations that are freed before the node returns

and, statically:

* a region per node (FORWARD / LOSS / BACKWARD / OPTIMIZER)
* a tensor role per node (PARAM / ACT / GRAD / OTHER)
* the set of *intermediate activations* (forward call_function nodes whose
  output is consumed in the backward pass) and their lifetimes

The class is read by ``activation_checkpoint.select_activations`` (Phase 2)
and ``visualizer.plot_memory_breakdown`` (Phase 1 plots).

Why the storage-aware sizing matters
------------------------------------
Three FX patterns share a single underlying GPU allocation but appear as
distinct nodes in the graph:

* in-place / view ops marked by their schema's ``alias_info`` (``relu_``,
  ``view``, ``transpose``, ``_foreach_add_``, ``_fused_adam``);
* multi-output ops whose result is a list/tuple, unpacked via
  ``operator.getitem`` (``aten._foreach_div`` followed by 62 getitems on
  ResNet18, or ``aten.split`` followed by getitems on its slices);
* placeholders that hold steady-state buffers (model parameters, Adam
  moments) and persist across iterations.

``_compute_aliases`` collapses each of these into a single *storage owner*,
and ``memory_timeline_by_role`` walks by owner so the bytes are counted
exactly once.  Placeholders are special-cased to be live for the whole
iteration since they exist before step 0 and after step N-1.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass
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
    """Four roles, matching the base code.

    OTHER absorbs everything that isn't one of the three "first-class" roles:
    optimizer-state placeholders (Adam moment buffers, step counters), the
    scratch tensors the foreach optimizer decomposition allocates during the
    step, backward-pass intermediates that aren't the final accumulated
    gradient, and runtime-only residuals such as cuDNN/cuBLAS workspaces when
    measured memory is folded into the timeline.  All of these are real GPU
    memory; "OTHER" is just the label.
    """
    PARAM = "param"
    ACT   = "activation"
    GRAD  = "gradient"
    OTHER = "other"


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
            yield from _iter_tensors(v)


def _output_alias_targets_inputs(node: fx.Node) -> bool:
    """True if the op's schema declares that any return aliases any argument.

    Catches in-place ops (``_foreach_add_``, ``relu_``), view ops
    (``view``, ``t``, ``transpose``), and fused optimizer ops
    (``_fused_adam``).  These contribute zero new storage; their users still
    extend the lifetime of the original storage owner.
    """
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is None:
        return False
    for ret in schema.returns:
        if ret.alias_info is not None:
            return True
    return False


def _is_getitem(node: fx.Node) -> bool:
    return (
        node.op == OP.CALL_FUNCTION
        and (
            node.target is operator.getitem
            or getattr(node.target, "__name__", None) == "getitem"
        )
    )


def _is_tensor_container_output(node: fx.Node) -> bool:
    val = node.meta.get("val")
    return (
        isinstance(val, (list, tuple))
        and next(_iter_tensors(val), None) is not None
    )


def _is_getitem_of_tensor_container(node: fx.Node) -> bool:
    if not _is_getitem(node) or not node.all_input_nodes:
        return False
    return _is_tensor_container_output(node.all_input_nodes[0])


def _node_output_bytes(node: fx.Node) -> int:
    """Logical bytes in this node's tensor output."""
    if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
        return 0
    val = node.meta.get("val")
    return sum(t.numel() * t.element_size() for t in _iter_tensors(val))


def _node_allocation_bytes(node: fx.Node) -> int:
    """Bytes of new storage allocated by this node.

    Three cases for multi-output ops:
      * **Aliasing** (``aten.split``, ``_foreach_mul_``): parent and getitems
        all contribute 0; storage chain links back to the inputs.
      * **Independent + only-getitem users** (``conv_backward``,
        ``cudnn_batch_norm``): bytes are attributed to each getitem
        individually so per-element role attribution (GRAD vs OTHER) works.
      * **Independent + mixed users**: parent owns full bytes (rare).
    """
    if _is_getitem_of_tensor_container(node):
        parent = node.all_input_nodes[0]
        if (parent.op == OP.CALL_FUNCTION
                and _output_alias_targets_inputs(parent)):
            return 0  # view of parent's input
        return _node_output_bytes(node)  # owns one element of independent output
    if node.op == OP.CALL_FUNCTION and _output_alias_targets_inputs(node):
        return 0
    if (node.op == OP.CALL_FUNCTION
            and _is_tensor_container_output(node)
            and all(_is_getitem(u) for u in node.users)):
        return 0  # bytes are on the getitems
    return _node_output_bytes(node)


def _alias_target_for_getitem(parent: fx.Node,
                               getitem: fx.Node) -> Optional[fx.Node]:
    """For ``getitem(parent, i)`` where parent's outputs alias parent's
    inputs, return the input node whose storage corresponds to ``output[i]``.

    Two patterns:
      * Single-input multi-output (``aten.split``, ``unbind``, ``chunk``):
        all outputs view the same input.
      * List-input multi-output in-place (``_foreach_mul_(a_list, ...)``):
        ``output[i]`` aliases ``a_list[i]``.
    """
    if len(getitem.args) < 2 or not parent.args:
        return None
    idx = getitem.args[1]
    first = parent.args[0]
    if isinstance(first, fx.Node):
        return first
    if (isinstance(first, (list, tuple))
            and isinstance(idx, int)
            and 0 <= idx < len(first)
            and isinstance(first[idx], fx.Node)):
        return first[idx]
    return None


_ROLE_PRIORITY: Dict[NodeType, int] = {
    NodeType.OTHER: 0,
    NodeType.ACT:   1,
    NodeType.GRAD:  2,
    NodeType.PARAM: 3,
}


def _merge_roles(a: NodeType, b: NodeType) -> NodeType:
    return a if _ROLE_PRIORITY[a] >= _ROLE_PRIORITY[b] else b


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
        self._compute_aliases()

        # runtime accumulators
        self._node_runtimes_ms: Dict[str, List[float]] = defaultdict(list)
        self._iter_latency_ms: List[float] = []
        self._measured_memory_runs: List[List[int]] = []
        self._measured_peak_memory_runs: List[List[int]] = []
        self._current_measured: List[int] = []
        self._current_measured_peak: List[int] = []
        self.avg_runtime_ms: Dict[str, float] = {}
        self.avg_iter_latency_ms: float = 0.0
        self.avg_measured_memory: List[int] = []
        self.avg_measured_peak_memory: List[int] = []

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
        """Assign every node one of {PARAM, ACT, GRAD, OTHER}.

        PARAM and GRAD come from the optimizer call: under the fused path
        they are read directly from ``_fused_adam``'s argument lists, under
        the foreach path PARAM is "placeholder used in both forward and
        optimizer" and GRAD is "backward call_function whose output flows
        into the optimizer".  ACT is filled in later by
        ``_find_intermediates``.  Everything else is OTHER — opt-state
        placeholders, foreach scratch tensors, gradient-of-activation
        intermediates, constants.
        """
        self.params: Set[fx.Node] = set()
        self.grads:  Set[fx.Node] = set()

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
            for n in self.nodes:
                if n.op != OP.PLACEHOLDER:
                    continue
                user_idx = [self.idx[u] for u in n.users if u in self.idx]
                in_fwd = any(i <  self.sep_idx for i in user_idx)
                in_opt = any(i >= self.opt_idx for i in user_idx)
                if in_fwd and in_opt:
                    self.params.add(n)
            for n in self.nodes:
                if n.op != OP.CALL_FUNCTION:
                    continue
                i = self.idx[n]
                if not (self.sep_bwd_idx <= i < self.opt_idx):
                    continue
                if any(self.idx[u] >= self.opt_idx for u in n.users
                       if u in self.idx):
                    self.grads.add(n)

        self.node_type: Dict[fx.Node, NodeType] = {}
        for n in self.nodes:
            if   n in self.params: self.node_type[n] = NodeType.PARAM
            elif n in self.grads:  self.node_type[n] = NodeType.GRAD
            else:                  self.node_type[n] = NodeType.OTHER

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
        """Per-node bytes and back-fill Intermediate.size_bytes."""
        self.node_logical_size_bytes: Dict[fx.Node, int] = {
            n: _node_output_bytes(n) for n in self.nodes
        }
        self.node_size_bytes: Dict[fx.Node, int] = {
            n: _node_allocation_bytes(n) for n in self.nodes
        }
        for inter in self.intermediates:
            inter.size_bytes = self.node_logical_size_bytes[inter.node]

    def _compute_aliases(self) -> None:
        """Map alias/view/in-place nodes to the node that owns their storage.

        Three categories chain through to a parent owner:
          * ``getitem(container, i)`` — owner = the container's owner.
          * single-output alias ops (``view``, ``relu_``, ...) — owner = first
            sized input's owner.
          * **multi-output in-place ops** (``_foreach_mul_``, ``_fused_adam``,
            ``aten.split``) — same rule as single-output.  Their schema marks
            returns as aliasing inputs but our older check skipped them
            because the output is a list/tuple instead of a single Tensor;
            without this, a chain like ``denom = _foreach_sqrt(v)`` →
            ``_foreach_mul_(denom, ...)`` → ``_foreach_addcdiv_`` would treat
            each in-place step as starting a fresh allocation, double-
            counting the optimizer scratch buffer at every step in the chain.
        """
        self.storage_owner: Dict[fx.Node, fx.Node] = {}
        self.aliases_by_owner: Dict[fx.Node, List[fx.Node]] = defaultdict(list)

        for n in self.nodes:
            owner = n
            if _is_getitem_of_tensor_container(n):
                parent = n.all_input_nodes[0]
                if (parent.op == OP.CALL_FUNCTION
                        and _output_alias_targets_inputs(parent)):
                    # Aliasing parent: chain to the input element this
                    # getitem corresponds to (split → input; foreach in-place
                    # → input list[i]).
                    target = _alias_target_for_getitem(parent, n)
                    if (target is not None
                            and self.node_logical_size_bytes.get(target, 0) > 0):
                        owner = self.storage_owner.get(target, target)
                # Independent multi-output (conv_backward etc.): leave the
                # getitem as its own owner so per-element role attribution
                # (GRAD vs OTHER) is preserved.
            elif n.op == OP.CALL_FUNCTION and _output_alias_targets_inputs(n):
                for inp in n.all_input_nodes:
                    if self.node_logical_size_bytes.get(inp, 0) > 0:
                        owner = self.storage_owner.get(inp, inp)
                        break
            self.storage_owner[n] = owner
            self.aliases_by_owner[owner].append(n)

    # --- runtime ------------------------------------------------------------ #

    def run(self, *args, **kwargs) -> Any:
        """Time the whole iteration in addition to per-node timing."""
        torch.cuda.synchronize()
        self._current_measured = []
        self._current_measured_peak = []
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self._iter_latency_ms.append(start.elapsed_time(end))
        self._measured_memory_runs.append(self._current_measured)
        self._measured_peak_memory_runs.append(self._current_measured_peak)
        return result

    def run_node(self, n: fx.Node) -> Any:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.synchronize()
        self._node_runtimes_ms[n.name].append(start.elapsed_time(end))
        self._current_measured.append(torch.cuda.memory_allocated())
        self._current_measured_peak.append(torch.cuda.max_memory_allocated())
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
        self.avg_measured_memory = self._average_runs(
            self._measured_memory_runs,
        )
        self.avg_measured_peak_memory = self._average_runs(
            self._measured_peak_memory_runs,
        )

    def reset_stats(self) -> None:
        self._node_runtimes_ms.clear()
        self._iter_latency_ms.clear()
        self._measured_memory_runs.clear()
        self._measured_peak_memory_runs.clear()
        self._current_measured = []
        self._current_measured_peak = []
        self.avg_runtime_ms.clear()
        self.avg_iter_latency_ms = 0.0
        self.avg_measured_memory = []
        self.avg_measured_peak_memory = []

    def _average_runs(self, runs: List[List[int]]) -> List[int]:
        if not runs:
            return []
        n = len(self.nodes)
        sums   = [0] * n
        counts = [0] * n
        for run in runs:
            for i, v in enumerate(run[:n]):
                sums[i]   += v
                counts[i] += 1
        return [sums[i] // counts[i] if counts[i] else 0 for i in range(n)]

    # --- queries ------------------------------------------------------------ #

    def memory_timeline_by_role(
        self,
        include_runtime_residual: bool = False,
    ) -> Dict[NodeType, List[int]]:
        """Live bytes at each step, broken down by tensor role.

        A node's contribution is added to every step from its production index
        through its last-use index, inclusive.

        If ``include_runtime_residual`` is true, the gap between the static
        FX-visible tensor total and the measured per-node CUDA peak is added
        to ``OTHER`` at each step.  This accounts for real allocations that
        are not outputs of graph nodes, e.g. cuDNN/cuBLAS workspaces and
        foreach-internal scratch buffers.
        """
        n = len(self.nodes)
        timeline: Dict[NodeType, List[int]] = {nt: [0] * n for nt in NodeType}
        for owner, aliases in self.aliases_by_owner.items():
            size = self.node_size_bytes.get(owner, 0)
            if size == 0:
                continue
            role = self.node_type[owner]
            if owner.op == OP.PLACEHOLDER:
                # Placeholders represent state that exists before the
                # iteration starts (params, optimizer state, batched
                # inputs) and is still on the GPU when it ends.  They are
                # the steady-state baseline — live for every step.
                produced = 0
                last_use = n - 1
                for alias in aliases:
                    role = _merge_roles(role, self.node_type[alias])
            else:
                produced = self.idx[owner]
                last_use = produced
                for alias in aliases:
                    last_use = max(last_use, self.idx[alias])
                    role = _merge_roles(role, self.node_type[alias])
                    user_idx = [self.idx[u] for u in alias.users
                                if u in self.idx]
                    last_use = max([last_use, *user_idx])
            for t in range(produced, min(last_use + 1, n)):
                timeline[role][t] += size
        if include_runtime_residual:
            measured = self.avg_measured_peak_memory or self.avg_measured_memory
            for t, measured_bytes in enumerate(measured[:n]):
                static_total = sum(timeline[nt][t] for nt in NodeType)
                residual = measured_bytes - static_total
                if residual > 0:
                    timeline[NodeType.OTHER][t] += residual
        return timeline

    def peak_memory_bytes(self, include_runtime_residual: bool = False) -> int:
        roles = self.memory_timeline_by_role(include_runtime_residual)
        if not self.nodes:
            return 0
        return max(sum(roles[nt][t] for nt in NodeType)
                   for t in range(len(self.nodes)))

    def measured_peak_memory_bytes(self) -> int:
        measured = self.avg_measured_peak_memory or self.avg_measured_memory
        return max(measured, default=0)

    def runtime_residual_by_step(self) -> List[int]:
        n = len(self.nodes)
        timeline = self.memory_timeline_by_role(include_runtime_residual=False)
        measured = self.avg_measured_peak_memory or self.avg_measured_memory
        residual = [0] * n
        for t, measured_bytes in enumerate(measured[:n]):
            static_total = sum(timeline[nt][t] for nt in NodeType)
            residual[t] = max(0, measured_bytes - static_total)
        return residual

    def iteration_latency_ms(self) -> float:
        return self.avg_iter_latency_ms

    # --- pretty printing ---------------------------------------------------- #

    def _role_node_counts(self) -> Dict[NodeType, int]:
        counts: Dict[NodeType, int] = defaultdict(int)
        for nt in self.node_type.values():
            counts[nt] += 1
        return counts

    def _peak_breakdown(
        self,
        include_runtime_residual: bool = False,
    ) -> "tuple[int, int, Dict[NodeType, int]]":
        """At the peak step, total live bytes and per-role breakdown."""
        timeline = self.memory_timeline_by_role(include_runtime_residual)
        n = len(self.nodes)
        if n == 0:
            return 0, 0, {nt: 0 for nt in NodeType}
        per_step_total = [sum(timeline[nt][t] for nt in NodeType)
                          for t in range(n)]
        peak_step = max(range(n), key=lambda t: per_step_total[t])
        breakdown = {nt: timeline[nt][peak_step] for nt in NodeType}
        return peak_step, per_step_total[peak_step], breakdown

    def print_summary(
        self,
        file=None,
        include_runtime_residual: bool = False,
    ) -> None:
        """One-screen summary: peak contribution per role + latency."""
        p = lambda *a, **k: print(*a, **k, file=file) if file else print(*a, **k)
        counts = self._role_node_counts()
        peak_step, peak_total, peak_by_role = self._peak_breakdown(
            include_runtime_residual,
        )
        region_counts: Dict[Region, int] = defaultdict(int)
        for r in self.region.values():
            region_counts[r] += 1

        p(f"  Nodes: {len(self.nodes)}  "
          f"(forward {region_counts[Region.FORWARD]}, "
          f"backward {region_counts[Region.BACKWARD]}, "
          f"optimizer {region_counts[Region.OPTIMIZER]})")
        p(f"  Intermediates: {len(self.intermediates)}")
        p()
        peak_label = "measured peak" if include_runtime_residual else "FX-visible peak"
        p(f"  At {peak_label} (step {peak_step}):")
        p(f"    {'Role':<8} {'Nodes':>6} {'Live':>10}")
        for nt in NodeType:
            p(f"    {nt.name:<8} {counts[nt]:>6}"
              f" {peak_by_role[nt] / 1024**2:>7.2f} MB")
        p(f"    {'TOTAL':<8} {len(self.nodes):>6}"
          f" {peak_total / 1024**2:>7.2f} MB")
        p()
        p(f"  Iteration latency: {self.avg_iter_latency_ms:>7.2f} ms"
          f"  (avg of {len(self._iter_latency_ms)} runs)")

    def write_full_log(self, path: str) -> None:
        """Write everything (per-node table, intermediate table, summary)
        to a text file.  This is what the verbose --debug output used to
        spam the console with."""
        with open(path, "w") as f:
            print("=" * 90, file=f)
            print("OPERATION SUMMARY", file=f)
            print("=" * 90, file=f)
            print(f"{'#':<5} {'Node':<35} {'Region':<10} {'Role':<10}"
                  f" {'Time(ms)':>10} {'Size(B)':>12} {'Alloc(B)':>12}  Target",
                  file=f)
            print("-" * 124, file=f)
            for i, n in enumerate(self.nodes):
                target = str(getattr(n, "target", n.op))[:50]
                print(f"{i:<5} {n.name[:34]:<35} {self.region[n].value:<10}"
                      f" {self.node_type[n].value:<10}"
                      f" {self.avg_runtime_ms.get(n.name, 0.0):>10.3f}"
                      f" {self.node_logical_size_bytes.get(n, 0):>12}"
                      f" {self.node_size_bytes.get(n, 0):>12}  {target}",
                      file=f)

            print("\n" + "=" * 90, file=f)
            print("INTERMEDIATE ACTIVATIONS", file=f)
            print("=" * 90, file=f)
            print(f"{'Name':<30} {'Size(KB)':>10} {'LastFwd':>8}"
                  f" {'FirstBwd':>9} {'Lifetime':>9} {'Recomp(ms)':>11}",
                  file=f)
            print("-" * 80, file=f)
            for inter in self.intermediates:
                lt = inter.first_bwd_idx - inter.last_fwd_idx
                print(f"{inter.node.name[:29]:<30}"
                      f" {inter.size_bytes / 1024:>10.2f}"
                      f" {inter.last_fwd_idx:>8} {inter.first_bwd_idx:>9}"
                      f" {lt:>9} {inter.recompute_ms:>11.4f}", file=f)

            print("\n" + "=" * 90, file=f)
            print("SUMMARY", file=f)
            print("=" * 90, file=f)
            self.print_summary(file=f)

    def write_json_log(self, path: str) -> None:
        """Dump the full profile (per-node table, intermediates, summary,
        timeline by role, measured-memory series) to JSON for downstream
        analysis (notebooks, plots, diff against another run)."""
        import json

        timeline = self.memory_timeline_by_role(include_runtime_residual=False)
        adjusted_timeline = self.memory_timeline_by_role(
            include_runtime_residual=True,
        )
        peak_step, peak_total, peak_by_role = self._peak_breakdown()
        measured_peak_step, measured_peak_total, measured_peak_by_role = (
            self._peak_breakdown(include_runtime_residual=True)
        )

        data = {
            "summary": {
                "n_nodes":              len(self.nodes),
                "n_intermediates":      len(self.intermediates),
                "iteration_latency_ms": self.avg_iter_latency_ms,
                "peak_step":            peak_step,
                "peak_total_bytes":     peak_total,
                "peak_by_role_bytes":   {nt.name: peak_by_role[nt]
                                         for nt in NodeType},
                "measured_peak_step":    measured_peak_step,
                "measured_peak_total_bytes": measured_peak_total,
                "measured_peak_by_role_bytes": {
                    nt.name: measured_peak_by_role[nt] for nt in NodeType
                },
                "sep_idx":              self.sep_idx,
                "sep_bwd_idx":          self.sep_bwd_idx,
                "opt_idx":              self.opt_idx,
            },
            "nodes": [
                {
                    "idx":         i,
                    "name":        n.name,
                    "region":      self.region[n].value,
                    "role":        self.node_type[n].value,
                    "runtime_ms":  self.avg_runtime_ms.get(n.name, 0.0),
                    "size_bytes":  self.node_logical_size_bytes.get(n, 0),
                    "allocation_size_bytes": self.node_size_bytes.get(n, 0),
                    "storage_owner": self.storage_owner.get(n, n).name,
                    "target":      str(getattr(n, "target", n.op))[:80],
                }
                for i, n in enumerate(self.nodes)
            ],
            "intermediates": [
                {
                    "name":          inter.node.name,
                    "size_bytes":    inter.size_bytes,
                    "last_fwd_idx":  inter.last_fwd_idx,
                    "first_bwd_idx": inter.first_bwd_idx,
                    "lifetime":      inter.first_bwd_idx - inter.last_fwd_idx,
                    "recompute_ms":  inter.recompute_ms,
                }
                for inter in self.intermediates
            ],
            "timeline_by_role_bytes": {nt.name: timeline[nt]
                                       for nt in NodeType},
            "timeline_by_role_with_runtime_residual_bytes": {
                nt.name: adjusted_timeline[nt] for nt in NodeType
            },
            "measured_memory_bytes":  self.avg_measured_memory,
            "measured_peak_memory_bytes": self.avg_measured_peak_memory,
            "runtime_residual_bytes": self.runtime_residual_by_step(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
