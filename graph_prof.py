"""
Graph Profiler for CS265 Activation Checkpointing Project.

This module implements a profiler that extends torch.fx.Interpreter to execute
an FX graph node-by-node, collecting per-node timing and memory measurements,
classifying tensors by role, and computing activation lifetimes.

The FX graph is produced by graph_tracer.compile(), which traces a full training
step (forward + backward + optimizer) into a single graph.  The graph contains
explicit separator nodes (%sep, %sep_backward) that mark the boundary between
the forward pass, loss computation, and backward pass.

Architecture
------------
The profiler performs two kinds of analysis:

1. **Static analysis** (in __init__): walks the graph once to identify forward/
   backward boundaries, classify nodes by type (PARAM, ACT, GRAD, OTHER),
   identify intermediate activations, and compute their lifetimes (last forward
   use, first backward use).

2. **Runtime profiling** (in run_node): executes each node while measuring
   wall-clock time via torch.cuda.Event and GPU memory delta via
   torch.cuda.memory_allocated().  Intermediate activations are swapped to CPU
   after their last forward use and swapped back before their first backward
   use, both to measure swap overhead and to keep GPU memory manageable during
   profiling.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch
import torch.fx as fx


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------


class OP(str, Enum):
    """FX node operation types."""
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """Classification of tensors in the graph."""
    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


class Region(Enum):
    """Which part of the training step a node belongs to."""
    FORWARD = 0
    LOSS = 1
    BACKWARD = 2
    OPTIMIZER = 3
    OTHER = 4


@dataclass
class IntermediateInfo:
    """Profiling data for a single intermediate activation (feature map).

    Attributes
    ----------
    node : fx.Node
        The FX graph node that produces this activation.
    memory_size : int
        Size in bytes, computed from FakeTensor metadata.
    last_fwd_access : int
        Index (in node_list) of the last forward-region user of this node.
    first_bwd_access : int
        Index (in node_list) of the first backward-region user of this node.
    inactive_time_ms : float
        Wall-clock time between last forward use and first backward use,
        measured during profiling (averaged over measurement iterations).
    recompute_cost_ms : float
        Wall-clock time to execute this node, measured during profiling
        (averaged over measurement iterations).
    swap_out_time_ms : float
        Time to transfer this tensor from GPU to CPU.
    swap_in_time_ms : float
        Time to transfer this tensor from CPU back to GPU.
    """
    node: fx.Node
    memory_size: int = 0
    last_fwd_access: int = -1
    first_bwd_access: int = -1
    inactive_time_ms: float = 0.0
    recompute_cost_ms: float = 0.0
    swap_out_time_ms: float = 0.0
    swap_in_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _tensor_size_bytes(val: Any) -> int:
    """Return the size in bytes of a tensor or tuple of tensors.

    Parameters
    ----------
    val : Any
        The value from ``node.meta['val']``, which is a FakeTensor (or a tuple
        of FakeTensors) when the graph was traced with ``make_fx`` in fake mode.

    Returns
    -------
    int
        Total bytes.  Returns 0 for non-tensor values.
    """
    if isinstance(val, torch.Tensor):
        return val.numel() * val.element_size()
    if isinstance(val, (tuple, list)):
        return sum(
            v.numel() * v.element_size()
            for v in val
            if isinstance(v, torch.Tensor)
        )
    return 0


# ---------------------------------------------------------------------------
# GraphProfiler
# ---------------------------------------------------------------------------


class GraphProfiler(fx.Interpreter):
    """Node-by-node graph executor that collects profiling statistics.

    The profiler is designed to be used in two phases:

    1. **Warm-up** — run a few iterations to stabilise CUDA caches and JIT.
    2. **Measurement** — reset stats, run several iterations, then aggregate.

    Example
    -------
    >>> profiler = GraphProfiler(gm)
    >>> with torch.no_grad():
    ...     for _ in range(warm_up):
    ...         profiler.run(*args)
    ...     profiler.reset_stats()
    ...     for _ in range(measure):
    ...         profiler.run(*args)
    >>> profiler.aggregate_stats()
    >>> profiler.print_stats()
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        module: fx.GraphModule,
        garbage_collect_values: bool = True,
    ):
        super().__init__(module, garbage_collect_values)

        # Ordered list of all nodes and fast index lookup.
        self.node_list: List[fx.Node] = list(self.module.graph.nodes)
        self.node_to_idx: Dict[fx.Node, int] = {
            n: i for i, n in enumerate(self.node_list)
        }

        # ------------------------------------------------------------------
        # 1. Locate separator nodes (forward / backward boundaries).
        #
        # The graph_tracer inserts two identity operations:
        #   %sep            — marks the end of the forward pass
        #   %sep_backward   — marks the start of the backward pass
        # Everything between them is the loss computation.
        # ------------------------------------------------------------------

        self.sep_node: Optional[fx.Node] = None
        self.sep_bwd_node: Optional[fx.Node] = None
        self.sep_index: int = -1
        self.sep_bwd_index: int = -1

        for i, node in enumerate(self.node_list):
            if node.op == OP.CALL_FUNCTION:
                if node.target == torch.ops.separator.sep.default:
                    self.sep_node = node
                    self.sep_index = i
                elif node.target == torch.ops.separator.sep_backward.default:
                    self.sep_bwd_node = node
                    self.sep_bwd_index = i

        assert self.sep_node is not None, (
            "Could not find separator.sep node — did you wrap the loss with "
            "SEPFunction.apply()?"
        )
        assert self.sep_bwd_node is not None, (
            "Could not find separator.sep_backward node."
        )

        # ------------------------------------------------------------------
        # 2. Locate the optimizer node and extract parameter / gradient sets.
        #
        # The _fused_adam node's arguments are:
        #   args[0] = list of parameter nodes
        #   args[1] = list of gradient nodes
        #   args[2] = list of exp_avg nodes
        #   args[3] = list of exp_avg_sq nodes
        #   ...
        # ------------------------------------------------------------------

        self.optimizer_node: Optional[fx.Node] = None
        self.optimizer_index: int = -1

        for i, node in enumerate(self.node_list):
            if (
                node.op == OP.CALL_FUNCTION
                and node.target == torch.ops.aten._fused_adam.default
            ):
                self.optimizer_node = node
                self.optimizer_index = i
                break

        self.param_nodes: Set[fx.Node] = set()
        self.grad_nodes: Set[fx.Node] = set()

        if self.optimizer_node is not None:
            self.param_nodes = set(self.optimizer_node.args[0])
            self.grad_nodes = set(self.optimizer_node.args[1])

        # ------------------------------------------------------------------
        # 3. Classify every node's region.
        # ------------------------------------------------------------------

        self.node_region: Dict[fx.Node, Region] = {}
        for i, node in enumerate(self.node_list):
            if self.optimizer_node is not None and i >= self.optimizer_index:
                self.node_region[node] = Region.OPTIMIZER
            elif i >= self.sep_bwd_index:
                self.node_region[node] = Region.BACKWARD
            elif i > self.sep_index:
                self.node_region[node] = Region.LOSS
            elif i <= self.sep_index:
                self.node_region[node] = Region.FORWARD
            else:
                self.node_region[node] = Region.OTHER

        # ------------------------------------------------------------------
        # 4. Classify every node's tensor type.
        # ------------------------------------------------------------------

        self.node_types: Dict[fx.Node, NodeType] = {}
        for node in self.node_list:
            if node in self.param_nodes:
                self.node_types[node] = NodeType.PARAM
            elif node in self.grad_nodes:
                self.node_types[node] = NodeType.GRAD
            else:
                # Default; intermediates are re-classified below.
                self.node_types[node] = NodeType.OTHER

        # ------------------------------------------------------------------
        # 5. Identify intermediate activations.
        #
        # An intermediate activation is a call_function node created during
        # the forward pass that has at least one user in the backward pass.
        # These are the tensors whose memory can be reclaimed by activation
        # checkpointing.
        # ------------------------------------------------------------------

        self.intermediate_nodes: List[fx.Node] = []
        self.intermediate_info: Dict[fx.Node, IntermediateInfo] = {}

        for node in self.node_list:
            idx = self.node_to_idx[node]

            # Must be in the forward region and be a computation node.
            if idx >= self.sep_index:
                continue
            if node.op != OP.CALL_FUNCTION:
                continue
            if node in self.param_nodes:
                continue

            # Must have at least one user in the backward region.
            has_bwd_user = any(
                self.node_to_idx.get(u, -1) >= self.sep_bwd_index
                for u in node.users
            )
            if not has_bwd_user:
                continue

            # Compute lifetime endpoints.
            fwd_user_indices = [
                self.node_to_idx[u]
                for u in node.users
                if self.node_to_idx.get(u, -1) <= self.sep_index
            ]
            bwd_user_indices = [
                self.node_to_idx[u]
                for u in node.users
                if self.node_to_idx.get(u, -1) >= self.sep_bwd_index
            ]

            last_fwd = max(fwd_user_indices) if fwd_user_indices else idx
            first_bwd = min(bwd_user_indices) if bwd_user_indices else -1

            # Tensor size from FakeTensor metadata.
            mem_size = _tensor_size_bytes(node.meta.get("val", None))

            info = IntermediateInfo(
                node=node,
                memory_size=mem_size,
                last_fwd_access=last_fwd,
                first_bwd_access=first_bwd,
            )

            self.intermediate_nodes.append(node)
            self.intermediate_info[node] = info
            self.node_types[node] = NodeType.ACT

        # Build swap schedule lookup tables.
        self._intermediate_name_to_node: Dict[str, fx.Node] = {
            n.name: n for n in self.intermediate_nodes
        }

        # At step i, which intermediates should be swapped *out*?
        self._swap_out_at: Dict[int, List[str]] = {}
        for info in self.intermediate_info.values():
            self._swap_out_at.setdefault(
                info.last_fwd_access, []
            ).append(info.node.name)

        # At step i, which intermediates need to be swapped *in*?
        self._swap_in_at: Dict[int, List[str]] = {}
        for info in self.intermediate_info.values():
            if info.first_bwd_access >= 0:
                self._swap_in_at.setdefault(
                    info.first_bwd_access, []
                ).append(info.node.name)

        # ------------------------------------------------------------------
        # 6. Compute node sizes for all nodes (used by the memory chart).
        # ------------------------------------------------------------------

        self.node_sizes: Dict[str, int] = {}
        for node in self.node_list:
            self.node_sizes[node.name] = _tensor_size_bytes(
                node.meta.get("val", None)
            )

        # ------------------------------------------------------------------
        # 7. Runtime profiling accumulators.
        # ------------------------------------------------------------------

        self._node_runtimes: Dict[str, List[float]] = {}
        self._node_mem_deltas: Dict[str, List[int]] = {}
        self._swap_out_times: Dict[str, List[float]] = {}
        self._swap_in_times: Dict[str, List[float]] = {}

        # Averaged stats (populated by aggregate_stats).
        self.avg_runtimes: Dict[str, float] = {}
        self.avg_mem_deltas: Dict[str, float] = {}
        self.avg_swap_out: Dict[str, float] = {}
        self.avg_swap_in: Dict[str, float] = {}

        # CPU-side storage for swapped-out tensors during a single run.
        self._cpu_store: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ run

    def run(
        self,
        *args,
        initial_env: Optional[Dict[fx.Node, Any]] = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self._cpu_store.clear()
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    # -------------------------------------------------------------- run_node

    def run_node(self, n: fx.Node) -> Any:
        idx = self.node_to_idx[n]

        # --- Swap-in -------------------------------------------------------
        # If we are in the backward region and this step is the first backward
        # user of some intermediate, move it back from CPU to GPU so that the
        # node can consume it.

        for name in self._swap_in_at.get(idx, []):
            if name not in self._cpu_store:
                continue
            cpu_tensor = self._cpu_store.pop(name)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gpu_tensor = cpu_tensor.cuda()
            end.record()
            torch.cuda.synchronize()

            self._swap_in_times.setdefault(name, []).append(
                start.elapsed_time(end)
            )

            # Write the tensor back into the interpreter's environment.
            source_node = self._intermediate_name_to_node.get(name)
            if source_node is not None:
                self.env[source_node] = gpu_tensor

        # --- Execute the node and measure timing + memory ------------------
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        mem_before = torch.cuda.memory_allocated()

        start_event.record()
        result = super().run_node(n)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        mem_delta = torch.cuda.memory_allocated() - mem_before

        self._node_runtimes.setdefault(n.name, []).append(elapsed_ms)
        self._node_mem_deltas.setdefault(n.name, []).append(mem_delta)

        # --- Swap-out ------------------------------------------------------
        # If this step is the last forward user of some intermediate, move it
        # to CPU to free GPU memory.

        for name in self._swap_out_at.get(idx, []):
            source_node = self._intermediate_name_to_node.get(name)
            if source_node is None or source_node not in self.env:
                continue
            gpu_tensor = self.env[source_node]
            if not isinstance(gpu_tensor, torch.Tensor) or not gpu_tensor.is_cuda:
                continue

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            cpu_tensor = gpu_tensor.cpu()
            end.record()
            torch.cuda.synchronize()

            self._swap_out_times.setdefault(name, []).append(
                start.elapsed_time(end)
            )
            self._cpu_store[name] = cpu_tensor

        return result

    # -------------------------------------------------------- aggregate_stats

    def aggregate_stats(self) -> None:
        """Average runtime and memory measurements over all recorded runs."""
        for name, runs in self._node_runtimes.items():
            self.avg_runtimes[name] = sum(runs) / len(runs) if runs else 0.0
        for name, mems in self._node_mem_deltas.items():
            self.avg_mem_deltas[name] = sum(mems) / len(mems) if mems else 0.0
        for name, times in self._swap_out_times.items():
            self.avg_swap_out[name] = sum(times) / len(times) if times else 0.0
        for name, times in self._swap_in_times.items():
            self.avg_swap_in[name] = sum(times) / len(times) if times else 0.0

        # Populate IntermediateInfo with averaged measurements.
        for node, info in self.intermediate_info.items():
            name = node.name
            info.recompute_cost_ms = self.avg_runtimes.get(name, 0.0)
            info.swap_out_time_ms = self.avg_swap_out.get(name, 0.0)
            info.swap_in_time_ms = self.avg_swap_in.get(name, 0.0)

            # Inactive time = wall-clock time between last forward use and
            # first backward use, approximated as the sum of node runtimes
            # over the interval (last_fwd_access, first_bwd_access).
            inactive_ms = 0.0
            for i in range(info.last_fwd_access + 1, info.first_bwd_access):
                step_name = self.node_list[i].name
                inactive_ms += self.avg_runtimes.get(step_name, 0.0)
            info.inactive_time_ms = inactive_ms

    # ---------------------------------------------------------- reset_stats

    def reset_stats(self) -> None:
        """Clear all accumulated measurements (call between warm-up and
        measurement phases)."""
        self._node_runtimes.clear()
        self._node_mem_deltas.clear()
        self._swap_out_times.clear()
        self._swap_in_times.clear()
        self.avg_runtimes.clear()
        self.avg_mem_deltas.clear()
        self.avg_swap_out.clear()
        self.avg_swap_in.clear()

    # ---------------------------------------------------------- print_stats

    def print_stats(self) -> None:
        """Print a comprehensive profiling report.

        The report contains four sections:
        1. Operation summary table (per-node timing and memory).
        2. Tensor categorization (counts and totals by role).
        3. Intermediate activation lifetime table.
        4. Peak memory estimate.
        """

        # === Section 1: Operation summary ==================================
        print("\n" + "=" * 90)
        print("OPERATION SUMMARY")
        print("=" * 90)
        print(
            f"{'#':<5} {'Node':<35} {'Region':<10} "
            f"{'Time(ms)':>10} {'Mem(B)':>12}"
        )
        print("-" * 90)

        for i, node in enumerate(self.node_list):
            name = node.name[:34]
            region = self.node_region.get(node, Region.OTHER).name[:9]
            rt = self.avg_runtimes.get(node.name, 0.0)
            md = self.avg_mem_deltas.get(node.name, 0.0)
            print(
                f"{i:<5} {name:<35} {region:<10} "
                f"{rt:>10.3f} {md:>+12.0f}"
            )

        # === Section 2: Tensor categorization ==============================
        print("\n" + "=" * 90)
        print("TENSOR CATEGORIZATION")
        print("=" * 90)

        role_counts: Dict[NodeType, int] = {}
        role_bytes: Dict[NodeType, int] = {}

        for node, ntype in self.node_types.items():
            role_counts[ntype] = role_counts.get(ntype, 0) + 1
            role_bytes[ntype] = (
                role_bytes.get(ntype, 0)
                + self.node_sizes.get(node.name, 0)
            )

        for ntype in [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER]:
            count = role_counts.get(ntype, 0)
            nbytes = role_bytes.get(ntype, 0)
            print(
                f"  {ntype.name:<18} count={count:<6} "
                f"total={nbytes / 1024:>10.2f} KB"
            )

        # === Section 3: Intermediate activation lifetimes ==================
        print("\n" + "=" * 90)
        print("INTERMEDIATE ACTIVATION LIFETIMES")
        print("=" * 90)
        print(
            f"{'Name':<30} {'Size(KB)':>10} {'LastFwd':>8} {'FirstBwd':>9} "
            f"{'Lifetime':>9} {'Recomp(ms)':>11} "
            f"{'SwapOut(ms)':>12} {'SwapIn(ms)':>11}"
        )
        print("-" * 110)

        total_act_mem = 0
        for node in self.intermediate_nodes:
            info = self.intermediate_info[node]
            name = node.name[:29]
            size_kb = info.memory_size / 1024
            lifetime = info.first_bwd_access - info.last_fwd_access
            total_act_mem += info.memory_size
            print(
                f"{name:<30} {size_kb:>10.2f} {info.last_fwd_access:>8} "
                f"{info.first_bwd_access:>9} {lifetime:>9} "
                f"{info.recompute_cost_ms:>11.4f} "
                f"{info.swap_out_time_ms:>12.4f} "
                f"{info.swap_in_time_ms:>11.4f}"
            )

        print(f"\nTotal intermediate activations: {len(self.intermediate_nodes)}")
        print(
            f"Total activation memory: {total_act_mem / 1024:.2f} KB "
            f"({total_act_mem / (1024 * 1024):.2f} MB)"
        )

        # === Section 4: Peak memory estimate ===============================
        print("\n" + "=" * 90)
        print("PEAK MEMORY ESTIMATE")
        print("=" * 90)

        live_mem = self._compute_live_memory_timeline()
        if live_mem:
            peak_step = max(range(len(live_mem)), key=lambda t: live_mem[t])
            peak_bytes = live_mem[peak_step]
            peak_node = (
                self.node_list[peak_step]
                if peak_step < len(self.node_list)
                else None
            )
            print(
                f"  Peak memory: {peak_bytes / (1024 * 1024):.2f} MB "
                f"at step {peak_step}"
                + (f" ({peak_node.name})" if peak_node else "")
            )
        else:
            print("  (no memory data)")

        print("=" * 90 + "\n")

    # ---------------------------------------- live memory timeline helpers

    def _compute_live_memory_timeline(self) -> List[int]:
        """Compute total live memory (bytes) at each step in the timeline.

        A tensor produced at step ``p`` by a call_function or placeholder node
        is live at step ``t`` if ``p <= t <= last_use``, where ``last_use`` is
        the maximum index among the node's users.

        Returns
        -------
        list of int
            Total live bytes at each step.
        """
        n_steps = len(self.node_list)
        timeline = [0] * n_steps

        for node in self.node_list:
            if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
                continue
            size = self.node_sizes.get(node.name, 0)
            if size == 0:
                continue

            produced_at = self.node_to_idx[node]
            user_indices = [
                self.node_to_idx[u]
                for u in node.users
                if u in self.node_to_idx
            ]
            last_use = max(user_indices) if user_indices else produced_at

            for t in range(produced_at, min(last_use + 1, n_steps)):
                timeline[t] += size

        return timeline

    def _compute_live_memory_timeline_by_role(
        self,
    ) -> Dict[NodeType, List[int]]:
        """Like _compute_live_memory_timeline but broken down by NodeType.

        Returns
        -------
        dict mapping NodeType -> list of int
            Per-role live memory at each step.
        """
        n_steps = len(self.node_list)
        by_role: Dict[NodeType, List[int]] = {
            nt: [0] * n_steps for nt in NodeType
        }

        for node in self.node_list:
            if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
                continue
            size = self.node_sizes.get(node.name, 0)
            if size == 0:
                continue

            role = self.node_types.get(node, NodeType.OTHER)
            produced_at = self.node_to_idx[node]
            user_indices = [
                self.node_to_idx[u]
                for u in node.users
                if u in self.node_to_idx
            ]
            last_use = max(user_indices) if user_indices else produced_at

            for t in range(produced_at, min(last_use + 1, n_steps)):
                by_role[role][t] += size

        return by_role
