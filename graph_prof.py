"""
This code extends torch.fx.Interpreter to execute a traced FX graph node-by-node.
It classifies tensors, computes the lifetimes of activations, and measures per-node timing and memory.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch
import torch.fx as fx


class OP(str, Enum):
    CALL_FUNCTION = "call_function" # an actual computation (relu, ...)
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method" 
    GET_ATTR = "get_attr"
    OUTPUT = "output" # return value of the graph
    PLACEHOLDER = "placeholder" # an input to the graph (parameters, optimizer states, batch data)


class NodeType(Enum):
    PARAM = 0
    ACT = 1
    GRAD = 2
    OPTIMIZER_STATE = 3
    OTHER = 4


class Region(Enum): # based on a node's position wrt to sep_index and sep_bwd_index.
    FORWARD = 0
    LOSS = 1
    BACKWARD = 2
    OPTIMIZER = 3
    OTHER = 4


@dataclass
class IntermediateInfo:
    node: fx.Node
    # filled during __init__
    memory_size: int = 0
    last_fwd_access: int = -1
    first_bwd_access: int = -1
    # filled after profiling run
    inactive_time_ms: float = 0.0
    recompute_cost_ms: float = 0.0
    swap_out_time_ms: float = 0.0
    swap_in_time_ms: float = 0.0


def _tensor_size_bytes(val: Any) -> int:
    if isinstance(val, torch.Tensor):
        return val.numel() * val.element_size()
    if isinstance(val, (tuple, list)):
        return sum(v.numel() * v.element_size() for v in val if isinstance(v, torch.Tensor))
    return 0


class GraphProfiler(fx.Interpreter):
    # We override __init__, run, and run_node of fx.Interpreter to add our profiling logic on top.
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        self.node_list: List[fx.Node] = list(self.module.graph.nodes)
        self.node_to_idx: Dict[fx.Node, int] = {n: i for i, n in enumerate(self.node_list)}

        # 1. Find separator nodes (%sep, %sep_backward). These partition the entire graph.
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

        assert self.sep_node is not None, "Could not find sep node"
        assert self.sep_bwd_node is not None, "Could not find sep_backward node"

        # 2. Find optimizer boundary and identify parameters/gradients.
        #    Strategy A: _fused_adam node (fused=True) => read params/grads from args.
        #    Strategy B: _foreach_* ops (foreach=True) => params = placeholders with users in both forward and optimizer regions.
        self.optimizer_node: Optional[fx.Node] = None
        self.optimizer_index: int = -1

        for i, node in enumerate(self.node_list):
            if node.op == OP.CALL_FUNCTION and node.target == torch.ops.aten._fused_adam.default:
                self.optimizer_node = node
                self.optimizer_index = i
                break

        if self.optimizer_index < 0:
            for i, node in enumerate(self.node_list):
                if i <= self.sep_bwd_index:
                    continue
                if node.op == OP.CALL_FUNCTION and "_foreach" in str(node.target):
                    self.optimizer_index = i
                    break

        self.param_nodes: Set[fx.Node] = set()
        self.grad_nodes: Set[fx.Node] = set()

        #  Identify parameters: for each PLACEHOLDER node, check if it has users in both forward AND optimizer
        if self.optimizer_node is not None:
            self.param_nodes = set(self.optimizer_node.args[0])
            self.grad_nodes = set(self.optimizer_node.args[1])
        elif self.optimizer_index >= 0:
            for node in self.node_list:
                if node.op != OP.PLACEHOLDER:
                    continue
                user_indices = [self.node_to_idx[u] for u in node.users if u in self.node_to_idx]
                has_fwd = any(idx < self.sep_index for idx in user_indices)
                has_opt = any(idx >= self.optimizer_index for idx in user_indices)
                if has_fwd and has_opt:
                    self.param_nodes.add(node)

        # 3. Assign each node a region.
        self.node_region: Dict[fx.Node, Region] = {}
        for i, node in enumerate(self.node_list):
            if self.optimizer_index >= 0 and i >= self.optimizer_index:
                self.node_region[node] = Region.OPTIMIZER
            elif i >= self.sep_bwd_index:
                self.node_region[node] = Region.BACKWARD
            elif i > self.sep_index:
                self.node_region[node] = Region.LOSS
            elif i <= self.sep_index:
                self.node_region[node] = Region.FORWARD
            else:
                self.node_region[node] = Region.OTHER

        # 4. Classify each node's tensor type.

        # We loop through all placeholder nodes that aren't already identified as params or grads. 
        # For each, are ALL its users in the optimizer region? If yes, it's an optimizer state.
        self.optimizer_state_nodes: Set[fx.Node] = set()
        if self.optimizer_index >= 0:
            for node in self.node_list:
                if node.op != OP.PLACEHOLDER or node in self.param_nodes or node in self.grad_nodes:
                    continue
                user_indices = [self.node_to_idx[u] for u in node.users if u in self.node_to_idx]
                if user_indices and all(idx >= self.optimizer_index for idx in user_indices):
                    self.optimizer_state_nodes.add(node)

        self.node_types: Dict[fx.Node, NodeType] = {}
        for node in self.node_list:
            if node in self.param_nodes:
                self.node_types[node] = NodeType.PARAM
            elif node in self.grad_nodes:
                self.node_types[node] = NodeType.GRAD
            elif node in self.optimizer_state_nodes:
                self.node_types[node] = NodeType.OPTIMIZER_STATE
            else:
                self.node_types[node] = NodeType.OTHER

        # 5. Identify intermediate activations: forward call_function nodes with backward users.
        self.intermediate_nodes: List[fx.Node] = []
        self.intermediate_info: Dict[fx.Node, IntermediateInfo] = {}

        for node in self.node_list:
            idx = self.node_to_idx[node]
            if idx >= self.sep_index or node.op != OP.CALL_FUNCTION or node in self.param_nodes:
                continue
            # Does this node have at least one user in the backward region? If not, skip.
            if not any(self.node_to_idx.get(u, -1) >= self.sep_bwd_index for u in node.users):
                continue

            fwd_users = [self.node_to_idx[u] for u in node.users if self.node_to_idx.get(u, -1) <= self.sep_index]
            bwd_users = [self.node_to_idx[u] for u in node.users if self.node_to_idx.get(u, -1) >= self.sep_bwd_index]

            info = IntermediateInfo(
                node=node,
                memory_size=_tensor_size_bytes(node.meta.get("val", None)),
                last_fwd_access=max(fwd_users) if fwd_users else idx,
                first_bwd_access=min(bwd_users) if bwd_users else -1,
            )
            self.intermediate_nodes.append(node)
            self.intermediate_info[node] = info
            self.node_types[node] = NodeType.ACT

        # Build swap schedule: at each step, which intermediates to swap out/in.

        # EXAMPLE: consider relu with last_fwd=87,  first_bwd=121 => 
        # _swap_out_at = {87: ["relu"]}
        # _swap_in_at = {121: ["relu"]}

        self._intermediate_name_to_node: Dict[str, fx.Node] = {n.name: n for n in self.intermediate_nodes}

        # when should I move this tensor to CPU? at its last_fwd_access step
        self._swap_out_at: Dict[int, List[str]] = {}
        for info in self.intermediate_info.values():
            self._swap_out_at.setdefault(info.last_fwd_access, []).append(info.node.name)
        # when should I bring this tensor back? at its first_bwd_access step
        self._swap_in_at: Dict[int, List[str]] = {}
        for info in self.intermediate_info.values():
            if info.first_bwd_access >= 0:
                self._swap_in_at.setdefault(info.first_bwd_access, []).append(info.node.name)

        # 6. Node sizes for memory chart.
        self.node_sizes: Dict[str, int] = {}
        for node in self.node_list:
            self.node_sizes[node.name] = _tensor_size_bytes(node.meta.get("val", None))

        # 7. Runtime profiling accumulators.
        self._node_runtimes: Dict[str, List[float]] = {}
        self._node_mem_deltas: Dict[str, List[int]] = {}
        self._swap_out_times: Dict[str, List[float]] = {}
        self._swap_in_times: Dict[str, List[float]] = {}
        self.avg_runtimes: Dict[str, float] = {}
        self.avg_mem_deltas: Dict[str, float] = {}
        self.avg_swap_out: Dict[str, float] = {}
        self.avg_swap_in: Dict[str, float] = {}
        self._cpu_store: Dict[str, torch.Tensor] = {}

    def run(self, *args, initial_env: Optional[Dict[fx.Node, Any]] = None, enable_io_processing: bool = True) -> Any:
        self._cpu_store.clear()
        return super().run(*args, initial_env=initial_env, enable_io_processing=enable_io_processing)

    # this helps collect four numbers phase 2 needs:  
    # (1) How long does this node take to execute? 
    # (2) memory_size, memory delta confirms our static size estimate
    # (3) How long does the PCIe transfer take for each tensor? 
    # (4) How long does the tensor sit idle? inactive_time_ms

    def run_node(self, n: fx.Node) -> Any:


        idx = self.node_to_idx[n]

        # Swap-in: bring intermediates back from CPU before their first backward use.
        # Check the timetable: does step idx need any tensors swapped back from CPU?
        for name in self._swap_in_at.get(idx, []):
            if name not in self._cpu_store:
                continue
            # Grab the tensor from CPU storage and remove it from the dict => free cpu memory
            cpu_tensor = self._cpu_store.pop(name)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            # This copies the tensor's bytes from CPU RAM to GPU memory
            gpu_tensor = cpu_tensor.cuda()
            end.record()
            # Python blocks here until the GPU finishes everything 
            torch.cuda.synchronize()
            self._swap_in_times.setdefault(name, []).append(start.elapsed_time(end))
            #  We need the node object because self.env is keyed by node
            source_node = self._intermediate_name_to_node.get(name)
            # Write the GPU tensor into the interpreter's environment
            if source_node is not None:
                self.env[source_node] = gpu_tensor #  self.env maps each node to the actual tensor it produced

        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        mem_before = torch.cuda.memory_allocated()
        start_event.record()

        # super() is fx.Interpreter. Its run_node does three things:
        # (1) Reads n's input nodes from self.env
        # (2) Calls the ATen op
        # (3) Stores the node output tensor in self.env[n] — now self.env[relu_node] holds the relu output
        result = super().run_node(n)
        end_event.record()

        # Block Python until the GPU finishes the kernel and records end_event
        torch.cuda.synchronize()

        self._node_runtimes.setdefault(n.name, []).append(start_event.elapsed_time(end_event))
        self._node_mem_deltas.setdefault(n.name, []).append(torch.cuda.memory_allocated() - mem_before)

        # Swap-out: move intermediates to CPU after their last forward use. At step idx, which tensors should be moved to CPU?
        for name in self._swap_out_at.get(idx, []):
            source_node = self._intermediate_name_to_node.get(name)
            if source_node is None or source_node not in self.env:
                continue
            gpu_tensor = self.env[source_node]
            if not isinstance(gpu_tensor, torch.Tensor) or not gpu_tensor.is_cuda:
                continue
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            cpu_tensor = gpu_tensor.cpu()
            end.record()
            torch.cuda.synchronize()
            self._swap_out_times.setdefault(name, []).append(start.elapsed_time(end))
            self._cpu_store[name] = cpu_tensor

        return result

    def aggregate_stats(self) -> None:
        """Average measurements over all recorded runs and populate IntermediateInfo."""
        for name, runs in self._node_runtimes.items():
            self.avg_runtimes[name] = sum(runs) / len(runs) if runs else 0.0
        for name, mems in self._node_mem_deltas.items():
            self.avg_mem_deltas[name] = sum(mems) / len(mems) if mems else 0.0
        for name, times in self._swap_out_times.items():
            self.avg_swap_out[name] = sum(times) / len(times) if times else 0.0
        for name, times in self._swap_in_times.items():
            self.avg_swap_in[name] = sum(times) / len(times) if times else 0.0

        for node, info in self.intermediate_info.items():
            info.recompute_cost_ms = self.avg_runtimes.get(node.name, 0.0)
            info.swap_out_time_ms = self.avg_swap_out.get(node.name, 0.0)
            info.swap_in_time_ms = self.avg_swap_in.get(node.name, 0.0)
            info.inactive_time_ms = sum(
                self.avg_runtimes.get(self.node_list[i].name, 0.0)
                for i in range(info.last_fwd_access + 1, info.first_bwd_access)
            )

    def reset_stats(self) -> None:
        """Clear all accumulated measurements."""
        self._node_runtimes.clear()
        self._node_mem_deltas.clear()
        self._swap_out_times.clear()
        self._swap_in_times.clear()
        self.avg_runtimes.clear()
        self.avg_mem_deltas.clear()
        self.avg_swap_out.clear()
        self.avg_swap_in.clear()

    def print_stats(self) -> None:
        """Print operation summary, tensor categorization, activation lifetimes, and peak memory."""

        # Section 1: Operation summary
        print("\n" + "=" * 90)
        print("OPERATION SUMMARY")
        print("=" * 90)
        print(f"{'#':<5} {'Node':<35} {'Region':<10} {'Time(ms)':>10} {'Mem(B)':>12}")
        print("-" * 90)
        for i, node in enumerate(self.node_list):
            region = self.node_region.get(node, Region.OTHER).name[:9]
            rt = self.avg_runtimes.get(node.name, 0.0)
            md = self.avg_mem_deltas.get(node.name, 0.0)
            print(f"{i:<5} {node.name[:34]:<35} {region:<10} {rt:>10.3f} {md:>+12.0f}")

        # Section 2: Tensor categorization
        print("\n" + "=" * 90)
        print("TENSOR CATEGORIZATION")
        print("=" * 90)
        role_counts: Dict[NodeType, int] = {}
        role_bytes: Dict[NodeType, int] = {}
        for node, ntype in self.node_types.items():
            role_counts[ntype] = role_counts.get(ntype, 0) + 1
            role_bytes[ntype] = role_bytes.get(ntype, 0) + self.node_sizes.get(node.name, 0)
        for ntype in [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OPTIMIZER_STATE, NodeType.OTHER]:
            print(f"  {ntype.name:<18} count={role_counts.get(ntype, 0):<6} total={role_bytes.get(ntype, 0) / 1024:>10.2f} KB")

        # Section 3: Intermediate activation lifetimes
        print("\n" + "=" * 90)
        print("INTERMEDIATE ACTIVATION LIFETIMES")
        print("=" * 90)
        print(f"{'Name':<30} {'Size(KB)':>10} {'LastFwd':>8} {'FirstBwd':>9} {'Lifetime':>9} {'Recomp(ms)':>11} {'SwapOut(ms)':>12} {'SwapIn(ms)':>11}")
        print("-" * 110)
        total_act_mem = 0
        for node in self.intermediate_nodes:
            info = self.intermediate_info[node]
            lifetime = info.first_bwd_access - info.last_fwd_access
            total_act_mem += info.memory_size
            print(f"{node.name[:29]:<30} {info.memory_size / 1024:>10.2f} {info.last_fwd_access:>8} {info.first_bwd_access:>9} {lifetime:>9} {info.recompute_cost_ms:>11.4f} {info.swap_out_time_ms:>12.4f} {info.swap_in_time_ms:>11.4f}")
        print(f"\nTotal intermediate activations: {len(self.intermediate_nodes)}")
        print(f"Total activation memory: {total_act_mem / 1024:.2f} KB ({total_act_mem / (1024 * 1024):.2f} MB)")

        # Section 4: Peak memory estimate
        print("\n" + "=" * 90)
        print("PEAK MEMORY ESTIMATE")
        print("=" * 90)
        live_mem = self._compute_live_memory_timeline()
        if live_mem:
            peak_step = max(range(len(live_mem)), key=lambda t: live_mem[t])
            peak_bytes = live_mem[peak_step]
            peak_node = self.node_list[peak_step] if peak_step < len(self.node_list) else None
            print(f"  Peak memory: {peak_bytes / (1024 * 1024):.2f} MB at step {peak_step}" + (f" ({peak_node.name})" if peak_node else ""))
        else:
            print("  (no memory data)")
        print("=" * 90 + "\n")

    def _compute_live_memory_timeline(self) -> List[int]:
        """Total live memory (bytes) at each step. A tensor is live from production to last use."""
        n_steps = len(self.node_list)
        timeline = [0] * n_steps
        for node in self.node_list:
            if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
                continue
            size = self.node_sizes.get(node.name, 0)
            if size == 0:
                continue
            produced_at = self.node_to_idx[node]
            user_indices = [self.node_to_idx[u] for u in node.users if u in self.node_to_idx]
            last_use = max(user_indices) if user_indices else produced_at
            for t in range(produced_at, min(last_use + 1, n_steps)):
                timeline[t] += size
        return timeline

    def _compute_live_memory_timeline_by_role(self) -> Dict[NodeType, List[int]]:
        """Like _compute_live_memory_timeline but broken down by NodeType."""
        n_steps = len(self.node_list)
        by_role: Dict[NodeType, List[int]] = {nt: [0] * n_steps for nt in NodeType}
        for node in self.node_list:
            if node.op not in (OP.CALL_FUNCTION, OP.PLACEHOLDER):
                continue
            size = self.node_sizes.get(node.name, 0)
            if size == 0:
                continue
            role = self.node_types.get(node, NodeType.OTHER)
            produced_at = self.node_to_idx[node]
            user_indices = [self.node_to_idx[u] for u in node.users if u in self.node_to_idx]
            last_use = max(user_indices) if user_indices else produced_at
            for t in range(produced_at, min(last_use + 1, n_steps)):
                by_role[role][t] += size
        return by_role
