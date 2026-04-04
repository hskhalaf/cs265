from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .graph import (
    ComputationalGraph,
    GraphEdge,
    GraphNode,
    OpPhase,
    ProfileResult,
    TensorRole,
)
from .memory import MemoryMonitor, MemorySnapshot
from .tensor_registry import TensorRegistry


class HookManager:
    """
    Installs and manages all profiling hooks on a model.

    Three hook slots per leaf module:
      - forward_pre:  records node_id, start time, and memory snapshot
      - forward_post: completes timing, registers inputs/outputs, builds GraphNode
      - backward:     captures gradient tensors and builds a BACKWARD GraphNode

    _pending maps id(module) -> (node_id, t0, mem_before) to correlate pre/post
    hooks for the same call. PyTorch fires pre-hooks DFS-pre-order and post-hooks
    DFS-post-order, so nested modules naturally pair correctly.
    """

    def __init__(self, graph: ComputationalGraph, registry: TensorRegistry, monitor: MemoryMonitor) -> None:
        self._graph = graph
        self._registry = registry
        self._monitor = monitor
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._pending: Dict[int, Tuple[int, float, MemorySnapshot]] = {}

    def attach(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            # Skip container modules — they add no computation and their hooks
            # fire out-of-order with respect to the topological sort.
            if len(list(module.children())) > 0:
                continue
            fqn = name if name else "root"
            h1 = module.register_forward_pre_hook(self._make_fwd_pre())
            h2 = module.register_forward_hook(self._make_fwd_post(fqn))
            h3 = module.register_full_backward_hook(self._make_bwd(fqn))
            self._handles.extend([h1, h2, h3])

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # Hook factories

    def _make_fwd_pre(self):
        def hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            node_id = self._graph.next_node_id()
            before = self._monitor.snapshot()
            t0 = time.perf_counter()
            self._pending[id(module)] = (node_id, t0, before)
        return hook

    def _make_fwd_post(self, fqn: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], outputs: Any) -> None:
            pending = self._pending.pop(id(module), None)
            if pending is None:
                return
            node_id, t0, before = pending

            t1 = time.perf_counter()
            after = self._monitor.snapshot()
            peak = torch.cuda.max_memory_allocated(self._monitor.device) if self._monitor.use_cuda else 0

            in_ids = []
            for t in _flatten_tensors(inputs):
                meta = self._registry.get_or_create(t)
                in_ids.append(meta.tensor_id)

            for pname, param in module.named_parameters(recurse=False):
                self._registry.get_or_create(param, TensorRole.PARAMETER, pname)

            # Use force_create so uniform-width networks always produce distinct
            # activation entries even when PyTorch reuses freed memory.
            out_ids = []
            for i, t in enumerate(_flatten_tensors(outputs)):
                role = TensorRole.ACTIVATION if t.grad_fn is not None else TensorRole.OTHER
                name = f"{fqn}.out{i}" if i > 0 else f"{fqn}.output"
                meta = self._registry.force_create(t, role, name)
                out_ids.append(meta.tensor_id)

            profile = ProfileResult(
                wall_time_ms=(t1 - t0) * 1000.0,
                gpu_memory_bytes=after.gpu_allocated_bytes - before.gpu_allocated_bytes,
                cpu_memory_bytes=after.cpu_traced_bytes - before.cpu_traced_bytes,
                peak_gpu_bytes=peak,
            )
            node = GraphNode(
                node_id=node_id,
                op_name=f"{type(module).__name__}.forward",
                phase=OpPhase.FORWARD,
                module_fqn=fqn,
                input_tensor_ids=in_ids,
                output_tensor_ids=out_ids,
                profile=profile,
            )
            self._graph.add_node(node)

            for in_id in in_ids:
                src = self._find_producer(in_id)
                if src != -1:
                    self._graph.add_edge(GraphEdge(src_node_id=src, dst_node_id=node_id, tensor_id=in_id))

        return hook

    def _make_bwd(self, fqn: str):
        def hook(module: nn.Module, grad_input: Tuple[Optional[torch.Tensor], ...], grad_output: Tuple[Optional[torch.Tensor], ...]) -> None:
            before = self._monitor.snapshot()
            t0 = time.perf_counter()

            # grad_output: gradients flowing into this module (from the layer ahead in forward)
            in_ids = []
            for g in _flatten_tensors(grad_output):
                meta = self._registry.get_or_create(g, TensorRole.GRADIENT)
                in_ids.append(meta.tensor_id)

            # grad_input: gradients flowing out (to the layer behind in forward)
            out_ids = []
            for g in _flatten_tensors(grad_input):
                if g is not None:
                    meta = self._registry.get_or_create(g, TensorRole.GRADIENT)
                    out_ids.append(meta.tensor_id)

            # param.grad may not be written yet at hook time; we do a best-effort
            # capture here and collect again after loss.backward() in executor.run().
            for pname, param in module.named_parameters(recurse=False):
                if param.grad is not None:
                    meta = self._registry.mark_gradient(param, param.grad, pname)
                    if meta.tensor_id not in out_ids:
                        out_ids.append(meta.tensor_id)

            t1 = time.perf_counter()
            after = self._monitor.snapshot()

            profile = ProfileResult(
                wall_time_ms=(t1 - t0) * 1000.0,
                gpu_memory_bytes=after.gpu_allocated_bytes - before.gpu_allocated_bytes,
                cpu_memory_bytes=0,
                peak_gpu_bytes=0,
            )
            node = GraphNode(
                node_id=self._graph.next_node_id(),
                op_name=f"{type(module).__name__}.backward",
                phase=OpPhase.BACKWARD,
                module_fqn=fqn,
                input_tensor_ids=in_ids,
                output_tensor_ids=out_ids,
                profile=profile,
            )
            self._graph.add_node(node)

        return hook

    def _find_producer(self, tensor_id: int) -> int:
        """Return the node_id of the most recent node that produced tensor_id, or -1."""
        for node in reversed(list(self._graph.nodes.values())):
            if tensor_id in node.output_tensor_ids:
                return node.node_id
        return -1


class WrappedOptimizer:
    """
    Thin wrapper around any torch.optim.Optimizer.

    Intercepts step() to measure timing/memory, detect newly allocated optimizer
    state tensors (handles Adam's lazy first-step init), and emit an OPTIMIZER
    phase GraphNode.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, graph: ComputationalGraph, registry: TensorRegistry, monitor: MemoryMonitor) -> None:
        self._opt = optimizer
        self._graph = graph
        self._registry = registry
        self._monitor = monitor

    def step(self, closure=None):
        before = self._monitor.snapshot()
        t0 = time.perf_counter()

        pre_state = self._collect_state_ptrs()
        result = self._opt.step(closure)
        post_state = self._collect_state_ptrs()

        t1 = time.perf_counter()
        after = self._monitor.snapshot()

        # Detect state tensors created during this step (handles lazy init)
        pre_keys = {(pid, k) for pid, k, _ in pre_state}
        opt_state_ids = []
        for param_id, key, tensor in post_state:
            if (param_id, key) not in pre_keys:
                meta = self._registry.mark_optimizer_state(tensor, f"opt_{key}")
                opt_state_ids.append(meta.tensor_id)

        updated_param_ids = []
        for group in self._opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    meta = self._registry.get_or_create(p, TensorRole.PARAMETER)
                    updated_param_ids.append(meta.tensor_id)

        profile = ProfileResult(
            wall_time_ms=(t1 - t0) * 1000.0,
            gpu_memory_bytes=after.gpu_allocated_bytes - before.gpu_allocated_bytes,
            cpu_memory_bytes=0,
            peak_gpu_bytes=0,
        )
        node = GraphNode(
            node_id=self._graph.next_node_id(),
            op_name=f"{type(self._opt).__name__}.step",
            phase=OpPhase.OPTIMIZER,
            module_fqn="optimizer",
            input_tensor_ids=updated_param_ids,
            output_tensor_ids=opt_state_ids + updated_param_ids,
            profile=profile,
        )
        self._graph.add_node(node)
        return result

    def _collect_state_ptrs(self) -> List[Tuple[int, str, torch.Tensor]]:
        """Return (param_id, state_key, tensor) for every tensor in optimizer.state."""
        result = []
        for group in self._opt.param_groups:
            for p in group["params"]:
                for k, v in self._opt.state.get(p, {}).items():
                    if isinstance(v, torch.Tensor):
                        result.append((id(p), k, v))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._opt, name)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._opt.param_groups

    @property
    def state(self):
        return self._opt.state


def _flatten_tensors(args: Any) -> List[torch.Tensor]:
    """Recursively extract all tensors from nested tuples/lists."""
    out: List[torch.Tensor] = []
    if isinstance(args, torch.Tensor):
        out.append(args)
    elif isinstance(args, (tuple, list)):
        for a in args:
            out.extend(_flatten_tensors(a))
    return out
