from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn as nn
from .graph import ComputationalGraph, OpPhase, TensorRole
from .hooks import HookManager, WrappedOptimizer
from .memory import MemoryMonitor
from .tensor_registry import TensorRegistry, compute_lifetimes


class ProfileReport:
    def __init__(self, graph: ComputationalGraph, registry: TensorRegistry) -> None:
        self.graph = graph
        self.registry = registry

    def print_summary(self) -> None:
        nodes = self.graph.nodes_in_topo_order()
        print("\n" + "=" * 80)
        print("COMPUTATIONAL GRAPH OPERATION SUMMARY")
        print("=" * 80)
        header = f"{'#':<4} {'Op':30} {'Phase':10} {'Module':25} {'Time(ms)':>10} {'Mem(B)':>12}"
        print(header)
        print("-" * 80)
        for seq, node in enumerate(nodes):
            p = node.profile
            time_ms = f"{p.wall_time_ms:.3f}" if p else "n/a"
            mem_b = f"{p.gpu_memory_bytes:+d}" if p else "n/a"
            print(
                f"{seq:<4} {node.op_name:30} {node.phase.name:10} "
                f"{node.module_fqn:25} {time_ms:>10} {mem_b:>12}"
            )

        print("\n" + "=" * 80)
        print("TENSOR CATEGORIZATION SUMMARY")
        print("=" * 80)
        for role in TensorRole:
            tensors = self.registry.all_by_role(role)
            if tensors:
                print(f"\n  {role.name} ({len(tensors)}):")
                for meta in tensors:
                    size_kb = meta.nbytes / 1024
                    print(f"    {meta.name:25} shape={str(meta.shape):20} {size_kb:.2f} KB")

        print("\n" + "=" * 80)
        print("ACTIVATION LIFETIME (STATIC ANALYSIS)")
        print("=" * 80)
        activations = self.registry.all_by_role(TensorRole.ACTIVATION)
        if activations:
            header2 = f"  {'Name':25} {'First Use':>10} {'Last Use':>10} {'Live Ops':>10} {'Size(KB)':>10}"
            print(header2)
            print("  " + "-" * 60)
            for meta in activations:
                first = meta.first_use_op if meta.first_use_op is not None else "?"
                last = meta.last_use_op if meta.last_use_op is not None else "?"
                live = last - first + 1 if isinstance(first, int) and isinstance(last, int) else "?"
                size_kb = meta.nbytes / 1024
                print(f"  {meta.name:25} {str(first):>10} {str(last):>10} {str(live):>10} {size_kb:>10.2f}")
        else:
            print("  No activations tracked.")

        print()


class ProfilerExecutor:
    """
    Orchestrates graph building and profiling.
    Usage:
        profiler = ProfilerExecutor(model, optimizer, device)
        profiler.run(X, Y, nn.MSELoss())
        report = profiler.get_report()
        report.print_summary()
        profiler.visualize("memory_breakdown.png")
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = next(model.parameters()).device
        self.model = model
        self.device = device

        self._graph = ComputationalGraph()
        self._registry = TensorRegistry(model)
        self._monitor = MemoryMonitor(device)
        self._hooks = HookManager(self._graph, self._registry, self._monitor)
        self._hooks.attach(model)
        self._opt = WrappedOptimizer(optimizer, self._graph, self._registry, self._monitor)
        self._ran = False

    def run(self, X: torch.Tensor, Y: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Execute one full training iteration and collect profiling data."""
        if self._ran:
            raise RuntimeError(
                "ProfilerExecutor.run() was already called. "
                "Create a new instance to profile another iteration."
            )

        self._registry.get_or_create(X, TensorRole.OTHER, "X")
        self._registry.get_or_create(Y, TensorRole.OTHER, "Y")

        output = self.model(X)
        loss = loss_fn(output, Y)
        self._registry.get_or_create(loss, TensorRole.OTHER, "L")
        loss.backward()

        # register_full_backward_hook fires before param.grad is written for
        # the bottom-most leaf modules, so we collect weight gradients here.
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self._registry.mark_gradient(param, param.grad, name)

        self._opt.step()

        self._graph.freeze()
        compute_lifetimes(self._graph, self._registry)
        self._hooks.detach()
        self._ran = True

    def get_report(self) -> ProfileReport:
        if not self._ran:
            raise RuntimeError("Call run() before get_report().")
        return ProfileReport(self._graph, self._registry)

    def visualize(self, output_path: str = "memory_breakdown.png", mode: str = "static") -> None:
        """Generate a peak memory breakdown chart. mode: 'static' or 'dynamic'."""
        from .visualizer import MemoryVisualizer
        if not self._ran:
            raise RuntimeError("Call run() before visualize().")
        viz = MemoryVisualizer(self._graph, self._registry)
        viz.plot_memory_timeline(output_path=output_path, mode=mode)
        print(f"Saved memory breakdown chart to: {output_path}")
