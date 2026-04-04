"""
CS265 Activation Checkpointing — Main Entry Point.

This script demonstrates the full pipeline:
1. Define a model and training step.
2. Compile the training step into an FX graph via graph_tracer.compile().
3. Profile the graph (Phase 1): per-node timing, memory, tensor classification,
   activation lifetimes.
4. Visualise the memory breakdown as a stacked bar chart.
5. Run the μ-TWO greedy selection algorithm (Phase 2): decide which activations
   to recompute vs. retain.
"""

import logging
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from visualizer import MemoryVisualizer
from activation_checkpoint import (
    select_activations_to_recompute,
    print_ac_decisions,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DummyModel(nn.Module):
    """Simple feedforward MLP used for development and testing.

    Architecture: ``layers`` repetitions of (Linear → ReLU).
    All layers share the same hidden dimension ``dim``.
    """

    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor,
):
    """Standard training iteration: forward → loss → SEP → backward → step.

    The loss is wrapped with ``SEPFunction.apply()`` so that the compiled FX
    graph contains explicit separator nodes marking the forward/backward
    boundary.
    """
    loss = model(batch).sum()
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()


# ---------------------------------------------------------------------------
# Graph transformation callback
# ---------------------------------------------------------------------------


def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
    """User-defined callback invoked by ``compile()`` on the first iteration.

    This function receives the traced FX graph and flattened input arguments.
    It performs:
    1. Graph profiling (warm-up + measurement iterations).
    2. Memory breakdown visualisation.
    3. μ-TWO activation checkpointing selection.

    Parameters
    ----------
    gm : fx.GraphModule
        The traced graph containing forward + backward + optimizer.
    args : tuple
        Flattened input tensors to execute the graph.

    Returns
    -------
    fx.GraphModule
        The (potentially modified) graph.
    """
    # --- Phase 1: Profiling ------------------------------------------------
    profiler = GraphProfiler(gm)
    warm_up_iters, profile_iters = 2, 3

    with torch.no_grad():
        for _ in range(warm_up_iters):
            profiler.run(*args)
        profiler.reset_stats()
        for _ in range(profile_iters):
            profiler.run(*args)

    profiler.aggregate_stats()
    profiler.print_stats()

    # --- Memory breakdown chart --------------------------------------------
    viz = MemoryVisualizer(profiler)
    viz.plot_memory_timeline("memory_breakdown.png")

    # --- Phase 2: μ-TWO selection ------------------------------------------
    to_recompute, to_retain = select_activations_to_recompute(profiler)
    print_ac_decisions(profiler, to_recompute, to_retain)

    # Phase 3 (graph rewriting) will be added here later.

    return gm


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def experiment():
    """Run the full profiling + AC pipeline on DummyModel."""
    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)

    batch_size = 1000
    layers = 10
    dim = 100
    num_iters = 5
    device_str = "cuda:0"

    model = DummyModel(dim=dim, layers=layers).to(device_str)
    batch = torch.randn(batch_size, dim).to(device_str)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        foreach=True,
        capturable=True,
    )

    # Initialise optimizer state (Adam's lazy init) so that it is present
    # when the graph is traced.
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param, device=device_str)
    optim.step()
    optim.zero_grad()

    # Compile and run.
    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optim, batch)

    # Subsequent iterations use the cached compiled graph.
    for _ in range(num_iters - 1):
        compiled_fn(model, optim, batch)

    print("Experiment completed successfully.")


if __name__ == "__main__":
    experiment()
