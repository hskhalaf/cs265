import logging
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from visualizer import MemoryVisualizer
from activation_checkpoint import select_activations_to_recompute, print_ac_decisions


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)

def train_step(model: nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor,):
    loss = model(batch).sum()
    loss = SEPFunction.apply(loss) # so that the compiled FX graph contains explicit separator nodes
    loss.backward()
    optim.step()
    optim.zero_grad()


def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
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
    viz = MemoryVisualizer(profiler)
    viz.plot_memory_timeline("memory_breakdown.png")

    to_recompute, to_retain = select_activations_to_recompute(profiler)
    print_ac_decisions(profiler, to_recompute, to_retain)

    # !!! Phase 3 (graph rewriting) will be added here later. !!!
    return gm

def experiment():
    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)

    batch_size = 1000
    layers = 10
    dim = 100
    num_iters = 5
    device_str = "cuda:0"

    model = DummyModel(dim=dim, layers=layers).to(device_str)
    batch = torch.randn(batch_size, dim).to(device_str)

    # foreach = True =>  Instead of updating parameters one at a time, batch all parameters into lists and process them with _foreach_* operations
    # The downside is that the optimizer step becomes hundreds of small FX nodes instead of one _fused_adam node.
    # capturable=True => Makes Adam's internals traceable by FX
    optim = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True, capturable=True)

    # Initialize optimizer state so that it is present when the graph is traced.
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param, device=device_str)
    optim.step()
    optim.zero_grad()

    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optim, batch)

    for _ in range(num_iters - 1):
        compiled_fn(model, optim, batch)

    print("Experiment completed successfully.")
    
if __name__ == "__main__":
    experiment()
