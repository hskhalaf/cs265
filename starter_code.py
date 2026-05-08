"""
CS265 — read-this-first entry point.

Runs the full pipeline on the small DummyModel:

    trace -> profile -> visualise -> select -> rewrite -> re-profile

The whole flow fits on one screen so you can see how the pieces connect.
"""

import torch

from models             import make_dummy, init_optimizer_state
from graph_tracer       import compile
from graph_prof         import GraphProfiler
from visualizer         import plot_memory_breakdown
from activation_checkpoint import (
    select_activations, rewrite_with_checkpointing, print_ac_decisions,
)


def graph_transformation(gm, args):
    """Callback invoked once on the freshly-traced GraphModule.

    1. Run a few warm-up + measurement iterations through the profiler.
    2. Print stats and save the memory breakdown chart.
    3. Run the µ-TWO selection to decide which activations to recompute.
    4. Rewrite the graph to recompute them.  Return the rewritten graph.
    """
    profiler = GraphProfiler(gm)
    with torch.no_grad():
        for _ in range(2):  # warm-up
            profiler.run(*args)
        profiler.reset_stats()
        for _ in range(3):  # measurement
            profiler.run(*args)
    profiler.aggregate_stats()
    profiler.print_stats()

    plot_memory_breakdown(profiler, "memory_breakdown.png",
                          title="DummyModel — Memory Breakdown")

    selection = select_activations(profiler)
    print_ac_decisions(profiler, selection)

    return rewrite_with_checkpointing(gm, profiler, selection.to_recompute)


def main():
    torch.manual_seed(20)
    model, optim, example_inputs, train_step = make_dummy(
        batch_size=1000, layers=10, dim=100,
    )
    init_optimizer_state(model, optim)

    compiled = compile(train_step, graph_transformation)
    compiled(model, optim, example_inputs)            # first call traces + transforms
    for _ in range(4):
        compiled(model, optim, example_inputs)        # subsequent calls run rewritten gm

    print("starter_code: done.")


if __name__ == "__main__":
    main()
