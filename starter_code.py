"""
Phase 1 entry point.

Trace -> profile -> print stats -> save memory-breakdown chart.
Nothing else (no AC selection, no graph rewriting).

Usage
-----
    python starter_code.py                      # all four models, default batch sizes
    python starter_code.py dummy                # one model
    python starter_code.py resnet50 -b 8        # custom batch size
    python starter_code.py dummy resnet18 -b 32 # several models, same batch size

Outputs
-------
    plots/memory_<model>.png   stacked-area memory breakdown (per role) over the iteration
    stdout                     per-node table, role totals, and a static-vs-measured peak summary
"""

from __future__ import annotations

import argparse
import os

import torch

from models         import MODELS, init_optimizer_state
from graph_tracer   import compile
from graph_prof     import GraphProfiler, NodeType
from visualizer     import plot_memory_breakdown


PLOTS_DIR = "plots"
LOGS_DIR  = "logs"

# Default batch size per model.  Override with `-b N` on the command line.
DEFAULT_BATCH = {"dummy": 1000, "resnet18": 16, "resnet50": 16, "bert": 4}


def profile_model(name: str, batch_size: int) -> None:
    print(f"\n{'=' * 70}\nPHASE 1: {name}  (batch_size={batch_size})\n{'=' * 70}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    def transform(gm, args):
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):                # warm-up
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(3):                # measurement
                profiler.run(*args)
        profiler.aggregate_stats()

        # ---- compute the static-vs-measured peak gap ----
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            gm(*args)
        measured  = torch.cuda.max_memory_allocated()
        estimated = profiler.peak_memory_bytes()
        gap = (measured - estimated) / max(measured, 1) * 100

        # ---- console: concise summary only ----
        profiler.print_summary()
        print()
        print(f"  Static estimate :  {estimated / 1024**2:>9.2f} MB")
        print(f"  Measured peak   :  {measured  / 1024**2:>9.2f} MB")
        print(f"  Unaccounted gap :  {gap:>9.1f} %  "
              f"(cuDNN/cuBLAS workspace + allocator)")
        _print_top_tensors(profiler, k=5)

        # ---- file: full verbose log ----
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = f"{LOGS_DIR}/{name}_bs{batch_size}.txt"
        profiler.write_full_log(log_path)
        print(f"\n  Full log:  {log_path}")

        # ---- plot ----
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = f"{PLOTS_DIR}/memory_{name}_bs{batch_size}.png"
        plot_memory_breakdown(profiler, plot_path,
                              title=f"{name} — Memory Breakdown (bs={batch_size})")
        print(f"  Plot:      {plot_path}")
        return gm

    compiled = compile(train_step, transform)
    compiled(model, optim, inputs)


def _print_top_tensors(profiler: GraphProfiler, k: int = 5) -> None:
    rows = sorted(
        profiler.node_size_bytes.items(), key=lambda kv: kv[1], reverse=True,
    )[:k]
    print(f"\n  Top {k} tensors:")
    for i, (node, size) in enumerate(rows):
        if size == 0:
            continue
        role = profiler.node_type.get(node, NodeType.OTHER).name
        print(f"    {i+1}.  {node.name[:30]:<30}  {role:<6}"
              f"  {size / 1024**2:>7.2f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*",
                   help=f"models to profile (any of: {list(MODELS)}). "
                        f"Default: all of them.")
    p.add_argument("-b", "--batch-size", type=int, default=None,
                   help="batch size to use for every selected model "
                        "(overrides the per-model default).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; this script needs a GPU.")
        return

    chosen = args.models or list(MODELS.keys())
    for name in chosen:
        if name not in MODELS:
            print(f"unknown model '{name}'; choose from {list(MODELS)}")
            continue
        bs = args.batch_size if args.batch_size is not None \
             else DEFAULT_BATCH[name]
        profile_model(name, bs)

    print(f"\nDone.  Plots saved under ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
