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


PLOTS_DIR     = "plots"
LOGS_DIR      = "logs"
SNAPSHOTS_DIR = "snapshots"

# Batch sizes to sweep per model.  Override with `-b 8 16 32` on the CLI.
DEFAULT_BATCH_SIZES = [4, 8, 32]


def profile_model(name: str, batch_size: int, snapshot: bool = False) -> None:
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

        # ---- three peak numbers, computed on a fresh measurement run ----
        torch.cuda.reset_peak_memory_stats()
        if snapshot:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        with torch.no_grad():
            gm(*args)
        if snapshot:
            os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
            snap_path = f"{SNAPSHOTS_DIR}/{name}_bs{batch_size}.pickle"
            torch.cuda.memory._dump_snapshot(snap_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"  Snapshot:  {snap_path}"
                  f"  (load at https://pytorch.org/memory_viz)")
        allocated = torch.cuda.max_memory_allocated()   # peak live tensor bytes
        reserved  = torch.cuda.max_memory_reserved()    # peak allocator-pool bytes
        estimated = profiler.peak_memory_bytes()        # our FX-only static walk
        invisible = allocated - estimated               # tensors we don't see (workspaces)
        padding   = reserved  - allocated               # allocator block padding

        # ---- console: concise summary only ----
        profiler.print_summary()
        print(f"  Estimate (FX nodes only):    {estimated / 1024**2:>9.2f} MB")
        print(f"  Allocated (real tensors):    {allocated / 1024**2:>9.2f} MB"
              f"   ({invisible / 1024**2:+.1f} MB  =  cuDNN/cuBLAS workspace + sim error)")
        print(f"  Reserved  (allocator pool):  {reserved  / 1024**2:>9.2f} MB"
              f"   ({padding   / 1024**2:+.1f} MB  =  caching-allocator block padding)")
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
    p.add_argument("-b", "--batch-sizes", type=int, nargs="+",
                   default=DEFAULT_BATCH_SIZES,
                   help=f"batch sizes to sweep for each model "
                        f"(default: {DEFAULT_BATCH_SIZES}).")
    p.add_argument("--snapshot", action="store_true",
                   help="record a CUDA memory snapshot per (model, batch) "
                        "and save to snapshots/<model>_bs<N>.pickle. "
                        "Load at https://pytorch.org/memory_viz to inspect "
                        "every allocation and its stack trace.")
    p.add_argument("--no-cudnn", action="store_true",
                   help="disable cuDNN entirely (cudnn.enabled=False).  "
                        "Forces convolutions through the native CUDA path "
                        "which allocates ~zero workspace.  Much slower, but "
                        "if the Estimate↔Allocated gap shrinks dramatically, "
                        "the gap was cuDNN workspace.  If it stays, the gap "
                        "is somewhere in our analysis.")
    args = p.parse_args()

    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN DISABLED — convolutions will run through native CUDA "
              "(slow, but no workspace).")

    if not torch.cuda.is_available():
        print("CUDA unavailable; this script needs a GPU.")
        return

    chosen = args.models or list(MODELS.keys())
    for name in chosen:
        if name not in MODELS:
            print(f"unknown model '{name}'; choose from {list(MODELS)}")
            continue
        for bs in args.batch_sizes:
            try:
                profile_model(name, bs, snapshot=args.snapshot)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {name} bs={bs}: {e}")
                torch.cuda.empty_cache()

    print(f"\nDone.  Plots in ./{PLOTS_DIR}/   logs in ./{LOGS_DIR}/")


if __name__ == "__main__":
    main()
