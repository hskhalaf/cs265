"""
Phase 1 entry point.

For each (model, batch_size) pair: trace the training step, profile it with
``GraphProfiler``, save a stacked-area memory-breakdown plot.  After the
sweep, save one peak-memory-vs-batch-size bar chart per model.

Usage
-----
    python starter_code.py                      # all four models, default batch sizes
    python starter_code.py dummy                # one model, default batch sizes
    python starter_code.py resnet50 -b 8        # one model, one batch size
    python starter_code.py dummy resnet18 -b 4 8 16 32

Outputs
-------
    plots/memory_<model>_bs<N>.png      stacked-area memory breakdown
    plots/peak_vs_batch_<model>.png     peak memory vs batch size (per model)
    stdout                              per-run summary + consolidated table
"""

from __future__ import annotations

import argparse
import os

import torch

from models       import MODELS, init_optimizer_state
from graph_tracer import compile
from graph_prof   import GraphProfiler
from visualizer   import plot_memory_breakdown, plot_peak_vs_batch


PLOTS_DIR = "plots"
LOGS_DIR = "logs"
DEFAULT_BATCH_SIZES = [4, 8, 16, 32]


def profile_model(name: str, batch_size: int) -> dict:
    """Profile one (model, batch_size).  Saves the breakdown plot and returns
    the headline numbers (peak memory, iteration latency, node counts) so the
    caller can aggregate them across batch sizes."""
    print(f"\n{'=' * 70}\nPHASE 1: {name}  (batch_size={batch_size})\n{'=' * 70}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    metrics: dict = {"model": name, "batch_size": batch_size}

    def transform(gm, args):
        # Two warm-up runs (kernel selection, allocator pre-warming),
        # then three measurement runs for stable timing.
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(3):
                profiler.run(*args)
        profiler.aggregate_stats()

        metrics.update({
            "peak_mb":         profiler.peak_memory_bytes() / 1024**2,
            "latency_ms":      profiler.iteration_latency_ms(),
            "n_intermediates": len(profiler.intermediates),
            "n_nodes":         len(profiler.nodes),
        })

        profiler.print_summary()

        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = f"{LOGS_DIR}/{name}_bs{batch_size}.txt"
        json_path = f"{LOGS_DIR}/{name}_bs{batch_size}.json"
        profiler.write_full_log(log_path)
        profiler.write_json_log(json_path)
        print(f"  Full log:  {log_path}")
        print(f"  JSON:      {json_path}")

        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = f"{PLOTS_DIR}/memory_{name}_bs{batch_size}.png"
        plot_memory_breakdown(profiler, plot_path,
                              title=f"{name} — Memory Breakdown (bs={batch_size})")
        return gm

    compile(train_step, transform)(model, optim, inputs)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*",
                   help=f"models to profile (any of: {list(MODELS)}). "
                        f"Default: all of them.")
    p.add_argument("-b", "--batch-sizes", type=int, nargs="+",
                   default=DEFAULT_BATCH_SIZES,
                   help=f"batch sizes to sweep (default: {DEFAULT_BATCH_SIZES}).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; this script needs a GPU.")
        return

    chosen = args.models or list(MODELS.keys())
    for name in chosen:
        if name not in MODELS:
            print(f"unknown model '{name}'; choose from {list(MODELS)}")
            return

    all_metrics: dict[str, list[dict]] = {name: [] for name in chosen}
    for name in chosen:
        for bs in args.batch_sizes:
            try:
                all_metrics[name].append(profile_model(name, bs))
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {name} bs={bs}: {e}")
                torch.cuda.empty_cache()

    # Per-model peak-vs-batch bar chart (the second Phase 1 deliverable).
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for name, rows in all_metrics.items():
        if len(rows) < 2:
            continue
        plot_peak_vs_batch(
            rows,
            f"{PLOTS_DIR}/peak_vs_batch_{name}.png",
            f"{name} — peak memory vs batch size (no AC)",
        )

    # Consolidated table to stdout.
    if any(all_metrics.values()):
        print(f"\n{'=' * 70}\nPHASE 1 SWEEP SUMMARY\n{'=' * 70}")
        print(f"{'Model':<10} {'BS':>4} {'Nodes':>6} {'Inter':>6}"
              f" {'Peak':>10} {'Latency':>10}")
        print("-" * 70)
        for name, rows in all_metrics.items():
            for r in rows:
                print(f"{name:<10} {r['batch_size']:>4} {r['n_nodes']:>6}"
                      f" {r['n_intermediates']:>6}"
                      f" {r['peak_mb']:>7.1f} MB"
                      f" {r['latency_ms']:>7.1f} ms")
        print("=" * 70)

    print(f"\nDone.  Plots in ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
