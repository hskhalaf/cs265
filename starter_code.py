"""
usage:
python starter_code.py                  
python starter_code.py resnet50 -b 8        
"""

from __future__ import annotations
import argparse
import os
import torch
from models import MODELS, init_optimizer_state
from graph_tracer import compile
from graph_prof import GraphProfiler
from visualizer import (plot_latency_vs_batch, plot_memory_breakdown, plot_peak_vs_batch)

PLOTS_DIR = "plots"
DEFAULT_BATCH_SIZES = [4, 8, 16, 32]


def profile_model(name: str, batch_size: int) -> dict:
    print(f"\n{'=' * 70}\nPHASE 1: {name}  (batch_size={batch_size})\n{'=' * 70}")
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)
    metrics: dict = {"model": name, "batch_size": batch_size}

    def transform(gm, args):
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(3):
                profiler.run(*args)
        profiler.aggregate_stats()
        metrics.update({
            "peak_mb": profiler.peak_memory_bytes() / 1024**2,
            "latency_ms": profiler.iteration_latency_ms(),
            "n_intermediates": len(profiler.intermediates),
            "n_nodes": len(profiler.nodes),
        })

        profiler.print_summary()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = f"{PLOTS_DIR}/memory_{name}_bs{batch_size}.png"
        plot_memory_breakdown(profiler, plot_path, title=f"{name} memory breakdown (batch size = {batch_size})")
        return gm

    compile(train_step, transform)(model, optim, inputs)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*")
    p.add_argument("-b", "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,  help=f"batch sizes to sweep (default: {DEFAULT_BATCH_SIZES}).")
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

    os.makedirs(PLOTS_DIR, exist_ok=True)
    for name, rows in all_metrics.items():
        if len(rows) < 2:
            continue
        plot_peak_vs_batch(rows, f"{PLOTS_DIR}/peak_vs_batch_{name}.png", f"{name} — peak memory vs batch size (no AC)",)
        plot_latency_vs_batch(rows, f"{PLOTS_DIR}/latency_vs_batch_{name}.png",f"{name} — latency vs batch size (no AC)",)

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
