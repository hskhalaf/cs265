"""
Phase 2 entry point.

This script runs the Phase 1 profiler, applies the activation-checkpointing
selector, and plots the simulated memory/latency tradeoff.  It does not rewrite
the graph.  The "AC" numbers here are therefore estimates from the static
simulator:

    peak_AC      = simulated peak after evicting selected activations
    latency_AC   = measured latency without AC + estimated recomputation time

Use Phase 3 to turn these estimates into measured values.
"""

from __future__ import annotations

import argparse
import os

import torch

from activation_checkpoint import (
    select_activations,
    simulate_timeline_by_role,
)
from graph_prof import GraphProfiler
from graph_tracer import compile
from models import MODELS, init_optimizer_state
from visualizer import (
    plot_latency_comparison_vs_batch,
    plot_memory_breakdown,
    plot_peak_memory_vs_batch,
)


PLOTS_DIR = "plots"
DEFAULT_BATCH_SIZES = [4, 8, 16, 32]
WARMUP_ITERS = 2
MEASURE_ITERS = 3


def _mb(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


def profile_phase2(name: str,
                   batch_size: int,
                   mem_limit_mb: float | None = None) -> dict:
    """Profile one model/batch pair and run the Phase 2 AC selector."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: {name}  (batch_size={batch_size})")
    print(f"{'=' * 70}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    limit_bytes = None if mem_limit_mb is None else int(mem_limit_mb * 1024 ** 2)
    metrics: dict = {"model": name, "batch_size": batch_size}

    def transform(gm, args):
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(MEASURE_ITERS):
                profiler.run(*args)
        profiler.aggregate_stats()

        selection = select_activations(profiler, mem_limit=limit_bytes)
        no_ac_ms = profiler.iteration_latency_ms()
        ac_est_ms = no_ac_ms + selection.estimated_recompute_ms

        metrics.update({
            "n_nodes": len(profiler.nodes),
            "n_intermediates": len(profiler.intermediates),
            "n_recompute": len(selection.to_recompute),
            "n_retain": len(selection.to_retain),
            "peak_no_ac_mb": _mb(selection.peak_before),
            "peak_ac_est_mb": _mb(selection.peak_after),
            "latency_no_ac_ms": no_ac_ms,
            "latency_ac_est_ms": ac_est_ms,
            "recompute_ms": selection.estimated_recompute_ms,
            "reason": selection.reason,
        })

        profiler.print_summary()
        print(f"\n  Phase 2 selection:")
        print(f"    Recompute:      {len(selection.to_recompute)} activations")
        print(f"    Retain:         {len(selection.to_retain)} activations")
        print(f"    Peak no AC:     {_mb(selection.peak_before):7.2f} MB")
        print(f"    Peak AC est.:   {_mb(selection.peak_after):7.2f} MB")
        print(f"    Latency no AC:  {no_ac_ms:7.2f} ms")
        print(f"    Latency AC est.: {ac_est_ms:6.2f} ms")
        print(f"    Stop reason:    {selection.reason}")

        os.makedirs(PLOTS_DIR, exist_ok=True)
        base = f"{name}_bs{batch_size}"
        plot_memory_breakdown(
            profiler,
            f"{PLOTS_DIR}/phase2_memory_{base}_no_ac.png",
            title=f"{name} — Memory Breakdown, no AC (bs={batch_size})",
        )
        plot_memory_breakdown(
            profiler,
            f"{PLOTS_DIR}/phase2_memory_{base}_ac_est.png",
            title=f"{name} — Simulated Memory Breakdown, AC (bs={batch_size})",
            timeline_by_role=simulate_timeline_by_role(
                profiler, selection.to_recompute,
            ),
        )
        return gm

    compile(train_step, transform)(model, optim, inputs)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*",
                        help=f"models to profile (any of: {list(MODELS)}). "
                             f"Default: all of them.")
    parser.add_argument("-b", "--batch-sizes", type=int, nargs="+",
                        default=DEFAULT_BATCH_SIZES,
                        help=f"batch sizes to sweep (default: {DEFAULT_BATCH_SIZES}).")
    parser.add_argument("--mem-limit-mb", type=float, default=None,
                        help="absolute target peak for the AC selector, in MB. "
                             "Default: reduce peak by half of total activation bytes.")
    args = parser.parse_args()

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
                row = profile_phase2(name, bs, mem_limit_mb=args.mem_limit_mb)
                all_metrics[name].append(row)
            except torch.cuda.OutOfMemoryError as exc:
                print(f"  [OOM] {name} bs={bs}: {exc}")
                torch.cuda.empty_cache()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    for name, rows in all_metrics.items():
        if len(rows) < 2:
            continue
        peak_rows = [
            {"batch_size": r["batch_size"],
             "ac_off": r["peak_no_ac_mb"],
             "ac_on": r["peak_ac_est_mb"]}
            for r in rows
        ]
        latency_rows = [
            {"batch_size": r["batch_size"],
             "ac_off": r["latency_no_ac_ms"],
             "ac_on": r["latency_ac_est_ms"]}
            for r in rows
        ]
        plot_peak_memory_vs_batch(
            peak_rows,
            f"{PLOTS_DIR}/phase2_peak_vs_batch_{name}.png",
            f"{name} — peak memory vs batch size",
            ac_label="AC estimate",
        )
        plot_latency_comparison_vs_batch(
            latency_rows,
            f"{PLOTS_DIR}/phase2_latency_vs_batch_{name}.png",
            f"{name} — latency vs batch size",
            ac_label="AC estimate",
        )

    if any(all_metrics.values()):
        print(f"\n{'=' * 70}")
        print("PHASE 2 SWEEP SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Model':<10} {'BS':>4} {'Recomp':>7} {'Retain':>7}"
              f" {'Peak off':>10} {'Peak AC':>10}"
              f" {'Lat off':>10} {'Lat AC':>10}")
        print("-" * 70)
        for name, rows in all_metrics.items():
            for r in rows:
                print(f"{name:<10} {r['batch_size']:>4}"
                      f" {r['n_recompute']:>7} {r['n_retain']:>7}"
                      f" {r['peak_no_ac_mb']:>7.1f} MB"
                      f" {r['peak_ac_est_mb']:>7.1f} MB"
                      f" {r['latency_no_ac_ms']:>7.1f} ms"
                      f" {r['latency_ac_est_ms']:>7.1f} ms")
        print("=" * 70)

    print(f"\nDone.  Plots in ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
