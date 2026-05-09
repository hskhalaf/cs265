"""
Phase 3 entry point — the deliverable harness.

End-to-end flow per (model, batch_size):
    1. Trace the training step into an FX GraphModule.
    2. Profile the original graph (Phase 1) → no-AC peak and latency.
    3. Run the µ-TWO greedy (Phase 2) → which activations to recompute.
    4. Rewrite the graph in place (Phase 3) → splice recomputation chains
       into the backward pass.
    5. Re-profile the rewritten graph → measured AC peak and latency.
    6. Save the per-(model, bs) breakdown chart twice (no AC, with AC) and,
       at the end of the sweep, save the per-model grouped bar charts:
         peak_vs_batch (no AC vs AC)
         latency_vs_batch (no AC vs AC)

Everything in steps 2-5 is measured, not estimated.

Usage:
    python phase3.py                                  # full sweep
    python phase3.py resnet18 -b 8 16 32              # one model, three bs
    python phase3.py resnet50 -b 8 --mem-limit-mb 600 # explicit AC target
"""

from __future__ import annotations

import argparse
import os

import torch

from activation_checkpoint import (
    print_ac_decisions,
    rewrite_with_checkpointing,
    select_activations,
)
from graph_prof   import GraphProfiler
from graph_tracer import compile
from models       import MODELS, init_optimizer_state
from visualizer   import (
    plot_latency_comparison_vs_batch,
    plot_memory_breakdown,
    plot_peak_memory_vs_batch,
)


PLOTS_DIR = "plots"
DEFAULT_BATCH_SIZES = [4, 8, 16, 32]
WARMUP_ITERS  = 2
MEASURE_ITERS = 3


def profile_gm(gm, args) -> GraphProfiler:
    """Build a fresh profiler on `gm`, warm up, measure, aggregate."""
    profiler = GraphProfiler(gm)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            profiler.run(*args)
        profiler.reset_stats()
        for _ in range(MEASURE_ITERS):
            profiler.run(*args)
    profiler.aggregate_stats()
    return profiler


def profile_phase3(name: str, batch_size: int,
                   mem_limit_mb: float | None = None) -> dict:
    """Run the full Phase 1 + 2 + 3 + measure flow for one (model, bs)."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: {name}  (batch_size={batch_size})")
    print(f"{'=' * 70}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    limit_bytes = None if mem_limit_mb is None else int(mem_limit_mb * 1024 ** 2)
    metrics: dict = {"model": name, "batch_size": batch_size}

    def transform(gm, args):
        # Phase 1: profile the original (no AC).
        prof_no_ac = profile_gm(gm, args)
        print("\n  -- no AC --")
        prof_no_ac.print_summary()

        # Phase 2: select activations to recompute.
        selection = select_activations(prof_no_ac, mem_limit=limit_bytes)
        print_ac_decisions(prof_no_ac, selection)

        # Phase 3: rewrite the graph in place, then re-profile.
        gm_ac = rewrite_with_checkpointing(gm, prof_no_ac, selection.to_recompute)
        prof_ac = profile_gm(gm_ac, args)
        print("\n  -- with AC --")
        prof_ac.print_summary()

        metrics.update({
            "n_recompute":      len(selection.to_recompute),
            "n_retain":         len(selection.to_retain),
            "peak_no_ac_mb":    prof_no_ac.peak_memory_bytes() / 1024**2,
            "peak_ac_mb":       prof_ac.peak_memory_bytes()    / 1024**2,
            "latency_no_ac_ms": prof_no_ac.iteration_latency_ms(),
            "latency_ac_ms":    prof_ac.iteration_latency_ms(),
            "selection_reason": selection.reason,
        })

        # Per-(model, bs) breakdown plots — same chart, twice.
        os.makedirs(PLOTS_DIR, exist_ok=True)
        base = f"{name}_bs{batch_size}"
        plot_memory_breakdown(
            prof_no_ac,
            f"{PLOTS_DIR}/phase3_memory_{base}_no_ac.png",
            title=f"{name} — Memory Breakdown, no AC (bs={batch_size})",
        )
        plot_memory_breakdown(
            prof_ac,
            f"{PLOTS_DIR}/phase3_memory_{base}_ac.png",
            title=f"{name} — Memory Breakdown, with AC (bs={batch_size})",
        )
        return gm_ac

    compile(train_step, transform)(model, optim, inputs)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*",
                   help=f"models to run (any of: {list(MODELS)}). "
                        f"Default: all of them.")
    p.add_argument("-b", "--batch-sizes", type=int, nargs="+",
                   default=DEFAULT_BATCH_SIZES,
                   help=f"batch sizes to sweep (default: {DEFAULT_BATCH_SIZES}).")
    p.add_argument("--mem-limit-mb", type=float, default=None,
                   help="absolute target peak for the AC selector, in MB. "
                        "Default: cut peak by half the activation memory.")
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
                all_metrics[name].append(
                    profile_phase3(name, bs, mem_limit_mb=args.mem_limit_mb))
            except torch.cuda.OutOfMemoryError as exc:
                print(f"  [OOM] {name} bs={bs}: {exc}")
                torch.cuda.empty_cache()

    # Per-model grouped bar charts.
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for name, rows in all_metrics.items():
        if len(rows) < 2:
            continue
        peak_rows = [
            {"batch_size": r["batch_size"],
             "ac_off":     r["peak_no_ac_mb"],
             "ac_on":      r["peak_ac_mb"]}
            for r in rows
        ]
        latency_rows = [
            {"batch_size": r["batch_size"],
             "ac_off":     r["latency_no_ac_ms"],
             "ac_on":      r["latency_ac_ms"]}
            for r in rows
        ]
        plot_peak_memory_vs_batch(
            peak_rows,
            f"{PLOTS_DIR}/phase3_peak_vs_batch_{name}.png",
            f"{name} — peak memory vs batch size",
        )
        plot_latency_comparison_vs_batch(
            latency_rows,
            f"{PLOTS_DIR}/phase3_latency_vs_batch_{name}.png",
            f"{name} — latency vs batch size",
        )

    # Consolidated table.
    if any(all_metrics.values()):
        print(f"\n{'=' * 86}\nPHASE 3 SWEEP SUMMARY\n{'=' * 86}")
        print(f"{'Model':<10} {'BS':>4} {'Recomp':>7} {'Retain':>7}"
              f" {'Peak off':>10} {'Peak AC':>10}"
              f" {'Lat off':>10} {'Lat AC':>10}")
        print("-" * 86)
        for name, rows in all_metrics.items():
            for r in rows:
                print(f"{name:<10} {r['batch_size']:>4}"
                      f" {r['n_recompute']:>7} {r['n_retain']:>7}"
                      f" {r['peak_no_ac_mb']:>7.1f} MB"
                      f" {r['peak_ac_mb']:>7.1f} MB"
                      f" {r['latency_no_ac_ms']:>7.1f} ms"
                      f" {r['latency_ac_ms']:>7.1f} ms")
        print("=" * 86)

    print(f"\nDone.  Plots in ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
