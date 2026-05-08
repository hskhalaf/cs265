"""
Deliverable harness.

For each model in MODELS_TO_SWEEP and each batch size, measures peak memory
and iteration latency once with activation checkpointing off and once with
it on.  Saves two grouped-bar plots per model under ``plots/``:

    plots/peak_<model>.png      peak memory   vs batch size, AC off vs on
    plots/latency_<model>.png   iter latency  vs batch size, AC off vs on

Usage:
    python benchmarks.py                # full sweep
    python benchmarks.py resnet18       # one model only
    python benchmarks.py --quick        # smaller batch ranges (smoke test)
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import List, Tuple

import torch

from models                import MODELS, init_optimizer_state
from graph_tracer          import compile
from graph_prof            import GraphProfiler
from visualizer            import plot_peak_memory_vs_batch, plot_latency_vs_batch
from activation_checkpoint import select_activations, rewrite_with_checkpointing


MODELS_TO_SWEEP: List[str] = ["resnet18", "resnet50", "bert"]

BATCH_SIZES = {
    "dummy":    [256, 512, 1000, 2000],
    "resnet18": [8,   16,  32,   64],
    "resnet50": [4,   8,   16,   32],
    "bert":     [2,   4,   8,    16],
}

QUICK_BATCH_SIZES = {
    "dummy":    [256, 512],
    "resnet18": [8, 16],
    "resnet50": [4, 8],
    "bert":     [2, 4],
}

WARM_UP_ITERS    = 2
MEASURE_ITERS    = 5
PLOTS_DIR        = "plots"


# --------------------------------------------------------------------------- #
# Single (model, batch_size, ac) measurement                                  #
# --------------------------------------------------------------------------- #


def run_one(model_name: str, batch_size: int, ac: bool) -> Tuple[float, float]:
    """Build, compile, run, and return ``(peak_mb, latency_ms)``."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(0)

    model, optim, inputs, train_step = MODELS[model_name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    def transform(gm, args):
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(WARM_UP_ITERS):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(MEASURE_ITERS):
                profiler.run(*args)
        profiler.aggregate_stats()
        if ac:
            selection = select_activations(profiler)
            return rewrite_with_checkpointing(
                gm, profiler, selection.to_recompute,
            )
        return gm

    compiled = compile(train_step, transform)

    # First call traces + (optionally) rewrites the graph.
    compiled(model, optim, inputs)

    # Warm up the (possibly rewritten) graph before measuring.
    for _ in range(WARM_UP_ITERS):
        compiled(model, optim, inputs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        compiled(model, optim, inputs)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    peak_mb    = torch.cuda.max_memory_allocated() / (1024 ** 2)
    latency_ms = (elapsed_s / MEASURE_ITERS) * 1000.0

    # Free everything before the next configuration.
    del model, optim, inputs, train_step, compiled
    torch.cuda.empty_cache()
    gc.collect()
    return peak_mb, latency_ms


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #


def sweep_model(model_name: str, batch_sizes: List[int]) -> List[dict]:
    """Returns one dict per batch size with both AC-off and AC-on numbers."""
    rows: List[dict] = []
    for bs in batch_sizes:
        print(f"\n--- {model_name}  bs={bs} ---")
        try:
            off_mb, off_ms = run_one(model_name, bs, ac=False)
            print(f"  AC off : peak={off_mb:7.1f} MB  latency={off_ms:7.2f} ms")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  AC off : OOM ({e})")
            off_mb, off_ms = float("nan"), float("nan")

        try:
            on_mb, on_ms = run_one(model_name, bs, ac=True)
            print(f"  AC on  : peak={on_mb:7.1f} MB  latency={on_ms:7.2f} ms")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  AC on  : OOM ({e})")
            on_mb, on_ms = float("nan"), float("nan")

        rows.append({
            "batch_size": bs,
            "ac_off_peak_mb":  off_mb,
            "ac_on_peak_mb":   on_mb,
            "ac_off_lat_ms":   off_ms,
            "ac_on_lat_ms":    on_ms,
        })
    return rows


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable; benchmarks need a GPU.")
        sys.exit(1)

    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--quick"]
    models = args or MODELS_TO_SWEEP
    batch_table = QUICK_BATCH_SIZES if quick else BATCH_SIZES

    os.makedirs(PLOTS_DIR, exist_ok=True)

    for model_name in models:
        bsizes = batch_table.get(model_name, [4, 8, 16])
        rows = sweep_model(model_name, bsizes)

        peak_rows = [{"batch_size": r["batch_size"],
                      "ac_off": r["ac_off_peak_mb"],
                      "ac_on":  r["ac_on_peak_mb"]} for r in rows]
        lat_rows  = [{"batch_size": r["batch_size"],
                      "ac_off": r["ac_off_lat_ms"],
                      "ac_on":  r["ac_on_lat_ms"]}  for r in rows]

        plot_peak_memory_vs_batch(
            peak_rows,
            f"{PLOTS_DIR}/peak_{model_name}.png",
            f"Peak Memory — {model_name}",
        )
        plot_latency_vs_batch(
            lat_rows,
            f"{PLOTS_DIR}/latency_{model_name}.png",
            f"Iteration Latency — {model_name}",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
