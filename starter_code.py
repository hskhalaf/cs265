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
from visualizer     import plot_memory_breakdown, plot_peak_vs_batch


PLOTS_DIR     = "plots"
LOGS_DIR      = "logs"
SNAPSHOTS_DIR = "snapshots"

# Batch sizes to sweep per model.  Override with `-b 8 16 32` on the CLI.
DEFAULT_BATCH_SIZES = [4, 8, 16, 32]


def profile_model(name: str, batch_size: int,
                  snapshot: bool = False) -> dict:
    """Run Phase 1 on one (model, batch_size).  Returns a dict of
    headline metrics that the sweep aggregator collects across runs."""
    print(f"\n{'=' * 70}\nPHASE 1: {name}  (batch_size={batch_size})\n{'=' * 70}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    # Captured by the closure below and read after compile() returns.
    metrics: dict = {"model": name, "batch_size": batch_size}

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
        estimated = profiler.peak_memory_bytes()        # FX-visible tensors only
        profiled  = profiler.peak_memory_bytes(
            include_runtime_residual=True,
        )                                               # node peaks + FX roles
        residual  = profiled - estimated                # runtime-only CUDA memory
        gm_delta  = allocated - profiled                # GraphModule vs interpreter
        padding   = reserved  - allocated               # allocator block padding

        metrics.update({
            "estimated_mb": estimated / 1024**2,
            "profiled_mb":  profiled  / 1024**2,
            "allocated_mb": allocated / 1024**2,
            "reserved_mb":  reserved  / 1024**2,
            "latency_ms":   profiler.iteration_latency_ms(),
            "n_intermediates": len(profiler.intermediates),
            "n_nodes":      len(profiler.nodes),
        })

        # ---- console: concise summary only ----
        profiler.print_summary(include_runtime_residual=True)
        print(f"  Static FX-visible tensors:   {estimated / 1024**2:>9.2f} MB")
        print(f"  Profiled node peak:          {profiled  / 1024**2:>9.2f} MB"
              f"   ({residual / 1024**2:+.1f} MB runtime residual folded into OTHER)")
        print(f"  Allocated (compiled graph):  {allocated / 1024**2:>9.2f} MB"
              f"   ({gm_delta / 1024**2:+.1f} MB vs profiler run)")
        print(f"  Reserved  (allocator pool):  {reserved  / 1024**2:>9.2f} MB"
              f"   ({padding   / 1024**2:+.1f} MB  =  caching-allocator block padding)")
        _print_top_tensors(profiler, k=5)

        # ---- file: full verbose log + machine-readable dump ----
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path  = f"{LOGS_DIR}/{name}_bs{batch_size}.txt"
        json_path = f"{LOGS_DIR}/{name}_bs{batch_size}.json"
        profiler.write_full_log(log_path)
        profiler.write_json_log(json_path)
        print(f"\n  Full log:  {log_path}")
        print(f"  JSON:      {json_path}")

        # ---- plot ----
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = f"{PLOTS_DIR}/memory_{name}_bs{batch_size}.png"
        plot_memory_breakdown(profiler, plot_path,
                              title=f"{name} — Memory Breakdown (bs={batch_size})")
        print(f"  Plot:      {plot_path}")
        return gm

    compiled = compile(train_step, transform)
    compiled(model, optim, inputs)
    return metrics


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

    # all_metrics[model] = list of per-batch metrics dicts
    all_metrics: dict[str, list[dict]] = {}

    for name in chosen:
        if name not in MODELS:
            print(f"unknown model '{name}'; choose from {list(MODELS)}")
            continue
        all_metrics[name] = []
        for bs in args.batch_sizes:
            try:
                m = profile_model(name, bs, snapshot=args.snapshot)
                all_metrics[name].append(m)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {name} bs={bs}: {e}")
                torch.cuda.empty_cache()

    # ---- Phase 1 deliverable: peak vs batch size, per model ----
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for name, rows in all_metrics.items():
        if len(rows) < 2:    # only one batch size — no curve to plot
            continue
        bar_rows = [{"batch_size": r["batch_size"],
                     "peak_mb":    r["allocated_mb"]} for r in rows]
        plot_peak_vs_batch(
            bar_rows,
            f"{PLOTS_DIR}/peak_vs_batch_{name}.png",
            f"{name} — peak memory vs batch size (no AC)",
        )

    # ---- Consolidated summary table ----
    if any(all_metrics.values()):
        print(f"\n{'=' * 92}")
        print("PHASE 1 SWEEP SUMMARY")
        print("=" * 92)
        print(f"{'Model':<10} {'BS':>4} {'Nodes':>6} {'Inter':>6}"
              f" {'FX':>10} {'Profiled':>10} {'Allocated':>10}"
              f" {'Reserved':>10} {'Latency':>10}")
        print("-" * 92)
        for name, rows in all_metrics.items():
            for r in rows:
                print(f"{name:<10} {r['batch_size']:>4} {r['n_nodes']:>6}"
                      f" {r['n_intermediates']:>6}"
                      f" {r['estimated_mb']:>8.1f} MB"
                      f" {r['profiled_mb']:>8.1f} MB"
                      f" {r['allocated_mb']:>8.1f} MB"
                      f" {r['reserved_mb']:>8.1f} MB"
                      f" {r['latency_ms']:>8.1f} ms")
        print("=" * 92)

        # Same table to disk for the report.
        os.makedirs(LOGS_DIR, exist_ok=True)
        summary_path = f"{LOGS_DIR}/phase1_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"{'Model':<10} {'BS':>4} {'Nodes':>6} {'Inter':>6}"
                    f" {'FX':>10} {'Profiled':>10} {'Allocated':>10}"
                    f" {'Reserved':>10} {'Latency':>10}\n")
            for name, rows in all_metrics.items():
                for r in rows:
                    f.write(f"{name:<10} {r['batch_size']:>4}"
                            f" {r['n_nodes']:>6} {r['n_intermediates']:>6}"
                            f" {r['estimated_mb']:>8.1f} MB"
                            f" {r['profiled_mb']:>8.1f} MB"
                            f" {r['allocated_mb']:>8.1f} MB"
                            f" {r['reserved_mb']:>8.1f} MB"
                            f" {r['latency_ms']:>8.1f} ms\n")
        print(f"Summary table saved to {summary_path}")

    print(f"\nDone.  Plots in ./{PLOTS_DIR}/   logs in ./{LOGS_DIR}/")


if __name__ == "__main__":
    main()
