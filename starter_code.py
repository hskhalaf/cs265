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

# Default batch size per model.  Override with `-b N` on the command line.
DEFAULT_BATCH = {"dummy": 1000, "resnet18": 16, "resnet50": 16, "bert": 4}


def profile_model(name: str, batch_size: int, debug: bool = False) -> None:
    print(f"\n{'=' * 70}\nPHASE 1: {name}  (batch_size={batch_size})\n{'=' * 70}")

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    def transform(gm, args):
        profiler = GraphProfiler(gm)

        # Run a few iterations: warm up the caches, then take measurements.
        with torch.no_grad():
            for _ in range(2):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(3):
                profiler.run(*args)
        profiler.aggregate_stats()
        profiler.print_stats()

        # Compare static estimate to measured peak so we know how far off the
        # FX-visible accounting is from real GPU allocator behaviour.
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            gm(*args)
        measured = torch.cuda.max_memory_allocated()
        estimated = profiler.peak_memory_bytes()
        gap = (measured - estimated) / max(measured, 1) * 100
        print(f"\n  Static estimate : {estimated / 1024**2:>8.2f} MB")
        print(  f"  Measured peak   : {measured  / 1024**2:>8.2f} MB")
        print(  f"  Unaccounted gap : {gap:>8.1f} %"
                f"   (cuDNN/cuBLAS workspace + allocator overhead)")
        _print_top_tensors(profiler, k=10)
        if debug:
            _print_per_node_debug(profiler)

        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = f"{PLOTS_DIR}/memory_{name}_bs{batch_size}.png"
        plot_memory_breakdown(profiler, plot_path,
                              title=f"{name} — Memory Breakdown (bs={batch_size})")
        return gm

    compiled = compile(train_step, transform)
    compiled(model, optim, inputs)


def _print_per_node_debug(profiler: GraphProfiler) -> None:
    """Full per-node table: (name, region, role, bytes, target).

    Useful for spotting nodes that were sized at 0 by the alias-based check
    when they shouldn't have been, or vice versa.  Look for rows where role
    or size doesn't match what you'd expect from the target column.
    """
    print(f"\n  Per-node debug ({len(profiler.nodes)} nodes):")
    print(f"  {'#':<5} {'Name':<32} {'Region':<10} {'Role':<14}"
          f" {'Bytes':>10}  Target")
    print("  " + "-" * 110)
    for i, n in enumerate(profiler.nodes):
        size   = profiler.node_size_bytes.get(n, 0)
        role   = profiler.node_type.get(n).name
        region = profiler.region.get(n).name
        target = str(getattr(n, "target", n.op))[:55]
        print(f"  {i:<5} {n.name[:31]:<32} {region:<10} {role:<14}"
              f" {size:>10}  {target}")


def _print_top_tensors(profiler: GraphProfiler, k: int = 10) -> None:
    """Top-K largest tensors and their roles — quick sanity check on what
    dominates the static peak."""
    rows = sorted(
        profiler.node_size_bytes.items(), key=lambda kv: kv[1], reverse=True,
    )[:k]
    print(f"\n  Top {k} tensors by size:")
    print(f"  {'#':<3} {'Name':<35} {'Role':<18} {'Size(KB)':>10}")
    for i, (node, size) in enumerate(rows):
        if size == 0:
            continue
        role = profiler.node_type.get(node, NodeType.OTHER).name
        print(f"  {i:<3} {node.name[:34]:<35} {role:<18} {size / 1024:>10.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*",
                   help=f"models to profile (any of: {list(MODELS)}). "
                        f"Default: all of them.")
    p.add_argument("-b", "--batch-size", type=int, default=None,
                   help="batch size to use for every selected model "
                        "(overrides the per-model default).")
    p.add_argument("--debug", action="store_true",
                   help="dump full per-node table (name, region, role, "
                        "bytes, target) — useful for spotting accounting bugs.")
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
        profile_model(name, bs, debug=args.debug)

    print(f"\nDone.  Plots saved under ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
