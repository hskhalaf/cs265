"""
Diagnose where the static-vs-measured memory gap comes from.

For one (model, batch_size), compares at every step:
  static[t]    = sum of node.size_bytes for nodes alive at step t (FX walk)
  measured[t]  = torch.cuda.memory_allocated() right after step t (real GPU)

Reports:
  1. Three peak numbers: static / interpreter-measured / gm()-measured.
  2. Top-K steps by |measured - static| with the op running at that step.
  3. Static vs measured at region boundaries (fwd start, sep, sep_bwd, opt).

If the gap is concentrated at a few ops, the labels in (2) tell us which
ops we're misclassifying or under-counting.  If the gap is roughly constant
across all steps, it's something pervasive (workspaces, allocator overhead).
If the gap appears only after the sep_bwd boundary, the bug is in our
backward-region accounting.

Usage:
    python diagnose.py dummy -b 8
    python diagnose.py resnet18 -b 8
    python diagnose.py dummy resnet18 resnet50 -b 8
"""

from __future__ import annotations

import argparse

import torch

from models       import MODELS, init_optimizer_state
from graph_tracer import compile
from graph_prof   import GraphProfiler, NodeType


def diagnose(name: str, batch_size: int, top_k: int = 10) -> None:
    print(f"\n{'=' * 90}")
    print(f"DIAGNOSTIC: {name}  bs={batch_size}")
    print('=' * 90)
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    model, optim, inputs, train_step = MODELS[name](batch_size=batch_size)
    init_optimizer_state(model, optim)

    def transform(gm, args):
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(3):
                profiler.run(*args)
        profiler.aggregate_stats()

        n        = len(profiler.nodes)
        timeline = profiler.memory_timeline_by_role()
        static   = [sum(timeline[r][t] for r in NodeType) for t in range(n)]
        measured = profiler.avg_measured_memory[:n]

        # Independent peak from a clean gm() call.
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            gm(*args)
        gm_peak = torch.cuda.max_memory_allocated()

        st_pk    = max(static)
        st_pk_t  = static.index(st_pk)
        mt_pk    = max(measured) if measured else 0
        mt_pk_t  = measured.index(mt_pk) if measured else -1

        print("\n  Peaks:")
        print(f"    Static  (FX walk):       {st_pk / 1024**2:>9.2f} MB"
              f"   @ step {st_pk_t:>4}  ({profiler.nodes[st_pk_t].name})")
        print(f"    Interp-measured:         {mt_pk / 1024**2:>9.2f} MB"
              f"   @ step {mt_pk_t:>4}  "
              f"({profiler.nodes[mt_pk_t].name if mt_pk_t >= 0 else '-'})")
        print(f"    gm() max_allocated:      {gm_peak / 1024**2:>9.2f} MB")
        gap_int = (mt_pk - st_pk) / max(mt_pk, 1) * 100
        gap_gm  = (gm_peak - st_pk) / max(gm_peak, 1) * 100
        print(f"    Gap (interp - static):  {(mt_pk - st_pk) / 1024**2:>+8.2f} MB"
              f"   ({gap_int:+.1f} %)")
        print(f"    Gap (gm()   - static):  {(gm_peak - st_pk) / 1024**2:>+8.2f} MB"
              f"   ({gap_gm:+.1f} %)")

        if not measured or len(measured) != n:
            print(f"\n  [WARN] measured length ({len(measured)}) != nodes ({n}); "
                  f"per-step diff skipped.")
            return gm

        # ---- Top discrepancies ----
        diffs = sorted(
            ((t, measured[t] - static[t]) for t in range(n)),
            key=lambda x: abs(x[1]), reverse=True,
        )
        print(f"\n  Top {top_k} steps by |measured - static|:")
        print(f"    {'Step':>5} {'Node':<26} {'Op':<34}"
              f" {'Reg':<5} {'Role':<5}"
              f" {'Static':>9} {'Meas':>9} {'Diff':>9}")
        print("    " + "-" * 122)
        for t, _ in diffs[:top_k]:
            node   = profiler.nodes[t]
            target = str(getattr(node, "target", node.op))[:33]
            print(f"    {t:>5} {node.name[:25]:<26} {target:<34}"
                  f" {profiler.region[node].value[:4]:<5}"
                  f" {profiler.node_type[node].name[:4]:<5}"
                  f" {static[t]   / 1024**2:>6.1f} MB"
                  f" {measured[t] / 1024**2:>6.1f} MB"
                  f" {(measured[t] - static[t]) / 1024**2:>+6.1f} MB")

        # ---- Boundaries ----
        print("\n  At region boundaries:")
        print(f"    {'Where':<24} {'Step':>5}"
              f" {'Static':>10} {'Meas':>10} {'Diff':>10}")
        print("    " + "-" * 64)
        for label, t in [
            ("start of forward",       0),
            ("end of forward (sep)",   profiler.sep_idx),
            ("start of backward",      profiler.sep_bwd_idx),
            ("start of optimizer",     profiler.opt_idx),
            ("last step",              n - 1),
        ]:
            if 0 <= t < n:
                print(f"    {label:<24} {t:>5}"
                      f" {static[t]   / 1024**2:>7.2f} MB"
                      f" {measured[t] / 1024**2:>7.2f} MB"
                      f" {(measured[t] - static[t]) / 1024**2:>+7.2f} MB")

        return gm

    compile(train_step, transform)(model, optim, inputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+",
                   help=f"models to diagnose (any of: {list(MODELS)})")
    p.add_argument("-b", "--batch-size", type=int, default=8)
    p.add_argument("-k", "--top-k", type=int, default=10)
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("Need CUDA.")
        return
    for name in args.models:
        if name not in MODELS:
            print(f"unknown model: {name}")
            continue
        diagnose(name, args.batch_size, args.top_k)


if __name__ == "__main__":
    main()
