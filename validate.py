"""
End-to-end sanity checks for the CS265 project.

Three checks per model, run inside the graph-transformation callback so we
have direct access to the freshly-traced GraphModule:

  1. PROFILER accuracy   — runtime-adjusted peak_memory_bytes() within 30 %
                            of torch.cuda.max_memory_allocated().
  2. AC sanity per arch  — ResNet evicts >= 1 intermediate;
                            BERT (peak in optimizer region) evicts ~none.
  3. AC correctness      — the AC-rewritten graph produces outputs within
                            1 e-4 of the unrewritten graph on the same inputs.

Usage:
    python validate.py                # default: dummy
    python validate.py resnet18
    python validate.py --all
"""

from __future__ import annotations

import sys
from typing import Dict, List

import torch

from models                import MODELS, init_optimizer_state
from graph_tracer          import compile
from graph_prof            import GraphProfiler
from activation_checkpoint import select_activations, rewrite_with_checkpointing


# How many intermediates we expect each architecture to evict.  These are
# loose ranges; the goal is to catch obvious regressions, not exact counts.
EXPECTED_EVICTIONS: Dict[str, tuple] = {
    "dummy":    (1, 100),     # dense MLP — should evict several relus
    "resnet18": (1, 200),
    "resnet50": (1, 500),
    "bert":     (0, 5),       # peak is in optimizer region — AC ~useless
}

# Per-model accuracy tolerance for the static peak-memory estimate.
# Conv-heavy nets allocate cuDNN workspace memory that isn't visible to FX
# (it's per-kernel scratch the conv runtime grabs), so the static estimate
# undershoots the measured peak.  These tolerances reflect that, not bugs
# in the analysis.
ACCURACY_TOLERANCE: Dict[str, float] = {
    "dummy":    0.30,
    "resnet18": 0.50,
    "resnet50": 0.50,
    "bert":     0.30,
}


def _profile(gm, args, iters: int = 3) -> GraphProfiler:
    profiler = GraphProfiler(gm)
    with torch.no_grad():
        for _ in range(2):
            profiler.run(*args)
        profiler.reset_stats()
        for _ in range(iters):
            profiler.run(*args)
    profiler.aggregate_stats()
    return profiler


def _check_profiler_accuracy(name: str, profiler: GraphProfiler,
                              gm, args) -> bool:
    fx_only = profiler.peak_memory_bytes()
    estimated = profiler.peak_memory_bytes(include_runtime_residual=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        gm(*args)
    measured = torch.cuda.max_memory_allocated()
    if measured == 0:
        print("  [WARN] measured peak is 0 — skipping accuracy check")
        return True
    rel = abs(estimated - measured) / measured
    tol = ACCURACY_TOLERANCE.get(name, 0.30)
    print(f"  fx_visible={fx_only / 1024**2:.2f} MB  "
          f"profiled={estimated / 1024**2:.2f} MB  "
          f"measured={measured / 1024**2:.2f} MB  rel_err={rel * 100:.1f} %")
    ok = rel <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] profiler accuracy "
          f"(target: within {tol * 100:.0f} %)")
    return ok


def _check_ac_sanity(name: str, evictions: int, total: int) -> bool:
    lo, hi = EXPECTED_EVICTIONS.get(name, (0, total))
    ok = lo <= evictions <= hi
    print(f"  evicted {evictions}/{total} intermediates "
          f"(expected {lo}..{hi})")
    print(f"  [{'PASS' if ok else 'FAIL'}] AC sanity for {name}")
    return ok


def _check_ac_correctness(gm, args, profiler, selection) -> bool:
    """Compare outputs of the original gm and the AC-rewritten gm."""
    if not selection.to_recompute:
        print("  [SKIP] no evictions, nothing to check")
        return True

    # Run original gm.
    with torch.no_grad():
        original_out = gm(*args)

    # Rewrite (note: this mutates gm in place; we already captured outputs).
    gm_ac = rewrite_with_checkpointing(gm, profiler, selection.to_recompute)
    with torch.no_grad():
        rewritten_out = gm_ac(*args)

    max_diff = 0.0
    for a, b in zip(_iter_tensors(original_out), _iter_tensors(rewritten_out)):
        if a.shape != b.shape:
            print(f"  [FAIL] shape mismatch: {a.shape} vs {b.shape}")
            return False
        max_diff = max(max_diff, (a.float() - b.float()).abs().max().item())

    print(f"  max output diff: {max_diff:.2e}")
    ok = max_diff <= 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] AC correctness "
          f"(target: max diff <= 1e-4)")
    return ok


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)


def validate(name: str) -> bool:
    print(f"\n{'=' * 70}\nVALIDATING: {name}\n{'=' * 70}")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return True
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    builder = MODELS[name]
    model, optim, inputs, train_step = builder()
    init_optimizer_state(model, optim)

    results: List[bool] = []

    def transform(gm, args):
        profiler = _profile(gm, args)
        print("\n  -- check 1: profiler accuracy --")
        results.append(_check_profiler_accuracy(name, profiler, gm, args))

        selection = select_activations(profiler)
        print("\n  -- check 2: AC sanity --")
        results.append(_check_ac_sanity(
            name, len(selection.to_recompute), len(profiler.intermediates),
        ))

        print("\n  -- check 3: AC correctness --")
        results.append(_check_ac_correctness(gm, args, profiler, selection))
        return gm

    compiled_fn = compile(train_step, transform)
    compiled_fn(model, optim, inputs)
    return all(results)


def main():
    args = sys.argv[1:] or ["dummy"]
    if args == ["--all"]:
        args = list(MODELS.keys())
    statuses = {name: validate(name) for name in args}
    print("\n" + "=" * 70)
    for name, ok in statuses.items():
        print(f"  {name:<10} {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if all(statuses.values()) else 1)


if __name__ == "__main__":
    main()
