from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List, Optional
import torch
import torch.fx as fx
from activation_checkpoint import (print_ac_decisions, rewrite_with_checkpointing, select_activations)
from graph_prof import GraphProfiler
from graph_tracer import compile
from models import MODELS, init_optimizer_state
from visualizer import (plot_latency_comparison_vs_batch, plot_memory_breakdown, plot_peak_memory_vs_batch)

PLOTS_DIR = "plots"
model_names: List[str] = ["dummy", "resnet18", "resnet50", "bert"]
model_batch_sizes: Dict[str, List[int]] = { "dummy": [4, 8, 16, 32], "resnet18": [4, 8, 16, 32], "resnet50": [4, 8, 16, 32], "bert": [4, 8, 16, 32]}
WARMUP_ITERS, MEASURE_ITERS = 3, 5


class Experiment:
    def __init__(self, model_name: str, batch_size: int, mem_limit_bytes: Optional[int] = None):
        assert model_name in MODELS, (f"Model {model_name} not found in {list(MODELS)}")
        torch.manual_seed(0)
        torch.cuda.empty_cache()

        self.model_name = model_name
        self.batch_size = batch_size
        self.mem_limit_bytes = mem_limit_bytes
        self.metrics: dict = {"model": model_name, "batch_size": batch_size}

        (self.model, self.optimizer, self.example_inputs, self.train_step) = MODELS[model_name](batch_size=batch_size)

    def init_opt_states(self) -> None:
        init_optimizer_state(self.model, self.optimizer)

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        print(f"\n{'=' * 70}\n"
              f"BENCHMARK: {self.model_name}  (batch_size={self.batch_size})\n"
              f"{'=' * 70}")
        
        prof_no_ac = self._profile(gm, args)
        print("\n  -- no AC --")
        prof_no_ac.print_summary()

        peak_no_ac_mb    = prof_no_ac.peak_memory_bytes() / 1024**2
        latency_no_ac_ms = prof_no_ac.iteration_latency_ms()
        self._save_breakdown_plot(prof_no_ac, "no_ac")

        selection = select_activations(prof_no_ac, mem_limit=self.mem_limit_bytes)
        print_ac_decisions(prof_no_ac, selection)

        gm_ac = rewrite_with_checkpointing(gm, prof_no_ac, selection.to_recompute)
        baseline   = self._snapshot_state()
        with torch.no_grad():
            gm(*self._clone_args(args))
        after_orig = self._snapshot_state()

        self._restore_state(baseline)
        with torch.no_grad():
            gm_ac(*self._clone_args(args))
        after_ac   = self._snapshot_state()
        self._restore_state(baseline)           

        max_diff = self._max_state_diff(after_orig, after_ac)
        ok = max_diff <= 1e-3                  
        print(f"\n  Param-update diff (max abs): {max_diff:.2e}  "
              f"[{'PASS' if ok else 'FAIL'}]")

        prof_ac = self._profile(gm_ac, args)
        print("\n  -- with AC --")
        prof_ac.print_summary()
        self.metrics.update({
            "n_recompute":      len(selection.to_recompute),
            "n_retain":         len(selection.to_retain),
            "peak_no_ac_mb":    peak_no_ac_mb,
            "peak_ac_mb":       prof_ac.peak_memory_bytes()    / 1024**2,
            "latency_no_ac_ms": latency_no_ac_ms,
            "latency_ac_ms":    prof_ac.iteration_latency_ms(),
            "max_output_diff":  max_diff,
            "ac_correct":       ok,
            "selection_reason": selection.reason,
        })

        self._save_breakdown_plot(prof_ac, "ac")
        return gm_ac

    def _profile(self, gm: fx.GraphModule, args: Any) -> GraphProfiler:
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(MEASURE_ITERS):
                profiler.run(*args)
        profiler.aggregate_stats()
        return profiler

    def _clone_args(self, args: Any) -> Any:
        def clone(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.clone()
            if isinstance(obj, (list, tuple)):
                return type(obj)(clone(x) for x in obj)
            return obj
        return clone(args)

    def _snapshot_state(self):
        params  = [p.detach().clone() for p in self.model.parameters()]
        buffers = [b.detach().clone() for b in self.model.buffers()]
        optim:  dict = {}
        for p_state in self.optimizer.state.values():
            for v in p_state.values():
                if isinstance(v, torch.Tensor):
                    optim[id(v)] = v.detach().clone()
        return params, buffers, optim

    def _restore_state(self, snapshot) -> None:
        params, buffers, optim = snapshot
        for p, p0 in zip(self.model.parameters(), params):
            p.data.copy_(p0)
        for b, b0 in zip(self.model.buffers(), buffers):
            b.data.copy_(b0)
        for p_state in self.optimizer.state.values():
            for v in p_state.values():
                if isinstance(v, torch.Tensor) and id(v) in optim:
                    v.copy_(optim[id(v)])

    def _max_state_diff(self, snap1, snap2) -> float:
        params1, buffers1, optim1 = snap1
        params2, buffers2, optim2 = snap2
        diffs: list[float] = []
        for p1, p2 in zip(params1, params2):
            diffs.append((p1.float() - p2.float()).abs().max().item())
        for b1, b2 in zip(buffers1, buffers2):
            diffs.append((b1.float() - b2.float()).abs().max().item())
        for k in optim1:
            if k in optim2:
                diffs.append((optim1[k].float() - optim2[k].float()).abs().max().item())
        return max(diffs) if diffs else 0.0

    def _save_breakdown_plot(self, profiler: GraphProfiler, suffix: str) -> None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        base    = f"{self.model_name}_bs{self.batch_size}"
        title   = ("no AC" if suffix == "no_ac" else "with AC")
        plot_memory_breakdown(profiler, f"{PLOTS_DIR}/benchmark_memory_{base}_{suffix}.png", title=f"{self.model_name} — Memory Breakdown, {title} (bs={self.batch_size})")

    def run(self) -> dict:
        self.init_opt_states()
        compiled_fn = compile(self.train_step, self.graph_transformation)
        compiled_fn(self.model, self.optimizer, self.example_inputs)
        return self.metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", help=f"models to run (any of: {model_names}). Default: all.")
    p.add_argument("-b", "--batch-sizes", type=int, nargs="+", default=None, help="batch sizes (overrides per-model defaults).")
    p.add_argument("--mem-limit-mb", type=float, default=None, help="absolute target peak (MB) for the AC selector. Default: cut peak by half the activation memory.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; this script needs a GPU.")
        return

    chosen = args.models or model_names
    for name in chosen:
        if name not in MODELS:
            print(f"unknown model '{name}'; choose from {list(MODELS)}")
            return

    limit_bytes = (None if args.mem_limit_mb is None
                   else int(args.mem_limit_mb * 1024 ** 2))

    all_metrics: Dict[str, List[dict]] = {name: [] for name in chosen}
    for name in chosen:
        bs_list = args.batch_sizes or model_batch_sizes.get(name, [4, 8, 16, 32])
        for bs in bs_list:
            try:
                exp = Experiment(name, bs, mem_limit_bytes=limit_bytes)
                all_metrics[name].append(exp.run())
            except torch.cuda.OutOfMemoryError as exc:
                print(f"  [OOM] {name} bs={bs}: {exc}")
                torch.cuda.empty_cache()

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
            f"{PLOTS_DIR}/benchmark_peak_vs_batch_{name}.png",
            f"{name} — peak memory vs batch size",
        )
        plot_latency_comparison_vs_batch(
            latency_rows,
            f"{PLOTS_DIR}/benchmark_latency_vs_batch_{name}.png",
            f"{name} — latency vs batch size",
        )

    if any(all_metrics.values()):
        print(f"\n{'=' * 96}\nBENCHMARK SWEEP SUMMARY\n{'=' * 96}")
        print(f"{'Model':<10} {'BS':>4} {'Recomp':>7} {'Retain':>7}"
              f" {'Peak off':>10} {'Peak AC':>10}"
              f" {'Lat off':>10} {'Lat AC':>10}"
              f" {'GradDiff':>10}")
        print("-" * 96)
        for name, rows in all_metrics.items():
            for r in rows:
                print(f"{name:<10} {r['batch_size']:>4}"
                      f" {r['n_recompute']:>7} {r['n_retain']:>7}"
                      f" {r['peak_no_ac_mb']:>7.1f} MB"
                      f" {r['peak_ac_mb']:>7.1f} MB"
                      f" {r['latency_no_ac_ms']:>7.1f} ms"
                      f" {r['latency_ac_ms']:>7.1f} ms"
                      f" {r['max_output_diff']:>9.2e}")
        print("=" * 96)

    print(f"\nDone. Plots in ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
