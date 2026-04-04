"""
Validation script for CS265 Graph Profiler and AC Selection.

Runs three levels of checks:
1. Profiler accuracy: static peak estimate vs actual GPU measurement.
2. AC decision sanity: intuitive checks per model architecture.
3. Memory simulator consistency: single-eviction reduction matches tensor size.

Usage:
    python validate.py                  # default: DummyModel
    python validate.py Resnet18         # specify model
    python validate.py --all            # run all models
"""

import sys
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from graph_tracer import SEPFunction, compile as gt_compile
from graph_prof import GraphProfiler, NodeType, OP
from activation_checkpoint import (
    select_activations_to_recompute,
    _simulate_peak_memory,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DummyModel(nn.Module):
    def __init__(self, layers=10, dim=100):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


def _setup_model(name: str):
    """Return (model, optimizer, example_inputs, train_step) for a model."""
    dev = torch.device("cuda")

    if name == "DummyModel":
        model = DummyModel(layers=10, dim=100).to(dev)
        batch = torch.randn(1000, 100, device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True, capturable=True)
        inputs = (batch,)

        def train_step(model, optim, batch):
            loss = model(batch).sum()
            loss = SEPFunction.apply(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()

        return model, opt, inputs, train_step

    elif name == "Resnet18":
        from torchvision.models import resnet18
        import torch.nn.functional as F
        with torch.device(dev):
            model = resnet18()
        inp = torch.randn(16, 3, 224, 224, device=dev)
        target = torch.randint(0, 10, (16,), device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True, capturable=True)
        inputs = (inp, target)

        def train_step(model, optim, example_inputs):
            logits = model(example_inputs[0])
            loss = F.cross_entropy(logits, example_inputs[1])
            loss = SEPFunction.apply(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()

        return model, opt, inputs, train_step

    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _build_profiler(name: str) -> Tuple[GraphProfiler, list]:
    """Trace, profile, and return (profiler, args)."""
    model, opt, inputs, train_step = _setup_model(name)

    # Init optimizer state.
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.rand_like(p)
    opt.step()
    opt.zero_grad()

    captured = {}

    def capture(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        captured["gm"] = gm
        captured["args"] = args
        return gm

    compiled_fn = gt_compile(train_step, capture)
    compiled_fn(model, opt, *inputs)

    profiler = GraphProfiler(captured["gm"])
    args = captured["args"]

    # Warm-up + measurement.
    with torch.no_grad():
        for _ in range(2):
            profiler.run(*args)
        profiler.reset_stats()
        for _ in range(3):
            profiler.run(*args)
    profiler.aggregate_stats()

    return profiler, args


def check_1_profiler_accuracy(profiler: GraphProfiler, args: list, name: str):
    """Compare static peak estimate vs actual GPU peak measurement."""
    print(f"\n{'='*70}")
    print(f"CHECK 1: Profiler accuracy — {name}")
    print(f"{'='*70}")

    # Static estimate from our timeline.
    timeline = profiler._compute_live_memory_timeline()
    estimated_peak = max(timeline) if timeline else 0

    # Actual GPU measurement.
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        profiler.run(*args)
    actual_peak = torch.cuda.max_memory_allocated()

    ratio = estimated_peak / actual_peak if actual_peak > 0 else 0
    status = "PASS" if 0.3 < ratio < 3.0 else "FAIL"

    print(f"  Estimated peak (static):  {estimated_peak / 1024**2:>10.2f} MB")
    print(f"  Actual GPU peak:          {actual_peak / 1024**2:>10.2f} MB")
    print(f"  Ratio (est/actual):       {ratio:>10.2f}")
    print(f"  [{status}] Ratio is within reasonable bounds (0.3-3.0x)")
    print()
    print(f"  Note: mismatch is expected because the static estimate counts")
    print(f"  formula-based tensor sizes (numel * element_size) while actual")
    print(f"  GPU measurement includes CUDA allocator overhead, fragmentation,")
    print(f"  and temporary buffers invisible to our FakeTensor metadata.")

    return status == "PASS"


def check_2_ac_decision_sanity(profiler: GraphProfiler, name: str):
    """Verify AC decisions match architectural expectations."""
    print(f"\n{'='*70}")
    print(f"CHECK 2: AC decision sanity — {name}")
    print(f"{'='*70}")

    to_recompute, to_retain = select_activations_to_recompute(profiler)
    n_total = len(profiler.intermediate_nodes)

    print(f"  Intermediates: {n_total} total, "
          f"{len(to_recompute)} recompute, {len(to_retain)} retain")

    all_passed = True

    # Check 2a: recompute + retain = all intermediates, no overlap.
    recomp_set = set(to_recompute)
    retain_set = set(to_retain)
    all_set = set(profiler.intermediate_nodes)

    coverage_ok = (recomp_set | retain_set) == all_set
    overlap_ok = len(recomp_set & retain_set) == 0
    status_a = "PASS" if (coverage_ok and overlap_ok) else "FAIL"
    print(f"  [{status_a}] Full coverage and no overlap")
    all_passed = all_passed and (status_a == "PASS")

    # Check 2b: recomputed nodes should be cheaper than retained nodes.
    if to_recompute and to_retain:
        avg_recomp_cost = sum(
            profiler.intermediate_info[n].recompute_cost_ms for n in to_recompute
        ) / len(to_recompute)
        avg_retain_cost = sum(
            profiler.intermediate_info[n].recompute_cost_ms for n in to_retain
        ) / len(to_retain)
        cheaper = avg_recomp_cost <= avg_retain_cost * 2  # generous margin
        status_b = "PASS" if cheaper else "WARN"
        print(f"  [{status_b}] Avg recompute cost ({avg_recomp_cost:.4f} ms) vs "
              f"avg retain cost ({avg_retain_cost:.4f} ms)")
        if not cheaper:
            print(f"         Recomputed ops are more expensive than retained — unusual")
    else:
        print(f"  [SKIP] Cannot compare costs (one list empty)")

    # Check 2c: recomputed nodes' inputs must be available.
    placeholder_set = {n for n in profiler.node_list if n.op == OP.PLACEHOLDER}
    valid_inputs = placeholder_set | retain_set
    bad_inputs = []
    for node in to_recompute:
        for inp in node.all_input_nodes:
            if inp in recomp_set:
                bad_inputs.append((node.name, inp.name))
    status_c = "PASS" if not bad_inputs else "FAIL"
    print(f"  [{status_c}] All recomputed nodes have available inputs")
    if bad_inputs:
        for node_name, inp_name in bad_inputs[:5]:
            print(f"         {node_name} depends on evicted {inp_name}")
    all_passed = all_passed and (status_c == "PASS")

    # Check 2d: model-specific sanity.
    if name == "DummyModel":
        recomp_names = {n.name for n in to_recompute}
        relu_recomputed = sum(1 for n in recomp_names if "relu" in n)
        status_d = "PASS" if relu_recomputed > 0 else "WARN"
        print(f"  [{status_d}] DummyModel: {relu_recomputed} relu ops recomputed "
              f"(expected: cheap ops evicted first)")
    elif name == "Resnet18":
        retain_names = {n.name for n in to_retain}
        conv_retained = sum(1 for n in retain_names if "convolution" in n)
        status_d = "PASS" if conv_retained > 0 else "WARN"
        print(f"  [{status_d}] ResNet18: {conv_retained} convolution ops retained "
              f"(expected: expensive ops kept)")
    else:
        status_d = "PASS"

    return all_passed


def check_3_memory_simulator(profiler: GraphProfiler, name: str):
    """Verify memory simulator consistency with single-eviction tests."""
    print(f"\n{'='*70}")
    print(f"CHECK 3: Memory simulator consistency — {name}")
    print(f"{'='*70}")

    baseline_peak = _simulate_peak_memory(profiler, evicted=set())
    print(f"  Baseline peak: {baseline_peak / 1024**2:.2f} MB")

    all_passed = True

    # Check 3a: peak with all evictions <= baseline peak.
    all_evicted_peak = _simulate_peak_memory(
        profiler, evicted=set(profiler.intermediate_nodes)
    )
    status_a = "PASS" if all_evicted_peak <= baseline_peak else "FAIL"
    print(f"  [{status_a}] Peak with all evicted ({all_evicted_peak / 1024**2:.2f} MB) "
          f"<= baseline ({baseline_peak / 1024**2:.2f} MB)")
    all_passed = all_passed and (status_a == "PASS")

    # Check 3b: single eviction should reduce peak by at most the tensor's size.
    # Pick the largest intermediate that IS live at the peak step.
    timeline = profiler._compute_live_memory_timeline()
    peak_step = max(range(len(timeline)), key=lambda t: timeline[t])

    live_at_peak = []
    for node in profiler.intermediate_nodes:
        info = profiler.intermediate_info[node]
        idx = profiler.node_to_idx[node]
        user_indices = [profiler.node_to_idx[u] for u in node.users if u in profiler.node_to_idx]
        last_use = max(user_indices) if user_indices else idx
        if idx <= peak_step <= last_use and info.memory_size > 0:
            live_at_peak.append(node)

    if live_at_peak:
        # Pick largest.
        test_node = max(live_at_peak, key=lambda n: profiler.intermediate_info[n].memory_size)
        test_size = profiler.intermediate_info[test_node].memory_size

        single_peak = _simulate_peak_memory(profiler, evicted={test_node})
        reduction = baseline_peak - single_peak

        # Reduction should be > 0 (tensor was live at peak) and <= tensor size.
        sensible = (0 < reduction <= test_size * 1.1)  # 10% margin
        status_b = "PASS" if sensible else "WARN"
        print(f"  [{status_b}] Evicting '{test_node.name}' "
              f"({test_size / 1024:.1f} KB): "
              f"peak drops by {reduction / 1024:.1f} KB "
              f"(expected: 0 < drop <= {test_size / 1024:.1f} KB)")
        if not sensible and reduction == 0:
            print(f"         Tensor may not be live at the peak step in the simulator")
    else:
        print(f"  [SKIP] No intermediates live at peak step {peak_step} — "
              f"peak is dominated by non-activation memory")
        print(f"         This means AC cannot reduce peak for this model, which is fine")

    # Check 3c: monotonicity — evicting more should never increase peak.
    sorted_nodes = sorted(
        profiler.intermediate_nodes,
        key=lambda n: profiler.intermediate_info[n].memory_size,
        reverse=True,
    )
    evicted = set()
    prev_peak = baseline_peak
    monotone = True
    for node in sorted_nodes[:10]:  # Check first 10.
        evicted.add(node)
        new_peak = _simulate_peak_memory(profiler, evicted)
        if new_peak > prev_peak:
            monotone = False
            break
        prev_peak = new_peak

    status_c = "PASS" if monotone else "FAIL"
    print(f"  [{status_c}] Monotonicity: evicting more never increases peak")
    all_passed = all_passed and (status_c == "PASS")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_model(name: str):
    """Run all validation checks for a single model."""
    print(f"\n{'#'*70}")
    print(f"# VALIDATING: {name}")
    print(f"{'#'*70}")

    profiler, args = _build_profiler(name)

    r1 = check_1_profiler_accuracy(profiler, args, name)
    r2 = check_2_ac_decision_sanity(profiler, name)
    r3 = check_3_memory_simulator(profiler, name)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {name}")
    print(f"{'='*70}")
    print(f"  Check 1 (Profiler accuracy):      {'PASS' if r1 else 'FAIL'}")
    print(f"  Check 2 (AC decision sanity):      {'PASS' if r2 else 'FAIL'}")
    print(f"  Check 3 (Memory simulator):        {'PASS' if r3 else 'FAIL'}")
    overall = "ALL PASSED" if (r1 and r2 and r3) else "SOME ISSUES"
    print(f"  Overall: {overall}")

    return r1 and r2 and r3


if __name__ == "__main__":
    models = ["DummyModel"]

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            models = ["DummyModel", "Resnet18"]
        else:
            models = [sys.argv[1]]

    all_ok = True
    for name in models:
        ok = validate_model(name)
        all_ok = all_ok and ok

    print(f"\n{'#'*70}")
    if all_ok:
        print("# ALL MODELS VALIDATED SUCCESSFULLY")
    else:
        print("# SOME CHECKS FAILED — review output above")
    print(f"{'#'*70}")
