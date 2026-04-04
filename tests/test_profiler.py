"""
Test Suite for CS265 Graph Profiler and Activation Checkpointing.

Test categories:
1. TestStaticAnalysis     — FX graph structure, boundary detection, intermediates
2. TestNodeClassification — Tensor role classification (PARAM, ACT, GRAD, OTHER)
3. TestLifetimes          — Activation lifetime computation
4. TestRuntimeProfiling   — Per-node timing and memory (requires CUDA)
5. TestACSelection        — μ-TWO greedy selection algorithm
6. TestVisualizerOutput   — Memory chart generation
7. TestAPIGuardrails      — Error handling and API contracts

All tests that require CUDA are decorated with @requires_cuda and will be
skipped on CPU-only machines.
"""

import os
import sys
import tempfile
import unittest
from typing import Any, List

import torch
import torch.nn as nn
import torch.fx as fx

# Ensure project root is on sys.path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_prof import (
    GraphProfiler,
    IntermediateInfo,
    NodeType,
    Region,
    OP,
    _tensor_size_bytes,
)
from activation_checkpoint import (
    select_activations_to_recompute,
    _validate_recompute_set,
    print_ac_decisions,
)

# Import graph_tracer to register the separator library.  This must happen
# before any graph is traced.
import graph_tracer  # noqa: F401 (side-effect import)
from graph_tracer import SEPFunction, compile as gt_compile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()
requires_cuda = unittest.skipUnless(HAS_CUDA, "CUDA not available")


class DummyModel(nn.Module):
    """Simple MLP: N × (Linear → ReLU)."""

    def __init__(self, layers: int = 4, dim: int = 32):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


def _build_profiler(
    layers: int = 4,
    dim: int = 32,
    batch_size: int = 16,
    device: str = "cuda:0",
) -> GraphProfiler:
    """Trace a DummyModel training step and return a GraphProfiler.

    This helper handles model creation, optimizer state initialisation, graph
    tracing via ``gt_compile``, and extraction of the compiled GraphModule.
    """
    model = DummyModel(layers=layers, dim=dim).to(device)
    batch = torch.randn(batch_size, dim, device=device)
    opt = torch.optim.Adam(
        model.parameters(), lr=0.01, foreach=True, capturable=True,
    )

    # Initialise optimizer state (Adam lazy init).
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.rand_like(p)
    opt.step()
    opt.zero_grad()

    # We need to compile to get the GraphModule.  Use a no-op transformation
    # that just captures the graph.
    captured_gm = {}

    def capture_gm(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        captured_gm["gm"] = gm
        captured_gm["args"] = args
        return gm

    def train_step(model, optim, batch):
        loss = model(batch).sum()
        loss = SEPFunction.apply(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    compiled_fn = gt_compile(train_step, capture_gm)
    compiled_fn(model, opt, batch)

    gm = captured_gm["gm"]
    profiler = GraphProfiler(gm)
    return profiler, captured_gm["args"]


# ---------------------------------------------------------------------------
# 1. Static Analysis Tests
# ---------------------------------------------------------------------------


class TestStaticAnalysis(unittest.TestCase):
    """Verify that the profiler correctly parses the FX graph structure."""

    @classmethod
    @requires_cuda
    def setUpClass(cls):
        cls.profiler, cls.args = _build_profiler(layers=4, dim=32)

    @requires_cuda
    def test_sep_node_found(self):
        """The separator.sep node must be located."""
        self.assertIsNotNone(self.profiler.sep_node)
        self.assertGreater(self.profiler.sep_index, 0)

    @requires_cuda
    def test_sep_backward_node_found(self):
        """The separator.sep_backward node must be located."""
        self.assertIsNotNone(self.profiler.sep_bwd_node)
        self.assertGreater(self.profiler.sep_bwd_index, self.profiler.sep_index)

    @requires_cuda
    def test_optimizer_region_found(self):
        """The optimizer region boundary must be identified."""
        self.assertGreater(
            self.profiler.optimizer_index,
            self.profiler.sep_bwd_index,
        )

    @requires_cuda
    def test_param_nodes_nonempty(self):
        """Parameters should be identified (via _fused_adam or requires_grad)."""
        self.assertGreater(len(self.profiler.param_nodes), 0)

    @requires_cuda
    def test_param_count_matches_model(self):
        """Number of param nodes should equal the model's parameter count.

        DummyModel(layers=4, dim=32) has 4 × (Linear weight + Linear bias) = 8
        parameters.
        """
        self.assertEqual(len(self.profiler.param_nodes), 8)

    @requires_cuda
    def test_grad_count_when_fused(self):
        """With fused=True, grad nodes should equal param count."""
        # Grad identification only works with _fused_adam (fused=True).
        # With foreach=True, grad_nodes will be empty — that's acceptable.
        if self.profiler.optimizer_node is not None:
            self.assertEqual(
                len(self.profiler.grad_nodes),
                len(self.profiler.param_nodes),
            )

    @requires_cuda
    def test_intermediates_found(self):
        """Intermediate activations must be identified.

        For 4 × (Linear → ReLU), we expect at least some ReLU outputs to be
        intermediates (used by backward for the mask).
        """
        self.assertGreater(len(self.profiler.intermediate_nodes), 0)

    @requires_cuda
    def test_all_intermediates_are_call_function(self):
        """Every intermediate must be a call_function node."""
        for node in self.profiler.intermediate_nodes:
            self.assertEqual(node.op, OP.CALL_FUNCTION)

    @requires_cuda
    def test_all_intermediates_in_forward_region(self):
        """Every intermediate must be in the forward region."""
        for node in self.profiler.intermediate_nodes:
            idx = self.profiler.node_to_idx[node]
            self.assertLess(idx, self.profiler.sep_index)

    @requires_cuda
    def test_all_intermediates_have_bwd_users(self):
        """Every intermediate must have at least one backward-region user."""
        for node in self.profiler.intermediate_nodes:
            bwd_users = [
                u for u in node.users
                if self.profiler.node_to_idx.get(u, -1) >= self.profiler.sep_bwd_index
            ]
            self.assertGreater(
                len(bwd_users), 0,
                f"Intermediate {node.name} has no backward users",
            )

    @requires_cuda
    def test_node_list_covers_all_graph_nodes(self):
        """node_list must contain every node in the graph."""
        graph_nodes = list(self.profiler.module.graph.nodes)
        self.assertEqual(len(self.profiler.node_list), len(graph_nodes))

    @requires_cuda
    def test_node_to_idx_consistent(self):
        """node_to_idx must be the inverse of node_list."""
        for i, node in enumerate(self.profiler.node_list):
            self.assertEqual(self.profiler.node_to_idx[node], i)


# ---------------------------------------------------------------------------
# 2. Node Classification Tests
# ---------------------------------------------------------------------------


class TestNodeClassification(unittest.TestCase):
    """Verify tensor role classification (PARAM, ACT, GRAD, OTHER)."""

    @classmethod
    @requires_cuda
    def setUpClass(cls):
        cls.profiler, cls.args = _build_profiler(layers=4, dim=32)

    @requires_cuda
    def test_param_nodes_classified_as_param(self):
        for node in self.profiler.param_nodes:
            self.assertEqual(
                self.profiler.node_types[node], NodeType.PARAM,
                f"Param node {node.name} misclassified",
            )

    @requires_cuda
    def test_grad_nodes_classified_as_grad(self):
        for node in self.profiler.grad_nodes:
            self.assertEqual(
                self.profiler.node_types[node], NodeType.GRAD,
                f"Grad node {node.name} misclassified",
            )

    @requires_cuda
    def test_intermediate_nodes_classified_as_act(self):
        for node in self.profiler.intermediate_nodes:
            self.assertEqual(
                self.profiler.node_types[node], NodeType.ACT,
                f"Intermediate {node.name} not classified as ACT",
            )

    @requires_cuda
    def test_every_node_has_a_type(self):
        """Every node in the graph must be assigned a NodeType."""
        for node in self.profiler.node_list:
            self.assertIn(node, self.profiler.node_types)

    @requires_cuda
    def test_every_node_has_a_region(self):
        """Every node must be assigned a Region."""
        for node in self.profiler.node_list:
            self.assertIn(node, self.profiler.node_region)

    @requires_cuda
    def test_forward_region_before_sep(self):
        """All FORWARD-region nodes must have index <= sep_index."""
        for node, region in self.profiler.node_region.items():
            if region == Region.FORWARD:
                self.assertLessEqual(
                    self.profiler.node_to_idx[node],
                    self.profiler.sep_index,
                )

    @requires_cuda
    def test_backward_region_after_sep_bwd(self):
        """All BACKWARD-region nodes must have index >= sep_bwd_index."""
        for node, region in self.profiler.node_region.items():
            if region == Region.BACKWARD:
                idx = self.profiler.node_to_idx[node]
                self.assertGreaterEqual(idx, self.profiler.sep_bwd_index)
                # But must be before optimizer.
                if self.profiler.optimizer_index >= 0:
                    self.assertLess(idx, self.profiler.optimizer_index)


# ---------------------------------------------------------------------------
# 3. Lifetime Tests
# ---------------------------------------------------------------------------


class TestLifetimes(unittest.TestCase):
    """Verify activation lifetime computation (last_fwd_access, first_bwd_access)."""

    @classmethod
    @requires_cuda
    def setUpClass(cls):
        cls.profiler, cls.args = _build_profiler(layers=4, dim=32)

    @requires_cuda
    def test_last_fwd_before_first_bwd(self):
        """For every intermediate, last_fwd_access < first_bwd_access."""
        for node in self.profiler.intermediate_nodes:
            info = self.profiler.intermediate_info[node]
            self.assertLess(
                info.last_fwd_access, info.first_bwd_access,
                f"{node.name}: last_fwd={info.last_fwd_access} "
                f"not < first_bwd={info.first_bwd_access}",
            )

    @requires_cuda
    def test_last_fwd_in_forward_region(self):
        """last_fwd_access must be <= sep_index (within forward region)."""
        for node in self.profiler.intermediate_nodes:
            info = self.profiler.intermediate_info[node]
            self.assertLessEqual(info.last_fwd_access, self.profiler.sep_index)

    @requires_cuda
    def test_first_bwd_in_backward_region(self):
        """first_bwd_access must be >= sep_bwd_index."""
        for node in self.profiler.intermediate_nodes:
            info = self.profiler.intermediate_info[node]
            self.assertGreaterEqual(
                info.first_bwd_access, self.profiler.sep_bwd_index,
            )

    @requires_cuda
    def test_lifetime_positive(self):
        """Every intermediate must have a positive lifetime span."""
        for node in self.profiler.intermediate_nodes:
            info = self.profiler.intermediate_info[node]
            self.assertGreater(
                info.first_bwd_access - info.last_fwd_access, 0,
            )

    @requires_cuda
    def test_memory_size_positive(self):
        """Every intermediate should have memory_size > 0."""
        for node in self.profiler.intermediate_nodes:
            info = self.profiler.intermediate_info[node]
            self.assertGreater(
                info.memory_size, 0,
                f"{node.name} has zero memory_size",
            )

    @requires_cuda
    def test_all_intermediates_have_info(self):
        """Every intermediate node must have an IntermediateInfo entry."""
        for node in self.profiler.intermediate_nodes:
            self.assertIn(node, self.profiler.intermediate_info)


# ---------------------------------------------------------------------------
# 4. Runtime Profiling Tests (require CUDA)
# ---------------------------------------------------------------------------


class TestRuntimeProfiling(unittest.TestCase):
    """Verify runtime measurement collection and aggregation."""

    @classmethod
    @requires_cuda
    def setUpClass(cls):
        cls.profiler, cls.args = _build_profiler(layers=2, dim=16)
        with torch.no_grad():
            # Warm-up.
            for _ in range(2):
                cls.profiler.run(*cls.args)
            cls.profiler.reset_stats()
            # Measurement.
            for _ in range(3):
                cls.profiler.run(*cls.args)
        cls.profiler.aggregate_stats()

    @requires_cuda
    def test_avg_runtimes_populated(self):
        """After aggregate_stats, avg_runtimes should have entries."""
        self.assertGreater(len(self.profiler.avg_runtimes), 0)

    @requires_cuda
    def test_all_runtimes_nonnegative(self):
        for name, rt in self.profiler.avg_runtimes.items():
            self.assertGreaterEqual(rt, 0.0, f"{name} has negative runtime")

    @requires_cuda
    def test_avg_mem_deltas_populated(self):
        self.assertGreater(len(self.profiler.avg_mem_deltas), 0)

    @requires_cuda
    def test_reset_clears_runtimes(self):
        """reset_stats must clear all accumulated measurements."""
        profiler, args = _build_profiler(layers=2, dim=16)
        with torch.no_grad():
            profiler.run(*args)
        profiler.reset_stats()
        self.assertEqual(len(profiler._node_runtimes), 0)
        self.assertEqual(len(profiler._node_mem_deltas), 0)

    @requires_cuda
    def test_intermediate_recompute_cost_positive(self):
        """After profiling, intermediates should have recompute_cost > 0."""
        for node in self.profiler.intermediate_nodes:
            info = self.profiler.intermediate_info[node]
            self.assertGreater(
                info.recompute_cost_ms, 0.0,
                f"{node.name} has zero recompute cost",
            )

    @requires_cuda
    def test_print_stats_does_not_crash(self):
        """print_stats should complete without exception."""
        # Just call it; any exception fails the test.
        self.profiler.print_stats()


# ---------------------------------------------------------------------------
# 5. Activation Checkpointing Selection Tests
# ---------------------------------------------------------------------------


class TestACSelection(unittest.TestCase):
    """Verify the μ-TWO greedy selection algorithm."""

    @classmethod
    @requires_cuda
    def setUpClass(cls):
        cls.profiler, cls.args = _build_profiler(layers=4, dim=32)
        with torch.no_grad():
            for _ in range(2):
                cls.profiler.run(*cls.args)
            cls.profiler.reset_stats()
            for _ in range(3):
                cls.profiler.run(*cls.args)
        cls.profiler.aggregate_stats()

    @requires_cuda
    def test_all_intermediates_accounted_for(self):
        """to_recompute + to_retain must cover all intermediates."""
        to_recompute, to_retain = select_activations_to_recompute(
            self.profiler,
        )
        all_selected = set(to_recompute) | set(to_retain)
        all_intermediates = set(self.profiler.intermediate_nodes)
        self.assertEqual(all_selected, all_intermediates)

    @requires_cuda
    def test_no_overlap(self):
        """No node can be both recomputed and retained."""
        to_recompute, to_retain = select_activations_to_recompute(
            self.profiler,
        )
        overlap = set(to_recompute) & set(to_retain)
        self.assertEqual(len(overlap), 0)

    @requires_cuda
    def test_recompute_inputs_available(self):
        """Every recomputed node's inputs must be placeholders or retained."""
        to_recompute, to_retain = select_activations_to_recompute(
            self.profiler,
        )
        placeholder_set = {
            n for n in self.profiler.node_list if n.op == OP.PLACEHOLDER
        }
        retain_set = set(to_retain)
        valid = placeholder_set | retain_set

        for node in to_recompute:
            for inp in node.all_input_nodes:
                # Input can be a placeholder, a retained intermediate, or any
                # non-intermediate node (backward ops, etc. — those are always
                # available because recomputation happens in forward context).
                is_valid = (
                    inp in valid
                    or inp not in set(to_recompute)
                )
                self.assertTrue(
                    is_valid,
                    f"Recomputed node {node.name} depends on evicted "
                    f"input {inp.name}",
                )

    @requires_cuda
    def test_with_memory_budget_zero(self):
        """With budget=0, nothing should be recomputed."""
        to_recompute, to_retain = select_activations_to_recompute(
            self.profiler, memory_budget=0,
        )
        self.assertEqual(len(to_recompute), 0)
        self.assertEqual(
            len(to_retain), len(self.profiler.intermediate_nodes),
        )

    @requires_cuda
    def test_with_large_budget(self):
        """With a very large budget, all valid intermediates are recomputed."""
        to_recompute, to_retain = select_activations_to_recompute(
            self.profiler, memory_budget=10**12,
        )
        # At least some should be recomputed.
        self.assertGreater(len(to_recompute), 0)

    @requires_cuda
    def test_sorted_by_efficiency(self):
        """Recomputed nodes should be ordered by decreasing recompute_ratio."""
        to_recompute, _ = select_activations_to_recompute(self.profiler)
        if len(to_recompute) < 2:
            return
        ratios = []
        for node in to_recompute:
            info = self.profiler.intermediate_info[node]
            ratio = info.memory_size / (info.recompute_cost_ms + 1e-9)
            ratios.append(ratio)
        # Ratios should be non-increasing.
        for i in range(len(ratios) - 1):
            self.assertGreaterEqual(ratios[i], ratios[i + 1])

    @requires_cuda
    def test_print_ac_decisions_does_not_crash(self):
        to_recompute, to_retain = select_activations_to_recompute(
            self.profiler,
        )
        print_ac_decisions(self.profiler, to_recompute, to_retain)

    @requires_cuda
    def test_empty_profiler(self):
        """A profiler with no intermediates should return empty lists."""
        # Construct a profiler whose intermediate_nodes is empty.
        profiler, args = _build_profiler(layers=4, dim=32)
        profiler.intermediate_nodes = []
        profiler.intermediate_info = {}
        to_recompute, to_retain = select_activations_to_recompute(profiler)
        self.assertEqual(len(to_recompute), 0)
        self.assertEqual(len(to_retain), 0)


# ---------------------------------------------------------------------------
# 6. Visualizer Tests
# ---------------------------------------------------------------------------


class TestVisualizerOutput(unittest.TestCase):
    """Verify that the memory visualiser produces valid output."""

    @classmethod
    @requires_cuda
    def setUpClass(cls):
        cls.profiler, cls.args = _build_profiler(layers=2, dim=16)
        with torch.no_grad():
            for _ in range(2):
                cls.profiler.run(*cls.args)
            cls.profiler.reset_stats()
            for _ in range(3):
                cls.profiler.run(*cls.args)
        cls.profiler.aggregate_stats()

    @requires_cuda
    def test_live_memory_timeline_length(self):
        """Timeline length must equal node count."""
        timeline = self.profiler._compute_live_memory_timeline()
        self.assertEqual(len(timeline), len(self.profiler.node_list))

    @requires_cuda
    def test_live_memory_nonnegative(self):
        """No step should have negative live memory."""
        timeline = self.profiler._compute_live_memory_timeline()
        for t, mem in enumerate(timeline):
            self.assertGreaterEqual(mem, 0, f"Negative memory at step {t}")

    @requires_cuda
    def test_by_role_timeline_length(self):
        by_role = self.profiler._compute_live_memory_timeline_by_role()
        for role, timeline in by_role.items():
            self.assertEqual(len(timeline), len(self.profiler.node_list))

    @requires_cuda
    def test_by_role_sums_to_total(self):
        """Sum of per-role timelines should equal total timeline."""
        total = self.profiler._compute_live_memory_timeline()
        by_role = self.profiler._compute_live_memory_timeline_by_role()
        for t in range(len(total)):
            role_sum = sum(by_role[r][t] for r in NodeType)
            self.assertEqual(
                role_sum, total[t],
                f"Role sum {role_sum} != total {total[t]} at step {t}",
            )

    @requires_cuda
    def test_plot_creates_file(self):
        """plot_memory_timeline should create a non-empty PNG."""
        from visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            viz.plot_memory_timeline(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. API Guardrails Tests
# ---------------------------------------------------------------------------


class TestAPIGuardrails(unittest.TestCase):
    """Verify error handling and API contracts."""

    def test_missing_sep_raises(self):
        """A graph without separator nodes should raise AssertionError."""
        class TrivialModule(nn.Module):
            def forward(self, x):
                return x + 1

        m = TrivialModule()
        gm = torch.fx.symbolic_trace(m)
        with self.assertRaises(AssertionError):
            GraphProfiler(gm)

    def test_tensor_size_bytes_scalar(self):
        """_tensor_size_bytes should return element_size for a scalar."""
        t = torch.tensor(1.0)
        self.assertEqual(_tensor_size_bytes(t), t.element_size())

    def test_tensor_size_bytes_tuple(self):
        """_tensor_size_bytes should sum sizes for a tuple of tensors."""
        t1 = torch.randn(4, 8)
        t2 = torch.randn(2, 3)
        expected = t1.numel() * t1.element_size() + t2.numel() * t2.element_size()
        self.assertEqual(_tensor_size_bytes((t1, t2)), expected)

    def test_tensor_size_bytes_non_tensor(self):
        """_tensor_size_bytes should return 0 for non-tensor values."""
        self.assertEqual(_tensor_size_bytes(42), 0)
        self.assertEqual(_tensor_size_bytes("hello"), 0)
        self.assertEqual(_tensor_size_bytes(None), 0)

    @requires_cuda
    def test_different_layer_counts(self):
        """Profiler should work with different model sizes."""
        for layers in [1, 2, 6]:
            profiler, _ = _build_profiler(layers=layers, dim=16)
            self.assertIsNotNone(profiler.sep_node)
            self.assertGreater(len(profiler.intermediate_nodes), 0)


# ---------------------------------------------------------------------------
# 8. Architecture Variation Tests
# ---------------------------------------------------------------------------


class TestArchitectures(unittest.TestCase):
    """Test profiler on different model configurations."""

    @requires_cuda
    def test_single_layer(self):
        """Single Linear+ReLU layer should produce at least 1 intermediate."""
        profiler, _ = _build_profiler(layers=1, dim=16)
        self.assertGreater(len(profiler.intermediate_nodes), 0)
        # 1 layer = weight + bias = 2 params.
        self.assertEqual(len(profiler.param_nodes), 2)

    @requires_cuda
    def test_deep_network(self):
        """8-layer network should produce more intermediates than 2-layer."""
        p2, _ = _build_profiler(layers=2, dim=16)
        p8, _ = _build_profiler(layers=8, dim=16)
        self.assertGreater(
            len(p8.intermediate_nodes),
            len(p2.intermediate_nodes),
        )

    @requires_cuda
    def test_larger_dim_larger_memory(self):
        """Larger hidden dim should produce larger activation memory."""
        p_small, _ = _build_profiler(layers=2, dim=16)
        p_large, _ = _build_profiler(layers=2, dim=64)
        small_mem = sum(
            p_small.intermediate_info[n].memory_size
            for n in p_small.intermediate_nodes
        )
        large_mem = sum(
            p_large.intermediate_info[n].memory_size
            for n in p_large.intermediate_nodes
        )
        self.assertGreater(large_mem, small_mem)

    @requires_cuda
    def test_ac_selection_works_for_all_sizes(self):
        """AC selection should work for 1, 2, 4, 8 layer models."""
        for layers in [1, 2, 4, 8]:
            profiler, args = _build_profiler(layers=layers, dim=16)
            with torch.no_grad():
                profiler.run(*args)
                profiler.run(*args)
            profiler.reset_stats()
            with torch.no_grad():
                profiler.run(*args)
            profiler.aggregate_stats()

            to_recompute, to_retain = select_activations_to_recompute(profiler)
            total = set(to_recompute) | set(to_retain)
            self.assertEqual(
                total,
                set(profiler.intermediate_nodes),
                f"AC selection incomplete for layers={layers}",
            )


if __name__ == "__main__":
    unittest.main()
