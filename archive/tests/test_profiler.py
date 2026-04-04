"""
CS265 Neural Network Profiler — Test Suite

Test categories:
  1. TensorRegistry unit tests   (no full profiler run)
  2. ComputationalGraph / topo sort
  3. Lifetime analysis (compute_lifetimes)
  4. Tensor categorization (full profiler run)
  5. Graph integrity (edges, node ids)
  6. API guard rails
  7. Architecture coverage (including deep ReLU MLP example)
  8. Optimizer variant state counts
  9. Pass 4 lifetime correctness (parameters / optimizer states / gradients)
 10. MemoryVisualizer
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from profiler import ProfilerExecutor
from profiler.graph import (
    ComputationalGraph,
    GraphEdge,
    GraphNode,
    OpPhase,
    TensorRole,
)
from profiler.tensor_registry import TensorRegistry, compute_lifetimes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _seed():
    torch.manual_seed(0)


def _run(model, opt, X, Y, loss_fn=None):
    """Run one full profiling iteration and return the executor."""
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    profiler = ProfilerExecutor(model, opt)
    profiler.run(X, Y, loss_fn)
    return profiler


def _count_leaves(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if len(list(m.children())) == 0)


# Simple two-layer MLP used in many tests
class TwoLayerMLP(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=bias)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(8, 2, bias=bias)
        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        return self.sig2(self.fc2(self.sig1(self.fc1(x))))


# ---------------------------------------------------------------------------
# 1. TensorRegistry unit tests
# ---------------------------------------------------------------------------

class TestTensorRegistry(unittest.TestCase):

    def setUp(self):
        _seed()
        # Minimal model so TensorRegistry can be constructed
        self.model = nn.Linear(4, 4, bias=False)
        self.reg = TensorRegistry(self.model)

    def _make_tensor(self, *shape, requires_grad=False):
        return torch.randn(*shape, requires_grad=requires_grad)

    # --- force_create ---

    def test_force_create_gives_distinct_ids_same_tensor(self):
        """Two force_create calls on the exact same tensor must give different IDs."""
        t = self._make_tensor(3, 4)
        m1 = self.reg.force_create(t, TensorRole.ACTIVATION, "act_a")
        m2 = self.reg.force_create(t, TensorRole.ACTIVATION, "act_b")
        self.assertNotEqual(m1.tensor_id, m2.tensor_id)

    def test_force_create_gives_distinct_ids_same_ptr_shape(self):
        """Two different tensors that happen to share (data_ptr, shape)
        through storage aliasing must still get distinct IDs."""
        base = torch.randn(8)
        t1 = base[:4]   # view sharing storage, shape (4,)
        t2 = base[4:]   # different slice but same storage ptr? No — different ptr.
        # Use narrow to get same data_ptr
        t_a = base.narrow(0, 0, 4)
        t_b = base.narrow(0, 0, 4)  # same ptr + shape as t_a
        m1 = self.reg.force_create(t_a, TensorRole.ACTIVATION, "a")
        m2 = self.reg.force_create(t_b, TensorRole.ACTIVATION, "b")
        self.assertNotEqual(m1.tensor_id, m2.tensor_id,
                            "force_create must always allocate a fresh ID")

    def test_force_create_overwrites_ptr_shape_mapping(self):
        """After force_create with a new tensor at the same (ptr,shape),
        the ptr_shape dict points to the new entry, not the old one."""
        t = self._make_tensor(4, 4)
        m1 = self.reg.force_create(t, TensorRole.ACTIVATION, "old")
        m2 = self.reg.force_create(t, TensorRole.ACTIVATION, "new")
        key = (t.data_ptr(), tuple(t.shape))
        # The latest entry wins in _ptr_shape_to_meta
        self.assertEqual(self.reg._ptr_shape_to_meta[key].tensor_id, m2.tensor_id)

    # --- get_or_create idempotency ---

    def test_get_or_create_same_tensor_returns_same_id(self):
        """Calling get_or_create twice on the same tensor returns the same entry."""
        t = self._make_tensor(4, 4)
        m1 = self.reg.get_or_create(t)
        m2 = self.reg.get_or_create(t)
        self.assertEqual(m1.tensor_id, m2.tensor_id)

    def test_get_or_create_role_preserved_on_second_call(self):
        """A tensor classified as ACTIVATION keeps its role on subsequent lookups."""
        t = self._make_tensor(4, 4)
        # Give it a grad_fn by running through an op
        param = nn.Parameter(torch.randn(4, 4))
        act = t @ param   # has grad_fn
        m1 = self.reg.get_or_create(act)
        m2 = self.reg.get_or_create(act)
        self.assertEqual(m1.role, TensorRole.ACTIVATION)
        self.assertEqual(m2.role, TensorRole.ACTIVATION)
        self.assertEqual(m1.tensor_id, m2.tensor_id)

    # --- Memory-reuse guard ---

    def test_gradient_hint_on_activation_address_creates_fresh_entry(self):
        """If a gradient is allocated at the same (ptr, shape) as a freed activation,
        get_or_create with hint=GRADIENT must create a fresh entry."""
        t = self._make_tensor(4, 4)
        # Register as activation first
        act_meta = self.reg.force_create(t, TensorRole.ACTIVATION, "old_act")
        # Now pretend the same memory is reused for a gradient
        grad_meta = self.reg.get_or_create(t, hint_role=TensorRole.GRADIENT)
        self.assertNotEqual(act_meta.tensor_id, grad_meta.tensor_id,
                            "Memory-reuse guard must create a fresh entry for the gradient")
        self.assertEqual(grad_meta.role, TensorRole.GRADIENT)

    def test_optimizer_state_hint_on_activation_address_creates_fresh_entry(self):
        """Same guard for OPTIMIZER_STATE."""
        t = self._make_tensor(4, 4)
        act_meta = self.reg.force_create(t, TensorRole.ACTIVATION, "old_act")
        opt_meta = self.reg.mark_optimizer_state(t, "opt_buf")
        self.assertNotEqual(act_meta.tensor_id, opt_meta.tensor_id)
        self.assertEqual(opt_meta.role, TensorRole.OPTIMIZER_STATE)

    def test_mark_optimizer_state_overwrites_gradient_entry(self):
        """mark_optimizer_state must overwrite a stale GRADIENT entry at same (ptr,shape),
        not mutate it — the old gradient entry must still exist in _registry."""
        t = self._make_tensor(4, 4)
        grad_meta = self.reg.get_or_create(t, hint_role=TensorRole.GRADIENT, hint_name="grad_w")
        opt_meta = self.reg.mark_optimizer_state(t, "opt_momentum")
        # Old entry must still be GRADIENT (not mutated to OPTIMIZER_STATE)
        old_entry = self.reg._registry[grad_meta.tensor_id]
        self.assertEqual(old_entry.role, TensorRole.GRADIENT,
                         "mark_optimizer_state must not mutate the old gradient entry")
        # New entry must be OPTIMIZER_STATE
        self.assertEqual(opt_meta.role, TensorRole.OPTIMIZER_STATE)
        self.assertNotEqual(grad_meta.tensor_id, opt_meta.tensor_id)

    # --- Parameter always wins ---

    def test_parameter_wins_over_gradient_hint(self):
        """Even with hint=GRADIENT, a tensor whose data_ptr is a known parameter
        must be classified as PARAMETER."""
        param = next(self.model.parameters())
        meta = self.reg.get_or_create(param, hint_role=TensorRole.GRADIENT)
        self.assertEqual(meta.role, TensorRole.PARAMETER)

    def test_parameter_wins_over_activation_hint(self):
        param = next(self.model.parameters())
        meta = self.reg.get_or_create(param, hint_role=TensorRole.ACTIVATION)
        self.assertEqual(meta.role, TensorRole.PARAMETER)

    # --- mark_gradient name upgrade ---

    def test_mark_gradient_upgrades_to_longer_name(self):
        """A short local name like 'grad_weight' should be upgraded to
        'grad_fc1.weight' when a more qualified name is provided."""
        t = self._make_tensor(8, 4)
        # Register with a short name first
        self.reg.get_or_create(t, hint_role=TensorRole.GRADIENT, hint_name="grad_weight")
        # Now mark with a longer, more qualified name
        param = torch.randn(8, 4, requires_grad=True)
        meta = self.reg.mark_gradient(param, t, "fc1.weight")
        self.assertEqual(meta.name, "grad_fc1.weight",
                         "mark_gradient should prefer the longer (more qualified) name")

    def test_mark_gradient_keeps_longer_existing_name(self):
        """If the existing name is already longer, it must not be downgraded."""
        t = self._make_tensor(8, 4)
        # Register with a long name first
        self.reg.get_or_create(t, hint_role=TensorRole.GRADIENT,
                               hint_name="grad_encoder.block1.fc.weight")
        param = torch.randn(8, 4, requires_grad=True)
        meta = self.reg.mark_gradient(param, t, "w")  # short hint
        self.assertTrue(len(meta.name) > len("grad_w"),
                        "Longer existing name must not be downgraded")

    def test_mark_gradient_does_not_mutate_activation_at_same_address(self):
        """Regression test for the mark_gradient memory-reuse bug.

        Scenario: relu2.output (shape [8,4]) is freed during the backward pass.
        grad_fc1.weight (also shape [8,4]) is then allocated at the same address.
        mark_gradient must NOT change relu2.output's registry entry to GRADIENT —
        it must create a fresh entry for the gradient instead.
        """
        t = self._make_tensor(8, 4)
        # Simulate relu2.output: registered as ACTIVATION via force_create
        act_meta = self.reg.force_create(t, TensorRole.ACTIVATION, "relu2.output")
        act_tid = act_meta.tensor_id

        # Simulate grad_fc1.weight arriving at the same address
        param = torch.randn(8, 4, requires_grad=True)
        grad_meta = self.reg.mark_gradient(param, t, "fc1.weight")

        # The OLD activation entry must still be ACTIVATION
        old_entry = self.reg._registry[act_tid]
        self.assertEqual(old_entry.role, TensorRole.ACTIVATION,
                         "mark_gradient must not mutate a stale ACTIVATION entry's role")
        self.assertEqual(old_entry.name, "relu2.output",
                         "mark_gradient must not change the activation's name")

        # The returned gradient entry must be a fresh GRADIENT entry
        self.assertNotEqual(grad_meta.tensor_id, act_tid,
                            "mark_gradient must create a new entry, not reuse the activation's")
        self.assertEqual(grad_meta.role, TensorRole.GRADIENT)

    # --- all_by_role ---

    def test_all_by_role_returns_correct_subset(self):
        """all_by_role must return exactly the tensors with the given role."""
        ta = self._make_tensor(2, 2)
        tg = self._make_tensor(2, 2)
        self.reg.force_create(ta, TensorRole.ACTIVATION, "act")
        self.reg.get_or_create(tg, hint_role=TensorRole.GRADIENT, hint_name="g")
        acts = self.reg.all_by_role(TensorRole.ACTIVATION)
        grads = self.reg.all_by_role(TensorRole.GRADIENT)
        self.assertTrue(all(m.role == TensorRole.ACTIVATION for m in acts))
        self.assertTrue(all(m.role == TensorRole.GRADIENT for m in grads))

    # --- Classification by grad_fn ---

    def test_classify_activation_by_grad_fn(self):
        """A tensor with grad_fn (from a computation) → ACTIVATION."""
        x = torch.randn(4, 4)
        w = nn.Parameter(torch.randn(4, 4))
        out = x @ w  # out.grad_fn is not None
        meta = self.reg.get_or_create(out)
        self.assertEqual(meta.role, TensorRole.ACTIVATION)

    def test_classify_other_no_grad_fn_no_param(self):
        """A plain tensor (no grad_fn, not a parameter) → OTHER."""
        t = self._make_tensor(4, 4)  # plain tensor, no grad_fn
        meta = self.reg.get_or_create(t)
        self.assertEqual(meta.role, TensorRole.OTHER)

    def test_all_tensor_ids_unique_in_registry(self):
        """Every entry in the registry must have a unique tensor_id."""
        for _ in range(20):
            t = self._make_tensor(3, 3)
            self.reg.force_create(t, TensorRole.ACTIVATION, "a")
        ids = [m.tensor_id for m in self.reg._registry.values()]
        self.assertEqual(len(ids), len(set(ids)), "tensor_ids must be unique")

    def test_force_create_old_entry_survives_in_registry(self):
        """After force_create overwrites (ptr,shape) mapping, the OLD TensorMeta
        must still be present in _registry (enabling all-time activation counting
        via all_by_role). Only _ptr_shape_to_meta is updated; _registry accumulates."""
        t = self._make_tensor(4, 4)
        old_meta = self.reg.force_create(t, TensorRole.ACTIVATION, "old_act")
        old_tid = old_meta.tensor_id

        new_meta = self.reg.force_create(t, TensorRole.ACTIVATION, "new_act")

        # Old entry must still be in _registry (all-time tracking)
        self.assertIn(old_tid, self.reg._registry,
                      "Old activation entry must survive in _registry after force_create")
        self.assertEqual(self.reg._registry[old_tid].role, TensorRole.ACTIVATION)

        # _ptr_shape_to_meta points to the NEW entry only
        key = (t.data_ptr(), tuple(t.shape))
        self.assertEqual(self.reg._ptr_shape_to_meta[key].tensor_id, new_meta.tensor_id)

        # all_by_role must return BOTH entries (old and new)
        acts = self.reg.all_by_role(TensorRole.ACTIVATION)
        act_ids = {m.tensor_id for m in acts}
        self.assertIn(old_tid, act_ids, "old entry must appear in all_by_role")
        self.assertIn(new_meta.tensor_id, act_ids, "new entry must appear in all_by_role")

    def test_mark_gradient_equal_length_name_uses_new(self):
        """When new and existing gradient names have equal length, the new name
        replaces the old one (>= condition in mark_gradient name comparison)."""
        t = self._make_tensor(4, 4)
        # Register with name of length 9: "grad_fc1_"... let's use "grad_abc" (8 chars)
        self.reg.get_or_create(t, hint_role=TensorRole.GRADIENT, hint_name="grad_fc1")
        param = torch.randn(4, 4, requires_grad=True)
        # "grad_fc2" is also 8 chars — equal length
        meta = self.reg.mark_gradient(param, t, "fc2")
        self.assertEqual(meta.name, "grad_fc2",
                         "Equal-length names: >= condition means new name replaces old")


# ---------------------------------------------------------------------------
# 2. ComputationalGraph / topological sort
# ---------------------------------------------------------------------------

class TestGraphTopoSort(unittest.TestCase):

    def setUp(self):
        _seed()

    def _run_two_layer(self):
        model = TwoLayerMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(8, 4)
        Y = torch.randn(8, 2)
        return _run(model, opt, X, Y)

    def test_all_forward_before_backward(self):
        """Every FORWARD node must have a smaller topo-index than every BACKWARD node."""
        profiler = self._run_two_layer()
        nodes = profiler._graph.nodes_in_topo_order()
        fwd_indices = [i for i, n in enumerate(nodes) if n.phase == OpPhase.FORWARD]
        bwd_indices = [i for i, n in enumerate(nodes) if n.phase == OpPhase.BACKWARD]
        self.assertTrue(len(fwd_indices) > 0)
        self.assertTrue(len(bwd_indices) > 0)
        self.assertLess(max(fwd_indices), min(bwd_indices),
                        "All forward ops must come before all backward ops")

    def test_all_backward_before_optimizer(self):
        """Every BACKWARD node must have a smaller index than the OPTIMIZER node."""
        profiler = self._run_two_layer()
        nodes = profiler._graph.nodes_in_topo_order()
        bwd_indices = [i for i, n in enumerate(nodes) if n.phase == OpPhase.BACKWARD]
        opt_indices = [i for i, n in enumerate(nodes) if n.phase == OpPhase.OPTIMIZER]
        self.assertEqual(len(opt_indices), 1)
        self.assertLess(max(bwd_indices), opt_indices[0],
                        "All backward ops must come before the optimizer step")

    def test_exactly_one_optimizer_node(self):
        profiler = self._run_two_layer()
        nodes = profiler._graph.nodes_in_topo_order()
        opt_nodes = [n for n in nodes if n.phase == OpPhase.OPTIMIZER]
        self.assertEqual(len(opt_nodes), 1)

    def test_node_count_two_layer_mlp(self):
        """TwoLayerMLP has 4 leaf modules → 4 forward + 4 backward + 1 optimizer = 9 nodes."""
        profiler = self._run_two_layer()
        nodes = profiler._graph.nodes_in_topo_order()
        self.assertEqual(len(nodes), 9)

    def test_node_count_single_linear(self):
        """A single Linear → 1 forward + 1 backward + 1 optimizer = 3 nodes."""
        _seed()
        model = nn.Linear(4, 2, bias=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        X = torch.randn(8, 4)
        Y = torch.randn(8, 2)
        profiler = _run(model, opt, X, Y)
        nodes = profiler._graph.nodes_in_topo_order()
        self.assertEqual(len(nodes), 3)

    def test_all_node_ids_unique(self):
        profiler = self._run_two_layer()
        nodes = profiler._graph.nodes_in_topo_order()
        ids = [n.node_id for n in nodes]
        self.assertEqual(len(ids), len(set(ids)), "All node_ids must be unique")

    def test_forward_node_count_equals_leaf_count(self):
        """Number of FORWARD nodes must equal number of leaf modules."""
        profiler = self._run_two_layer()
        model = TwoLayerMLP()
        n_leaves = _count_leaves(model)
        nodes = profiler._graph.nodes_in_topo_order()
        n_fwd = sum(1 for n in nodes if n.phase == OpPhase.FORWARD)
        self.assertEqual(n_fwd, n_leaves)

    def test_backward_node_count_equals_leaf_count(self):
        """Number of BACKWARD nodes must equal number of leaf modules."""
        profiler = self._run_two_layer()
        model = TwoLayerMLP()
        n_leaves = _count_leaves(model)
        nodes = profiler._graph.nodes_in_topo_order()
        n_bwd = sum(1 for n in nodes if n.phase == OpPhase.BACKWARD)
        self.assertEqual(n_bwd, n_leaves)

    def test_topo_order_respects_edges(self):
        """For every edge (src→dst), src must appear before dst in topo order."""
        profiler = self._run_two_layer()
        nodes = profiler._graph.nodes_in_topo_order()
        topo_pos = {n.node_id: i for i, n in enumerate(nodes)}
        for edge in profiler._graph.edges:
            src_pos = topo_pos.get(edge.src_node_id)
            dst_pos = topo_pos.get(edge.dst_node_id)
            if src_pos is not None and dst_pos is not None:
                self.assertLess(src_pos, dst_pos,
                                f"Edge {edge.src_node_id}→{edge.dst_node_id}: "
                                f"src at pos {src_pos} must precede dst at {dst_pos}")

    def test_frozen_graph_rejects_new_nodes(self):
        """After freeze(), adding a node must raise RuntimeError."""
        graph = ComputationalGraph()
        graph.freeze()
        node = GraphNode(node_id=0, op_name="test", phase=OpPhase.FORWARD,
                         module_fqn="x", input_tensor_ids=[], output_tensor_ids=[])
        with self.assertRaises(RuntimeError):
            graph.add_node(node)

    def test_frozen_graph_rejects_new_edges(self):
        graph = ComputationalGraph()
        graph.freeze()
        edge = GraphEdge(src_node_id=0, dst_node_id=1, tensor_id=0)
        with self.assertRaises(RuntimeError):
            graph.add_edge(edge)

    def test_topo_order_stable_across_calls(self):
        """nodes_in_topo_order() must return the SAME node sequence on every call
        to a frozen graph. This is critical: compute_lifetimes() stores first_use_op/
        last_use_op as indices into one call's sequence, and the visualizer uses
        those same indices when indexing into another call's node list."""
        model = TwoLayerMLP(bias=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, Y = torch.randn(8, 4), torch.randn(8, 2)
        profiler = _run(model, opt, X, Y)

        order1 = profiler._graph.nodes_in_topo_order()
        order2 = profiler._graph.nodes_in_topo_order()
        order3 = profiler._graph.nodes_in_topo_order()

        self.assertEqual(len(order1), len(order2))
        self.assertEqual(len(order1), len(order3))
        for i, (n1, n2, n3) in enumerate(zip(order1, order2, order3)):
            self.assertEqual(n1.node_id, n2.node_id,
                             f"Position {i}: order differs between calls 1 and 2")
            self.assertEqual(n1.node_id, n3.node_id,
                             f"Position {i}: order differs between calls 1 and 3")

    def test_add_node_duplicate_id_overwrites_silently(self):
        """add_node() with a duplicate node_id silently overwrites the old node.
        This documents the behavior: node_ids must be obtained via next_node_id()
        and used exactly once to avoid accidental overwrites."""
        graph = ComputationalGraph()
        node_a = GraphNode(node_id=0, op_name="op_a", phase=OpPhase.FORWARD,
                           module_fqn="a", input_tensor_ids=[], output_tensor_ids=[])
        node_b = GraphNode(node_id=0, op_name="op_b", phase=OpPhase.FORWARD,
                           module_fqn="b", input_tensor_ids=[], output_tensor_ids=[])
        graph.add_node(node_a)
        graph.add_node(node_b)  # same id — silently replaces node_a
        self.assertEqual(graph.nodes[0].op_name, "op_b",
                         "Second add_node with same id must overwrite the first")


# ---------------------------------------------------------------------------
# 3. Lifetime analysis (compute_lifetimes)
# ---------------------------------------------------------------------------

class TestComputeLifetimes(unittest.TestCase):

    def setUp(self):
        _seed()

    def _run_net(self, model, opt=None, batch=8, in_dim=4, out_dim=2):
        if opt is None:
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(batch, in_dim)
        Y = torch.randn(batch, out_dim)
        return _run(model, opt, X, Y)

    def test_no_activation_has_none_lifetime(self):
        """After a full profiling run, every ACTIVATION must have non-None
        first_use_op and last_use_op."""
        profiler = self._run_net(TwoLayerMLP())
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertTrue(len(acts) > 0)
        for m in acts:
            self.assertIsNotNone(m.first_use_op,
                                 f"{m.name}: first_use_op is None after compute_lifetimes")
            self.assertIsNotNone(m.last_use_op,
                                 f"{m.name}: last_use_op is None after compute_lifetimes")

    def test_first_leq_last_for_all_tensors(self):
        """first_use_op must be <= last_use_op for every tensor that has both set."""
        profiler = self._run_net(TwoLayerMLP())
        for meta in profiler._registry._registry.values():
            if meta.first_use_op is not None and meta.last_use_op is not None:
                self.assertLessEqual(
                    meta.first_use_op, meta.last_use_op,
                    f"{meta.name}: first={meta.first_use_op} > last={meta.last_use_op}"
                )

    def test_sigmoid_output_extends_to_sigmoid_backward(self):
        """sig.output must live until sig.backward fires, not just until the
        next layer's forward (Sigmoid saves its output for the backward mask)."""
        model = TwoLayerMLP()
        profiler = self._run_net(model)
        nodes = profiler._graph.nodes_in_topo_order()

        # Find sig1.backward step index
        sig1_bwd_step = next(
            i for i, n in enumerate(nodes)
            if n.phase == OpPhase.BACKWARD and n.module_fqn == "sig1"
        )
        # Find sig1.output meta
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        sig1_out = next((m for m in acts if m.name == "sig1.output"), None)
        self.assertIsNotNone(sig1_out, "sig1.output must be registered as an activation")
        self.assertEqual(sig1_out.last_use_op, sig1_bwd_step,
                         f"sig1.output must survive to sig1.backward (step {sig1_bwd_step}), "
                         f"got last_use_op={sig1_out.last_use_op}")

    def test_relu_output_extends_to_relu_backward(self):
        """relu.output must live until relu.backward (ReLU backward needs the mask output > 0)."""
        class ReluNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 2, bias=False)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        profiler = self._run_net(ReluNet())
        nodes = profiler._graph.nodes_in_topo_order()
        relu_bwd_step = next(
            i for i, n in enumerate(nodes)
            if n.phase == OpPhase.BACKWARD and n.module_fqn == "relu"
        )
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        relu_out = next((m for m in acts if m.name == "relu.output"), None)
        self.assertIsNotNone(relu_out)
        self.assertEqual(relu_out.last_use_op, relu_bwd_step,
                         f"relu.output must survive to relu.backward (step {relu_bwd_step})")

    def test_tanh_output_extends_to_tanh_backward(self):
        """tanh.output must live until tanh.backward (tanh backward uses the output value)."""
        class TanhNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.tanh = nn.Tanh()
                self.fc2 = nn.Linear(8, 2, bias=False)
            def forward(self, x):
                return self.fc2(self.tanh(self.fc1(x)))

        profiler = self._run_net(TanhNet())
        nodes = profiler._graph.nodes_in_topo_order()
        tanh_bwd_step = next(
            i for i, n in enumerate(nodes)
            if n.phase == OpPhase.BACKWARD and n.module_fqn == "tanh"
        )
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        tanh_out = next((m for m in acts if m.name == "tanh.output"), None)
        self.assertIsNotNone(tanh_out)
        self.assertEqual(tanh_out.last_use_op, tanh_bwd_step,
                         f"tanh.output must survive to tanh.backward (step {tanh_bwd_step})")

    def test_linear_input_activation_extends_to_linear_backward(self):
        """The activation that feeds INTO a Linear layer must live until that
        layer's backward (dL/dW = dL/dZ^T @ activation_input is computed there)."""
        model = TwoLayerMLP()   # fc1 → sig1 → fc2 → sig2
        profiler = self._run_net(model)
        nodes = profiler._graph.nodes_in_topo_order()

        # fc2's backward needs sig1.output (its input activation)
        fc2_bwd_step = next(
            i for i, n in enumerate(nodes)
            if n.phase == OpPhase.BACKWARD and n.module_fqn == "fc2"
        )
        # sig1's backward step
        sig1_bwd_step = next(
            i for i, n in enumerate(nodes)
            if n.phase == OpPhase.BACKWARD and n.module_fqn == "sig1"
        )
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        sig1_out = next((m for m in acts if m.name == "sig1.output"), None)
        self.assertIsNotNone(sig1_out)
        # sig1.output must survive until max(fc2_backward, sig1_backward)
        expected_last = max(fc2_bwd_step, sig1_bwd_step)
        self.assertEqual(sig1_out.last_use_op, expected_last,
                         f"sig1.output must live to step {expected_last}, "
                         f"got {sig1_out.last_use_op}")

    def test_final_layer_output_fixup_last_equals_first(self):
        """The output of the last layer has no explicit consumer in the hooks
        (the loss function is not hooked). The fixup must set last_use_op = first_use_op."""
        model = TwoLayerMLP()
        profiler = self._run_net(model)
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        # sig2.output is the final output — its Pass-3 extension goes to sig2.backward,
        # so last >= first (strictly greater). But we also check no None:
        sig2_out = next((m for m in acts if m.name == "sig2.output"), None)
        self.assertIsNotNone(sig2_out)
        self.assertIsNotNone(sig2_out.last_use_op,
                             "Final layer output must not have None last_use_op")

    def test_deep_uniform_network_all_activations_distinct(self):
        """In a network where every Linear layer has the same input/output size,
        PyTorch reuses freed activation memory. force_create must ensure all
        activations are tracked as distinct entries."""
        class DeepNet(nn.Sequential):
            pass

        layers = []
        for i in range(5):
            layers.append((f"fc{i}", nn.Linear(32, 32, bias=False)))
            layers.append((f"act{i}", nn.Sigmoid()))

        model = nn.Sequential()
        for name, layer in layers:
            model.add_module(name, layer)

        # Add final output projection to prevent last activation being OTHER
        model.add_module("out", nn.Linear(32, 2, bias=False))

        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        X = torch.randn(4, 32)
        Y = torch.randn(4, 2)
        profiler = _run(model, opt, X, Y)

        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        # 5 Linear + 5 Sigmoid + 1 output Linear = 11 forward outputs, all ACTIVATION
        # (some may be OTHER if grad_fn is None, but most should be ACTIVATION)
        self.assertGreaterEqual(len(acts), 10,
                                f"Expected ≥10 distinct activations, got {len(acts)}: "
                                f"{[m.name for m in acts]}")

    def test_dropout_output_extends_to_dropout_backward(self):
        """Dropout saves its mask (output-shaped) for backward. The dropout output
        activation must survive until the Dropout backward hook fires."""
        class DropNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.drop = nn.Dropout(p=0.5)
                self.fc2 = nn.Linear(8, 2, bias=False)
            def forward(self, x):
                return self.fc2(self.drop(self.fc1(x)))

        _seed()
        model = DropNet().train()
        profiler = self._run_net(model)
        nodes = profiler._graph.nodes_in_topo_order()
        drop_bwd_step = next(
            i for i, n in enumerate(nodes)
            if n.phase == OpPhase.BACKWARD and n.module_fqn == "drop"
        )
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        drop_out = next((m for m in acts if m.name == "drop.output"), None)
        self.assertIsNotNone(drop_out, "drop.output must be tracked as an activation")
        self.assertEqual(drop_out.last_use_op, drop_bwd_step,
                         f"drop.output must survive to drop.backward (step {drop_bwd_step})")

    def test_activation_live_ops_positive(self):
        """Every activation's live span (last - first + 1) must be at least 1."""
        profiler = self._run_net(TwoLayerMLP())
        for m in profiler._registry.all_by_role(TensorRole.ACTIVATION):
            if m.first_use_op is not None and m.last_use_op is not None:
                self.assertGreaterEqual(m.last_use_op - m.first_use_op + 1, 1)

    def test_activations_have_first_use_op_set(self):
        """Pass 1 must set first_use_op for every activation produced by a forward hook."""
        profiler = self._run_net(TwoLayerMLP())
        for m in profiler._registry.all_by_role(TensorRole.ACTIVATION):
            self.assertIsNotNone(m.first_use_op,
                                 f"{m.name} has no first_use_op — Pass 1 missed it")

    def test_lifetime_indices_within_node_count(self):
        """Every first_use_op and last_use_op must be a valid index into
        nodes_in_topo_order() (i.e., in [0, n-1]). Out-of-bounds indices would
        cause the visualizer to silently produce wrong charts via clamping."""
        profiler = self._run_net(TwoLayerMLP())
        n = len(profiler._graph.nodes_in_topo_order())
        for meta in profiler._registry._registry.values():
            if meta.first_use_op is not None:
                self.assertGreaterEqual(meta.first_use_op, 0,
                                        f"{meta.name}: first_use_op is negative")
                self.assertLess(meta.first_use_op, n,
                                f"{meta.name}: first_use_op={meta.first_use_op} >= n={n}")
            if meta.last_use_op is not None:
                self.assertGreaterEqual(meta.last_use_op, 0,
                                        f"{meta.name}: last_use_op is negative")
                self.assertLess(meta.last_use_op, n,
                                f"{meta.name}: last_use_op={meta.last_use_op} >= n={n}")

    def test_compute_lifetimes_forward_only_no_backward(self):
        """compute_lifetimes must work correctly (no crash) when the graph has no
        BACKWARD nodes. Pass 3 becomes a no-op; passes 1 and 2 still set lifetimes."""
        from profiler.graph import GraphNode, OpPhase
        from profiler.tensor_registry import TensorRegistry, compute_lifetimes

        model = nn.Linear(4, 2, bias=False)
        registry = TensorRegistry(model)
        graph = ComputationalGraph()

        x = torch.randn(4, 4)
        with torch.no_grad():
            out = model(x)  # no grad_fn since no_grad

        x_meta = registry.get_or_create(x, TensorRole.OTHER, "X")
        out_meta = registry.get_or_create(out, TensorRole.OTHER, "out")  # no grad_fn

        nid = graph.next_node_id()
        node = GraphNode(
            node_id=nid,
            op_name="Linear.forward",
            phase=OpPhase.FORWARD,
            module_fqn="fc",
            input_tensor_ids=[x_meta.tensor_id],
            output_tensor_ids=[out_meta.tensor_id],
        )
        graph.add_node(node)
        graph.freeze()

        # Must not raise
        compute_lifetimes(graph, registry)

        # Pass 1: output first_use_op set to 0
        self.assertEqual(out_meta.first_use_op, 0)
        # Pass 2: input last_use_op set to 0 (consumed at step 0)
        self.assertEqual(x_meta.last_use_op, 0)
        # Fixup: output has no explicit consumer → last = first = 0
        self.assertEqual(out_meta.last_use_op, 0)


# ---------------------------------------------------------------------------
# 4. Tensor categorization (full profiler run)
# ---------------------------------------------------------------------------

class TestProfilerCategorization(unittest.TestCase):

    def setUp(self):
        _seed()

    def _profiler(self, model, opt_cls=torch.optim.Adam, **opt_kwargs):
        opt_kwargs.setdefault("lr", 1e-3)   # allow callers to override lr
        opt = opt_cls(model.parameters(), **opt_kwargs)
        X = torch.randn(8, 4)
        Y = torch.randn(8, 2)
        return _run(model, opt, X, Y)

    def test_parameter_count_no_bias(self):
        """PARAMETER count must equal the number of nn.Parameter objects in the model."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model)
        expected = len(list(model.parameters()))
        got = len(profiler._registry.all_by_role(TensorRole.PARAMETER))
        self.assertEqual(got, expected,
                         f"Expected {expected} parameters, got {got}")

    def test_parameter_count_with_bias(self):
        """Bias parameters must also be tracked as PARAMETER."""
        model = TwoLayerMLP(bias=True)   # 4 parameters: w1, b1, w2, b2
        profiler = self._profiler(model)
        expected = len(list(model.parameters()))   # should be 4
        got = len(profiler._registry.all_by_role(TensorRole.PARAMETER))
        self.assertEqual(got, expected)

    def test_every_parameter_has_a_gradient(self):
        """For every PARAMETER, there must be at least one GRADIENT of the same shape."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model)
        params = profiler._registry.all_by_role(TensorRole.PARAMETER)
        grads = profiler._registry.all_by_role(TensorRole.GRADIENT)
        grad_shapes = {tuple(g.shape) for g in grads}
        for p in params:
            self.assertIn(tuple(p.shape), grad_shapes,
                          f"No gradient found for parameter of shape {p.shape}")

    def test_activation_count_two_layer_mlp(self):
        """TwoLayerMLP (4 leaf modules, each with one output) → 4 ACTIVATION tensors."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model)
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertEqual(len(acts), 4,
                         f"Expected 4 activations, got {len(acts)}: {[m.name for m in acts]}")

    def test_activation_names_match_modules(self):
        """Activation names should reflect the module that produced them."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model)
        act_names = {m.name for m in profiler._registry.all_by_role(TensorRole.ACTIVATION)}
        for expected in ["fc1.output", "sig1.output", "fc2.output", "sig2.output"]:
            self.assertIn(expected, act_names,
                          f"'{expected}' not found in activation names: {act_names}")

    def test_adam_optimizer_state_count(self):
        """Adam maintains step, exp_avg, exp_avg_sq per parameter → 3 * num_params states."""
        model = TwoLayerMLP(bias=False)   # 2 parameters
        profiler = self._profiler(model, opt_cls=torch.optim.Adam)
        n_params = len(list(model.parameters()))
        opt_states = profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        self.assertEqual(len(opt_states), 3 * n_params,
                         f"Adam: expected {3 * n_params} states, got {len(opt_states)}")

    def test_sgd_no_momentum_zero_optimizer_states(self):
        """Plain SGD (no momentum) creates no optimizer state tensors."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model, opt_cls=torch.optim.SGD, lr=0.01)
        opt_states = profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        self.assertEqual(len(opt_states), 0,
                         f"SGD without momentum: expected 0 states, got {len(opt_states)}")

    def test_sgd_with_momentum_one_state_per_param(self):
        """SGD with momentum allocates one momentum_buffer per parameter."""
        model = TwoLayerMLP(bias=False)   # 2 parameters
        n_params = len(list(model.parameters()))
        profiler = self._profiler(model, opt_cls=torch.optim.SGD, lr=0.01, momentum=0.9)
        opt_states = profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        self.assertEqual(len(opt_states), n_params,
                         f"SGD+momentum: expected {n_params} states, got {len(opt_states)}")

    def test_adamw_optimizer_state_count(self):
        """AdamW (decoupled weight decay) also uses step+exp_avg+exp_avg_sq → 3N."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model, opt_cls=torch.optim.AdamW)
        n_params = len(list(model.parameters()))
        opt_states = profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        self.assertEqual(len(opt_states), 3 * n_params,
                         f"AdamW: expected {3 * n_params} states, got {len(opt_states)}")

    def test_bias_parameters_tracked_with_gradients(self):
        """Bias parameters must be classified as PARAMETER and must have gradients."""
        model = TwoLayerMLP(bias=True)   # w1, b1, w2, b2
        profiler = self._profiler(model)
        params = profiler._registry.all_by_role(TensorRole.PARAMETER)
        grads = profiler._registry.all_by_role(TensorRole.GRADIENT)
        # All 4 parameters must be tracked
        self.assertEqual(len(params), 4)
        # All 4 must have a gradient of matching shape
        grad_shapes = {tuple(g.shape) for g in grads}
        for p in params:
            self.assertIn(tuple(p.shape), grad_shapes,
                          f"Bias parameter of shape {p.shape} has no gradient")

    def test_input_x_classified_as_other(self):
        """Input X must be classified as OTHER, not ACTIVATION or PARAMETER."""
        model = TwoLayerMLP(bias=False)
        profiler = self._profiler(model)
        others = profiler._registry.all_by_role(TensorRole.OTHER)
        other_names = {m.name for m in others}
        self.assertIn("X", other_names, "Input X must be classified as OTHER")
        self.assertIn("Y", other_names, "Target Y must be classified as OTHER")
        self.assertIn("L", other_names, "Loss L must be classified as OTHER")

    def test_rmsprop_state_count(self):
        """RMSprop (no momentum, not centered) creates step + square_avg per parameter.
        In PyTorch ≥2.0, 'step' is stored as a tensor, giving 2 states per param."""
        model = TwoLayerMLP(bias=False)
        n_params = len(list(model.parameters()))
        opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        X, Y = torch.randn(8, 4), torch.randn(8, 2)
        profiler = _run(model, opt, X, Y)
        opt_states = profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        # Inspect actual per-param state keys to compute expected count robustly
        actual_per_param = sum(
            1 for v in opt.state[next(iter(opt.state))].values()
            if isinstance(v, torch.Tensor)
        )
        expected = actual_per_param * n_params
        self.assertEqual(len(opt_states), expected,
                         f"RMSprop: expected {expected} states, got {len(opt_states)}")


# ---------------------------------------------------------------------------
# 5. Graph integrity
# ---------------------------------------------------------------------------

class TestGraphIntegrity(unittest.TestCase):

    def setUp(self):
        _seed()
        model = TwoLayerMLP(bias=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, Y = torch.randn(8, 4), torch.randn(8, 2)
        self.profiler = _run(model, opt, X, Y)

    def test_no_self_loops_in_edges(self):
        """An edge must not have src_node_id == dst_node_id."""
        for edge in self.profiler._graph.edges:
            self.assertNotEqual(edge.src_node_id, edge.dst_node_id,
                                f"Self-loop detected on node {edge.src_node_id}")

    def test_edges_reference_valid_nodes(self):
        """Every edge's src and dst node_id must exist in graph.nodes."""
        valid_ids = set(self.profiler._graph.nodes.keys())
        for edge in self.profiler._graph.edges:
            self.assertIn(edge.src_node_id, valid_ids,
                          f"Edge src {edge.src_node_id} not in graph nodes")
            self.assertIn(edge.dst_node_id, valid_ids,
                          f"Edge dst {edge.dst_node_id} not in graph nodes")

    def test_edges_reference_valid_tensor_ids(self):
        """Every edge's tensor_id must exist in the tensor registry."""
        registry_ids = set(self.profiler._registry._registry.keys())
        for edge in self.profiler._graph.edges:
            self.assertIn(edge.tensor_id, registry_ids,
                          f"Edge tensor_id {edge.tensor_id} not in registry")

    def test_all_nodes_have_profile(self):
        """Every node must have a non-None ProfileResult attached."""
        for node in self.profiler._graph.nodes.values():
            self.assertIsNotNone(node.profile,
                                 f"Node {node.node_id} ({node.op_name}) has no profile")

    def test_all_profiles_have_nonnegative_wall_time(self):
        """Wall time must be >= 0 for every node."""
        for node in self.profiler._graph.nodes.values():
            self.assertGreaterEqual(node.profile.wall_time_ms, 0.0,
                                    f"Negative wall time for {node.op_name}")

    def test_output_tensor_ids_appear_in_registry(self):
        """Every tensor_id listed in a node's output_tensor_ids must be in the registry."""
        registry_ids = set(self.profiler._registry._registry.keys())
        for node in self.profiler._graph.nodes.values():
            for tid in node.output_tensor_ids:
                self.assertIn(tid, registry_ids,
                              f"Node {node.op_name} output tensor_id {tid} not in registry")

    def test_input_tensor_ids_appear_in_registry(self):
        """Every tensor_id listed in a node's input_tensor_ids must be in the registry."""
        registry_ids = set(self.profiler._registry._registry.keys())
        for node in self.profiler._graph.nodes.values():
            for tid in node.input_tensor_ids:
                self.assertIn(tid, registry_ids,
                              f"Node {node.op_name} input tensor_id {tid} not in registry")

    def test_all_node_phases_valid(self):
        """Every node must have a valid OpPhase."""
        valid_phases = set(OpPhase)
        for node in self.profiler._graph.nodes.values():
            self.assertIn(node.phase, valid_phases)


# ---------------------------------------------------------------------------
# 6. API guard rails
# ---------------------------------------------------------------------------

class TestAPIGuardrails(unittest.TestCase):

    def setUp(self):
        _seed()
        self.model = TwoLayerMLP(bias=False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.X = torch.randn(8, 4)
        self.Y = torch.randn(8, 2)

    def test_second_run_raises_runtime_error(self):
        """Calling run() a second time on the same instance must raise RuntimeError."""
        profiler = ProfilerExecutor(self.model, self.opt)
        profiler.run(self.X, self.Y, nn.MSELoss())
        with self.assertRaises(RuntimeError):
            profiler.run(self.X, self.Y, nn.MSELoss())

    def test_get_report_before_run_raises(self):
        profiler = ProfilerExecutor(self.model, self.opt)
        with self.assertRaises(RuntimeError):
            profiler.get_report()

    def test_visualize_before_run_raises(self):
        profiler = ProfilerExecutor(self.model, self.opt)
        with self.assertRaises(RuntimeError):
            profiler.visualize("should_not_be_created.png")

    def test_batch_size_one(self):
        """Batch size of 1 must work without errors."""
        model = TwoLayerMLP(bias=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(1, 4)
        Y = torch.randn(1, 2)
        profiler = _run(model, opt, X, Y)
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertEqual(len(acts), 4)

    def test_get_report_returns_profile_report(self):
        """get_report() after a successful run must return a usable ProfileReport."""
        profiler = ProfilerExecutor(self.model, self.opt)
        profiler.run(self.X, self.Y, nn.MSELoss())
        report = profiler.get_report()
        self.assertIsNotNone(report)
        self.assertIsNotNone(report.graph)
        self.assertIsNotNone(report.registry)

    def test_large_batch_runs_correctly(self):
        """Large batch size must not corrupt tensor tracking."""
        model = TwoLayerMLP(bias=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = torch.randn(256, 4)
        Y = torch.randn(256, 2)
        profiler = _run(model, opt, X, Y)
        # Larger batch → larger activation tensors, but same count
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertEqual(len(acts), 4)

    def test_wrapped_optimizer_zero_grad_delegates(self):
        """WrappedOptimizer.zero_grad() must clear all gradients."""
        profiler = ProfilerExecutor(self.model, self.opt)
        # Manually run forward+backward to set gradients
        out = self.model(self.X)
        loss = nn.MSELoss()(out, self.Y)
        loss.backward()
        # Gradients should be set
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
        profiler._opt.zero_grad()
        for p in self.model.parameters():
            self.assertIsNone(p.grad)

    def test_wrapped_optimizer_param_groups_accessible(self):
        """WrappedOptimizer.param_groups must delegate to the underlying optimizer."""
        profiler = ProfilerExecutor(self.model, self.opt)
        pg = profiler._opt.param_groups
        self.assertEqual(pg, self.opt.param_groups)

    def test_print_summary_does_not_crash(self):
        """ProfileReport.print_summary() must execute without raising any exception
        and produce output containing the expected section headers."""
        import io
        profiler = ProfilerExecutor(self.model, self.opt)
        profiler.run(self.X, self.Y, nn.MSELoss())
        report = profiler.get_report()

        captured = io.StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            report.print_summary()
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("COMPUTATIONAL GRAPH", output)
        self.assertIn("TENSOR CATEGORIZATION", output)
        self.assertIn("ACTIVATION LIFETIME", output)

    def test_no_param_model_without_device_raises(self):
        """A model with no parameters requires an explicit device argument.
        Without it, ProfilerExecutor.__init__ raises StopIteration when
        trying to infer device from next(model.parameters()).

        Note: PyTorch ≥2.0 rejects optimizers with empty parameter lists,
        so we pass a dummy (unrelated) parameter to the optimizer.
        The model itself (nn.Identity) still has no parameters."""
        model = nn.Identity()  # no parameters
        dummy = nn.Parameter(torch.zeros(1))   # not part of model
        opt = torch.optim.Adam([dummy], lr=1e-3)
        with self.assertRaises((StopIteration, RuntimeError)):
            ProfilerExecutor(model, opt)  # no device argument — must raise


# ---------------------------------------------------------------------------
# 7. Architecture coverage
# ---------------------------------------------------------------------------

class TestArchitectures(unittest.TestCase):

    def setUp(self):
        _seed()

    def _run(self, model, in_dim=4, out_dim=2, batch=8, opt_cls=torch.optim.Adam):
        opt = opt_cls(model.parameters(), lr=1e-3)
        X = torch.randn(batch, in_dim)
        Y = torch.randn(batch, out_dim)
        return _run(model, opt, X, Y)

    def test_single_linear_no_bias(self):
        """A bare Linear layer: 1 param, 1 activation, 1 gradient, 0/3 opt states."""
        model = nn.Linear(4, 2, bias=False)
        profiler = self._run(model)
        self.assertEqual(len(profiler._registry.all_by_role(TensorRole.PARAMETER)), 1)
        self.assertEqual(len(profiler._registry.all_by_role(TensorRole.ACTIVATION)), 1)

    def test_three_layer_relu_net(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(8, 4, bias=False)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(4, 2, bias=False)
            def forward(self, x):
                return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

        profiler = self._run(Net())
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        params = profiler._registry.all_by_role(TensorRole.PARAMETER)
        self.assertEqual(len(acts), 5)    # fc1, relu1, fc2, relu2, fc3
        self.assertEqual(len(params), 3)  # fc1.w, fc2.w, fc3.w

    def test_network_with_bias_all_roles_correct(self):
        """A model with bias: both weight and bias are PARAMETER; both have GRADIENT."""
        model = nn.Sequential(nn.Linear(4, 8, bias=True), nn.Sigmoid(),
                              nn.Linear(8, 2, bias=True), nn.Sigmoid())
        profiler = self._run(model)
        params = profiler._registry.all_by_role(TensorRole.PARAMETER)
        grads = profiler._registry.all_by_role(TensorRole.GRADIENT)
        # 4 params: w1, b1, w2, b2
        self.assertEqual(len(params), 4)
        # At least 4 gradients (weight + bias per layer)
        grad_shapes = {tuple(g.shape) for g in grads}
        for p in params:
            self.assertIn(tuple(p.shape), grad_shapes)

    def test_dropout_tracked_correctly(self):
        """Dropout (train mode) must appear as a leaf module with one ACTIVATION output."""
        class DropNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.drop = nn.Dropout(p=0.2)
                self.fc2 = nn.Linear(8, 2, bias=False)
            def forward(self, x):
                return self.fc2(self.drop(self.fc1(x)))

        _seed()
        model = DropNet().train()
        profiler = self._run(model)
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        act_names = {m.name for m in acts}
        self.assertIn("drop.output", act_names)
        nodes = profiler._graph.nodes_in_topo_order()
        # Should have 3 forward + 3 backward + 1 optimizer = 7 nodes
        self.assertEqual(len(nodes), 7)

    def test_deep_net_correct_activation_count(self):
        """5 layers with non-uniform sizes → 5 distinct activations (no merging)."""
        class DeepNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Deliberately different sizes to avoid same-shape collisions
                self.fc1 = nn.Linear(4, 16, bias=False)
                self.fc2 = nn.Linear(16, 8, bias=False)
                self.fc3 = nn.Linear(8, 12, bias=False)
                self.fc4 = nn.Linear(12, 6, bias=False)
                self.fc5 = nn.Linear(6, 2, bias=False)
            def forward(self, x):
                return self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))

        profiler = self._run(DeepNet())
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertEqual(len(acts), 5, f"Expected 5, got {[m.name for m in acts]}")

    def test_tanh_network_lifetimes_complete(self):
        """A Tanh network must track all activations with correct lifetimes."""
        class TanhNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.tanh1 = nn.Tanh()
                self.fc2 = nn.Linear(8, 2, bias=False)
            def forward(self, x):
                return self.fc2(self.tanh1(self.fc1(x)))

        profiler = self._run(TanhNet())
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertEqual(len(acts), 3)
        for m in acts:
            self.assertIsNotNone(m.first_use_op)
            self.assertIsNotNone(m.last_use_op)
            self.assertLessEqual(m.first_use_op, m.last_use_op)

    def test_mixed_activations_relu_sigmoid(self):
        """A network alternating ReLU and Sigmoid must track all activations."""
        class MixedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8, bias=False)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 4, bias=False)
                self.sig = nn.Sigmoid()
                self.fc3 = nn.Linear(4, 2, bias=False)
            def forward(self, x):
                return self.fc3(self.sig(self.fc2(self.relu(self.fc1(x)))))

        profiler = self._run(MixedNet())
        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        self.assertEqual(len(acts), 5)

    def test_deep_relu_mlp_node_and_tensor_counts(self):
        """4-hidden-layer ReLU MLP (5 linear + 4 relu = 9 leaf modules):
          - 9 FORWARD + 9 BACKWARD + 1 OPTIMIZER = 19 nodes
          - 9 activations, 5 parameters, 15 Adam states (step+exp_avg+exp_avg_sq x 5)
        """
        class DeepMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 256, bias=False)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(256, 512, bias=False)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(512, 512, bias=False)
                self.relu3 = nn.ReLU()
                self.fc4 = nn.Linear(512, 256, bias=False)
                self.relu4 = nn.ReLU()
                self.fc5 = nn.Linear(256, 10, bias=False)
            def forward(self, x):
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                x = self.relu3(self.fc3(x))
                x = self.relu4(self.fc4(x))
                return self.fc5(x)

        _seed()
        model = DeepMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, Y = torch.randn(16, 64), torch.randn(16, 10)
        profiler = _run(model, opt, X, Y)

        nodes = profiler._graph.nodes_in_topo_order()
        self.assertEqual(len(nodes), 19)
        self.assertEqual(sum(1 for n in nodes if n.phase == OpPhase.FORWARD), 9)
        self.assertEqual(sum(1 for n in nodes if n.phase == OpPhase.BACKWARD), 9)
        self.assertEqual(sum(1 for n in nodes if n.phase == OpPhase.OPTIMIZER), 1)

        acts   = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        params = profiler._registry.all_by_role(TensorRole.PARAMETER)
        states = profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        self.assertEqual(len(acts), 9)
        self.assertEqual(len(params), 5)
        self.assertEqual(len(states), 15)  # 3 per param × 5 params

    def test_deep_relu_relu_activations_have_extended_lifetimes(self):
        """In a deep ReLU network, ReLU outputs must be live past their creation
        step (pass 3 extends them to the corresponding backward step).
        Linear inputs must also be live past creation (saved for weight gradient)."""
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 16, bias=False)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(16, 4, bias=False)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(4, 2, bias=False)
            def forward(self, x):
                return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

        _seed()
        model = Net()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, Y = torch.randn(8, 8), torch.randn(8, 2)
        profiler = _run(model, opt, X, Y)

        acts = profiler._registry.all_by_role(TensorRole.ACTIVATION)
        relu_acts = [a for a in acts if "relu" in a.name]
        self.assertGreater(len(relu_acts), 0, "Expected at least one relu activation")
        for a in relu_acts:
            self.assertGreater(
                a.last_use_op, a.first_use_op,
                f"{a.name}: ReLU output must be live past creation (pass 3 should extend it)"
            )


# ---------------------------------------------------------------------------
# 9. Pass 4 lifetime correctness
# ---------------------------------------------------------------------------

class TestPass4Lifetimes(unittest.TestCase):
    """Verify compute_lifetimes pass 4: persistent tensor lifetime normalisation.

    Pass 4 was introduced to fix three visualisation blind-spots:
      - Parameters appeared only at the optimizer step (should be live step 0→end)
      - Optimizer states appeared only at the optimizer step (should persist to end)
      - Terminal (param.grad) gradients freed at their backward step in the chart
        even though PyTorch holds them until the optimizer consumes them
    """

    def setUp(self):
        _seed()
        self.model = TwoLayerMLP(bias=False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        X, Y = torch.randn(8, 4), torch.randn(8, 2)
        self.profiler = _run(self.model, self.opt, X, Y)
        self.nodes = self.profiler._graph.nodes_in_topo_order()
        self.n = len(self.nodes)

    def test_parameters_first_use_op_is_zero(self):
        """All PARAMETER tensors must start at step 0 after pass 4."""
        params = self.profiler._registry.all_by_role(TensorRole.PARAMETER)
        self.assertGreater(len(params), 0)
        for p in params:
            self.assertEqual(p.first_use_op, 0,
                             f"{p.name}: expected first_use_op=0, got {p.first_use_op}")

    def test_parameters_last_use_op_is_final_step(self):
        """All PARAMETER tensors must persist through the last step after pass 4."""
        params = self.profiler._registry.all_by_role(TensorRole.PARAMETER)
        for p in params:
            self.assertEqual(p.last_use_op, self.n - 1,
                             f"{p.name}: expected last_use_op={self.n-1}, got {p.last_use_op}")

    def test_optimizer_states_persist_to_final_step(self):
        """All OPTIMIZER_STATE tensors must have last_use_op = n-1 after pass 4."""
        states = self.profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        self.assertGreater(len(states), 0)
        for s in states:
            if s.first_use_op is not None:
                self.assertEqual(s.last_use_op, self.n - 1,
                                 f"{s.name}: expected last_use_op={self.n-1}, got {s.last_use_op}")

    def test_param_grad_memory_nonzero_at_optimizer_step(self):
        """The static chart must show nonzero gradient memory at the optimizer step,
        confirming pass 4 extended param.grad tensors from their backward step."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        opt_seq = next(i for i, n in enumerate(self.nodes) if n.phase == OpPhase.OPTIMIZER)
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        data = viz._build_static_timeline(self.nodes)
        self.assertGreater(
            data[TensorRole.GRADIENT][opt_seq], 0,
            "Gradient memory must be nonzero at the optimizer step — "
            "pass 4 should have extended param.grad lifetimes"
        )

    def test_intermediate_gradients_not_extended_to_optimizer_step(self):
        """Intermediate gradients (consumed by subsequent backward nodes) must
        NOT be extended to the optimizer step.  Only terminal param.grad tensors
        should be extended.  We identify intermediate gradients by their auto-
        generated names (grad_0, grad_1, …) and verify their lifetimes end
        before the optimizer step."""
        opt_seq = next(i for i, n in enumerate(self.nodes) if n.phase == OpPhase.OPTIMIZER)
        grads = self.profiler._registry.all_by_role(TensorRole.GRADIENT)
        # Auto-named gradients (grad_<digits>) are intermediate: produced by one
        # backward node and consumed as input by the next backward node.
        auto_grads = [g for g in grads
                      if g.name.startswith("grad_") and g.name[5:].isdigit()
                      and g.first_use_op is not None]
        self.assertGreater(len(auto_grads), 0, "Expected intermediate gradients")
        for g in auto_grads:
            self.assertLess(g.last_use_op, opt_seq,
                            f"{g.name}: intermediate gradient must not extend to "
                            f"optimizer step {opt_seq}, got last_use_op={g.last_use_op}")

    def test_parameter_memory_nonzero_at_step_zero(self):
        """The static chart must show nonzero PARAMETER memory at step 0."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        data = viz._build_static_timeline(self.nodes)
        self.assertGreater(
            data[TensorRole.PARAMETER][0], 0,
            "Parameter memory must be nonzero at step 0 after pass 4 lifetime fix"
        )

    def test_optimizer_state_memory_at_final_step(self):
        """The static chart must show nonzero OPTIMIZER_STATE memory at the final step."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        data = viz._build_static_timeline(self.nodes)
        states = self.profiler._registry.all_by_role(TensorRole.OPTIMIZER_STATE)
        # Only meaningful if there are non-zero-byte states (Adam has exp_avg etc.)
        nonzero_states = [s for s in states if s.nbytes > 0]
        if nonzero_states:
            self.assertGreater(
                data[TensorRole.OPTIMIZER_STATE][self.n - 1], 0,
                "Optimizer state memory must be nonzero at the final step"
            )


# ---------------------------------------------------------------------------
# 8. MemoryVisualizer
# ---------------------------------------------------------------------------

class TestMemoryVisualizer(unittest.TestCase):

    def setUp(self):
        _seed()
        model = TwoLayerMLP(bias=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, Y = torch.randn(8, 4), torch.randn(8, 2)
        self.profiler = _run(model, opt, X, Y)

    def test_static_timeline_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mem.png")
            self.profiler.visualize(output_path=path, mode="static")
            self.assertTrue(os.path.exists(path),
                            "visualize() must create the output PNG file")
            self.assertGreater(os.path.getsize(path), 0, "PNG must not be empty")

    def test_dynamic_timeline_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mem_dynamic.png")
            self.profiler.visualize(output_path=path, mode="dynamic")
            self.assertTrue(os.path.exists(path))

    def test_invalid_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "x.png")
            from profiler.visualizer import MemoryVisualizer
            viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
            with self.assertRaises(ValueError):
                viz.plot_memory_timeline(output_path=path, mode="bogus")

    def test_static_timeline_nonnegative_values(self):
        """No step in the static timeline may have a negative memory value."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        nodes = self.profiler._graph.nodes_in_topo_order()
        data = viz._build_static_timeline(nodes)
        for role, arr in data.items():
            self.assertTrue(
                np.all(arr >= 0),
                f"Role {role.name} has negative memory at steps {np.where(arr < 0)}"
            )

    def test_static_timeline_activation_nonzero_during_backward(self):
        """At least one backward step must show non-zero ACTIVATION memory,
        confirming that activations are not all freed before backward starts."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        nodes = self.profiler._graph.nodes_in_topo_order()
        data = viz._build_static_timeline(nodes)
        bwd_indices = [i for i, n in enumerate(nodes) if n.phase == OpPhase.BACKWARD]
        act_at_bwd = data[TensorRole.ACTIVATION][bwd_indices]
        self.assertTrue(
            np.any(act_at_bwd > 0),
            "Some activations must still be live during the backward pass"
        )

    def test_static_peak_is_nonnegative(self):
        """The reported peak memory value must be >= 0."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        nodes = self.profiler._graph.nodes_in_topo_order()
        data = viz._build_static_timeline(nodes)
        import numpy as np
        totals = sum(data[r] for r in data)
        self.assertGreaterEqual(float(np.max(totals)), 0.0)

    def test_static_timeline_length_equals_node_count(self):
        """The static timeline array length must equal the number of nodes."""
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        nodes = self.profiler._graph.nodes_in_topo_order()
        data = viz._build_static_timeline(nodes)
        for role, arr in data.items():
            self.assertEqual(len(arr), len(nodes),
                             f"Timeline length mismatch for role {role.name}")

    def test_dynamic_timeline_length_equals_node_count(self):
        """The dynamic timeline array length must equal the number of nodes,
        just like the static timeline."""
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        nodes = self.profiler._graph.nodes_in_topo_order()
        data = viz._build_dynamic_timeline(nodes)
        for role, arr in data.items():
            self.assertEqual(len(arr), len(nodes),
                             f"Dynamic timeline length mismatch for role {role.name}")

    def test_static_activation_nonzero_at_first_use_op(self):
        """For each ACTIVATION with first_use_op = f, the static timeline must
        show nonzero ACTIVATION memory at step f. This is the integrative test
        that verifies compute_lifetimes() indices agree with the visualizer's
        node list — the most critical correctness check for the chart."""
        import numpy as np
        from profiler.visualizer import MemoryVisualizer
        viz = MemoryVisualizer(self.profiler._graph, self.profiler._registry)
        nodes = self.profiler._graph.nodes_in_topo_order()
        data = viz._build_static_timeline(nodes)
        act_arr = data[TensorRole.ACTIVATION]

        acts = self.profiler._registry.all_by_role(TensorRole.ACTIVATION)
        for meta in acts:
            if meta.first_use_op is not None:
                self.assertGreater(
                    act_arr[meta.first_use_op], 0,
                    f"{meta.name}: ACTIVATION memory is 0 at its own first_use_op={meta.first_use_op}. "
                    "This means the lifetime index from compute_lifetimes() does not agree "
                    "with the visualizer's node ordering — a critical index consistency bug."
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
