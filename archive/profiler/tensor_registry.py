from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from .graph import ComputationalGraph, TensorMeta, TensorRole


class TensorRegistry:
    """
    Single source of truth for tensor identity and classification.

    Uses (data_ptr, shape) as a composite lookup key instead of id() to survive
    Python wrapper garbage collection while the underlying storage is still live.

    Classification priority:
      1. data_ptr in param set          -> PARAMETER
      2. data_ptr in known grad set     -> GRADIENT
      3. explicit hint == GRADIENT      -> GRADIENT
      4. other explicit hint            -> use it
      5. grad_fn is not None            -> ACTIVATION
      6. fallback                       -> OTHER
    """

    def __init__(self, model: nn.Module) -> None:
        self._param_ptrs: Set[int] = {p.data_ptr() for p in model.parameters()}
        self._param_names: Dict[int, str] = {
            p.data_ptr(): name for name, p in model.named_parameters()
        }
        self._grad_ptrs: Dict[int, str] = {}       # grad data_ptr -> param name
        self._registry: Dict[int, TensorMeta] = {} # id -> TensorMeta (all-time store)
        self._ptr_shape_to_meta: Dict[Tuple[int, tuple], TensorMeta] = {}
        self._name_counters: Dict[str, int] = {}
        self._next_id: int = 0

    # Public API

    def get_or_create(self, tensor: torch.Tensor, hint_role: Optional[TensorRole] = None, hint_name: Optional[str] = None) -> TensorMeta:
        """Look up tensor by (data_ptr, shape); classify and register on first encounter.

        Memory-reuse guard: if a GRADIENT or OPTIMIZER_STATE is expected at an
        address currently held by an ACTIVATION or OTHER, the prior tensor was
        freed and its memory reused. A fresh entry is created instead of returning
        the stale one. Forward outputs should use force_create() to avoid the same
        issue when networks with uniform hidden size reuse activation addresses.
        """
        key = (tensor.data_ptr(), tuple(tensor.shape))
        if key in self._ptr_shape_to_meta:
            existing = self._ptr_shape_to_meta[key]
            _new_roles = (TensorRole.GRADIENT, TensorRole.OPTIMIZER_STATE)
            _old_roles = (TensorRole.ACTIVATION, TensorRole.OTHER, TensorRole.PARAMETER)
            if hint_role in _new_roles and existing.role in _old_roles:
                pass  # memory reused — fall through to create a fresh entry
            else:
                return existing

        role = self._classify(tensor, hint_role)
        name = self._auto_name(role, tensor, hint_name)
        tid = self._next_id
        self._next_id += 1
        meta = TensorMeta(
            tensor_id=tid, name=name, role=role,
            shape=tensor.shape, dtype=tensor.dtype, device=tensor.device,
            nbytes=tensor.numel() * tensor.element_size(),
        )
        self._registry[tid] = meta
        self._ptr_shape_to_meta[key] = meta
        return meta

    def force_create(self, tensor: torch.Tensor, role: TensorRole, name: str) -> TensorMeta:
        """Unconditionally create a new entry, overwriting any stale (data_ptr, shape) mapping.

        Required for forward outputs in uniform-width networks: PyTorch reuses freed
        activation memory for the next layer's output, so naive lookup would conflate
        distinct activations that happen to occupy the same address at different times.
        The old TensorMeta remains in _registry for lifetime accounting.
        """
        key = (tensor.data_ptr(), tuple(tensor.shape))
        tid = self._next_id
        self._next_id += 1
        meta = TensorMeta(
            tensor_id=tid, name=name, role=role,
            shape=tensor.shape, dtype=tensor.dtype, device=tensor.device,
            nbytes=tensor.numel() * tensor.element_size(),
        )
        self._registry[tid] = meta
        self._ptr_shape_to_meta[key] = meta
        return meta

    def mark_gradient(self, param: torch.Tensor, grad: torch.Tensor, param_name: str) -> TensorMeta:
        """Register a parameter gradient, upgrading to the most-qualified name seen.

        If the gradient was already registered under a local name (e.g. "weight"),
        and the fully-qualified name (e.g. "fc2.weight") is longer, updates the name.
        Memory-reuse guard: if the address belongs to an ACTIVATION, falls through
        to get_or_create which creates a fresh GRADIENT entry at that address.
        """
        grad_name = f"grad_{param_name}"
        self._grad_ptrs[grad.data_ptr()] = param_name
        key = (grad.data_ptr(), tuple(grad.shape))
        if key in self._ptr_shape_to_meta:
            meta = self._ptr_shape_to_meta[key]
            if meta.role == TensorRole.GRADIENT:
                if len(grad_name) >= len(meta.name):
                    meta.name = grad_name
                return meta
            # Role mismatch: freed activation memory reused by gradient
        return self.get_or_create(grad, TensorRole.GRADIENT, grad_name)

    def mark_optimizer_state(self, tensor: torch.Tensor, name: str) -> TensorMeta:
        """Register an optimizer state buffer, always allocating a fresh entry.

        Optimizer buffers (momentum, exp_avg) are often allocated at the same
        address as freed gradient tensors, so force-creating avoids mutating a
        stale gradient entry to OPTIMIZER_STATE.
        """
        key = (tensor.data_ptr(), tuple(tensor.shape))
        tid = self._next_id
        self._next_id += 1
        meta = TensorMeta(
            tensor_id=tid, name=name, role=TensorRole.OPTIMIZER_STATE,
            shape=tensor.shape, dtype=tensor.dtype, device=tensor.device,
            nbytes=tensor.numel() * tensor.element_size(),
        )
        self._registry[tid] = meta
        self._ptr_shape_to_meta[key] = meta
        return meta

    def all_by_role(self, role: TensorRole) -> list[TensorMeta]:
        return [m for m in self._registry.values() if m.role == role]

    # Internal helpers

    def _classify(self, tensor: torch.Tensor, hint: Optional[TensorRole]) -> TensorRole:
        ptr = tensor.data_ptr()
        if ptr in self._param_ptrs:
            return TensorRole.PARAMETER
        if ptr in self._grad_ptrs:
            return TensorRole.GRADIENT
        if hint == TensorRole.GRADIENT:
            return TensorRole.GRADIENT
        if hint is not None:
            return hint
        if tensor.grad_fn is not None:
            return TensorRole.ACTIVATION
        return TensorRole.OTHER

    def _auto_name(self, role: TensorRole, tensor: torch.Tensor, hint_name: Optional[str]) -> str:
        if hint_name:
            return hint_name
        ptr = tensor.data_ptr()
        if role == TensorRole.PARAMETER:
            return self._param_names.get(ptr, f"param_{ptr}")
        prefix_map = {
            TensorRole.GRADIENT: "grad",
            TensorRole.ACTIVATION: "act",
            TensorRole.OPTIMIZER_STATE: "opt",
            TensorRole.OTHER: "other",
        }
        prefix = prefix_map.get(role, "tensor")
        count = self._name_counters.get(prefix, 0)
        self._name_counters[prefix] = count + 1
        return f"{prefix}_{count}"


def compute_lifetimes(graph: ComputationalGraph, registry: TensorRegistry) -> None:
    """Set TensorMeta.first_use_op and last_use_op for every tensor.

    Pass 1: first producer (node whose output_tensor_ids contains the tensor).
    Pass 2: last explicit consumer (node whose input_tensor_ids contains it).
    Pass 3: extend activation lifetimes via forward-backward module pairing.
            Autograd saves are invisible to hooks: activation functions save their
            output, linear/conv layers save their input. Without this pass the
            peak memory estimate is under-counted.
    Fixup:  outputs never consumed get last_use = first_use.
    Pass 4: normalise persistent tensor lifetimes for a correct chart:
            - PARAMETERs are live for the entire iteration (step 0 → n-1).
            - OPTIMIZER_STATE persists from creation to the end of the timeline.
            - Terminal (param.grad) gradients are extended to the optimizer step;
              intermediate gradients consumed by backward nodes are not extended.
    """
    from .graph import OpPhase

    nodes = graph.nodes_in_topo_order()

    # Pass 1: first producer
    for seq, node in enumerate(nodes):
        for tid in node.output_tensor_ids:
            meta = registry._registry.get(tid)
            if meta is not None and meta.first_use_op is None:
                meta.first_use_op = seq

    # Pass 2: last explicit consumer
    for seq, node in enumerate(nodes):
        for tid in node.input_tensor_ids:
            meta = registry._registry.get(tid)
            if meta is not None:
                if meta.last_use_op is None or seq > meta.last_use_op:
                    meta.last_use_op = seq

    # Pass 3: extend activation lifetimes via forward-backward module pairing
    fqn_to_bwd_seq: Dict[str, int] = {}
    for seq, node in enumerate(nodes):
        if node.phase == OpPhase.BACKWARD:
            fqn_to_bwd_seq[node.module_fqn] = seq  # keep last if duplicates exist

    for seq, node in enumerate(nodes):
        if node.phase != OpPhase.FORWARD:
            continue
        bwd_seq = fqn_to_bwd_seq.get(node.module_fqn)
        if bwd_seq is None:
            continue
        is_linear_like = any(s in node.op_name.lower() for s in ("linear", "conv", "bilinear", "embedding"))
        # linear/conv save their input; activation functions save their output
        tids_to_extend = node.input_tensor_ids if is_linear_like else node.output_tensor_ids
        for tid in tids_to_extend:
            meta = registry._registry.get(tid)
            if meta is not None and meta.role == TensorRole.ACTIVATION:
                if meta.last_use_op is None or bwd_seq > meta.last_use_op:
                    meta.last_use_op = bwd_seq

    # Fixup: outputs never consumed
    for node in nodes:
        for tid in node.output_tensor_ids:
            meta = registry._registry.get(tid)
            if meta is not None and meta.last_use_op is None:
                meta.last_use_op = meta.first_use_op

    # Pass 4: normalise persistent tensor lifetimes
    n_steps = len(nodes)
    opt_seq: Optional[int] = None
    for seq, node in enumerate(nodes):
        if node.phase == OpPhase.OPTIMIZER:
            opt_seq = seq

    for meta in registry._registry.values():
        if meta.first_use_op is None:
            continue
        if meta.role == TensorRole.PARAMETER:
            meta.first_use_op = 0
            meta.last_use_op = n_steps - 1
        elif meta.role == TensorRole.OPTIMIZER_STATE:
            meta.last_use_op = n_steps - 1
        elif meta.role == TensorRole.GRADIENT and opt_seq is not None:
            # Terminal gradients (param.grad) have last_use == first_use after
            # the fixup because no node explicitly consumes them as input.
            # Intermediate gradients consumed by later backward nodes have
            # last_use > first_use and must NOT be extended.
            if meta.last_use_op == meta.first_use_op and opt_seq > meta.last_use_op:
                meta.last_use_op = opt_seq
