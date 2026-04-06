"""
Graph Tracer — traces a training step into a single FX graph.

Provided by the CS265 course starter code (Qitong Wang, DASLab).
Captures forward + backward + optimizer into one rewritable GraphModule
using make_fx() with FakeTensorMode.
"""

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Union
from utils import SPMD_DECOMP_TABLE

import torch
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import CodeGen, _PyTreeCodeGen, _PyTreeInfo
from torch.nn.utils import stateless
from torch.utils.hooks import RemovableHandle


# --- Separator ops: identity functions that mark forward/backward boundaries ---

def sep(x: torch.Tensor) -> torch.Tensor:
    return x

def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    return grad

separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")


def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec)
    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))

DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(
    torch.ops.separator.sep.default, _identity_prop_rule)
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(
    torch.ops.separator.sep_backward.default, _identity_prop_rule)


class SEPFunction(torch.autograd.Function):
    """Custom autograd function that inserts %sep / %sep_backward marker nodes."""
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep_backward(grad_x)


# --- Gradient tagging (for SPMD, removed after tracing) ---

_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")
_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


# --- Input flattening ---

class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    def process_inputs(self, *args: Any) -> Any:
        return args
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    """Shift input flattening responsibility from graph module to caller."""
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            orig_args=None,
            in_spec=None,
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


# --- Context managers for tracing ---

@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """Tag gradients with dummy ops during tracing (removed afterwards)."""
    hooks: List[RemovableHandle] = []
    try:
        for p in params.values():
            hooks.append(p.register_hook(lambda grad: torch.ops.dummy.tag_grad(grad)))
        yield
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def _rematerialize_optimizer(opt: optim.Optimizer, named_states: Dict[str, Any], params: Dict[str, nn.Parameter]):
    """Temporarily replace optimizer state with proxy tensors for tracing."""
    assert opt is not None
    orig_states = copy(opt.state)
    for n in named_states:
        opt.state[params[n]] = named_states[n]
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()
    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states


@contextmanager
def _enable_compile():
    """Monkey-patch is_compiling() to return True so optimizer ops are traced."""
    def f_true():
        return True
    orig = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig


# --- Compilation ---

@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


def _compile(func: Callable, *args: Any, **kwargs: Any):
    """Trace func into an FX graph containing forward + backward + optimizer."""
    # 1. Extract model and optimizer from args.
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None
            mod = arg
        if isinstance(arg, optim.Optimizer):
            assert opt is None
            opt = arg
    assert mod is not None

    # 2. Lift parameters, buffers, optimizer state as function arguments.
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))
    named_states: Dict[str, nn.Parameter] = {}
    for n, p in params.items():
        if p in opt.state:
            named_states[n] = opt.state[p]

    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(mod, {**params, **buffers}), \
             _rematerialize_optimizer(opt, named_states, params) if opt else nullcontext():
            with gradients_tagging(params):
                ret = func(*args, **kwargs)
            return ret, list(mod.parameters()), list(named_states.values())

    # 3. Trace with fake tensors.
    fake_mode = FakeTensorMode()
    args = pytree.tree_map_only(torch.Tensor, lambda t: fake_mode.from_tensor(t), args)
    kwargs = pytree.tree_map_only(torch.Tensor, lambda t: fake_mode.from_tensor(t), kwargs)

    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode="fake",
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    # 4. Clean up: remove detach and tag_grad nodes.
    for node in gm.graph.nodes:
        if node.target in (torch.ops.aten.detach.default, torch.ops.dummy.tag_grad.default):
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    gm = _to_caller_flattened_graph_module(gm)
    flat_state, _ = pytree.tree_flatten([{**params, **buffers}, named_states])
    return _CompiledResult(gm, mod, opt, flat_state)


COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(func: Callable, gm_transformation: Callable):
    """Compile func: trace on first call, apply gm_transformation, cache result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        first_iter = False
        compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
        if compiled_obj is None:
            first_iter = True
            compiled_obj = _compile(func, *args, **kwargs)
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
        flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]
        if first_iter and gm_transformation:
            compiled_obj.gm = gm_transformation(compiled_obj.gm, flat_inps)
        with torch.no_grad():
            output = compiled_obj.gm(*flat_inps)[0]
        return output
    return wrapper
