"""
Phase 1 graph profiler.

Input: a traced training iteration (forward pass, loss, backward pass, optimizer step) in one FX graph.  

The profiler walks that graph in topological order and records the following:
1. the region for each operation (forward, loss, backward, or optimizer)
2. the tensor role for each output (parameter, activation, gradient, optimizer state, or other)
3. the output size and the amount of new storage the operation creates
4. the first and last use of each activation saved for backward
5. CUDA timing and allocator measurements for each operation

Several FX nodes can refer to the same GPU allocation 
Example: views, in-place ops, and getitem nodes that unpack multi-output operations.

The profiler maps each node to the node that owns its storage, 
then counts each allocation once over the interval where that storage is live.
"""

from __future__ import annotations
import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
import torch.fx as fx



class NodeType(IntEnum):
    OTHER = 0
    OPT_STATE = 1
    ACT   = 2
    GRAD  = 3
    PARAM = 4

# the numeric values serve as the priority order. 
# this is useful when several aliases share one storage allocation.  
# e.g., if an activation is later viewed by another node, the shared storage is activation memory.

class Region(Enum):
    FORWARD   = "forward"
    LOSS      = "loss"
    BACKWARD  = "backward"
    OPTIMIZER = "optimizer"

@dataclass
class Intermediate:
    node:          fx.Node
    size_bytes:    int   = 0
    last_fwd_idx:  int   = -1
    first_bwd_idx: int   = -1
    recompute_ms:  float = 0.0


DISPLAY_ROLES = (NodeType.PARAM, NodeType.OPT_STATE, NodeType.ACT, NodeType.GRAD, NodeType.OTHER)

def iter_tensors(val: Any) -> Iterable[torch.Tensor]:
    if isinstance(val, torch.Tensor): yield val
    elif isinstance(val, (list, tuple)):
        for v in val: 
            yield from iter_tensors(v)

def alias_in_schema(node: fx.Node) -> bool: # to avoid double counting
    schema = getattr(node.target, "_schema", None)
    return schema is not None and any(r.alias_info is not None for r in schema.returns)

def is_getitem(node: fx.Node) -> bool: # useful bcz anything that returns more than one tensor gets unpacked through operator.getitem. however getitem is a reference, not an allocation
    return (node.op == "call_function" and (node.target is operator.getitem or getattr(node.target, "__name__", None) == "getitem"))

def contains_tensor(val: Any) -> bool: 
    return next(iter_tensors(val), None) is not None

def is_container(node: fx.Node) -> bool:
    val = node.meta.get("val")
    return isinstance(val, (list, tuple)) and contains_tensor(val)

def is_getitem_of_container(node: fx.Node) -> bool:
    return (is_getitem(node) and bool(node.all_input_nodes) and is_container(node.all_input_nodes[0]))


def output_bytes(node: fx.Node) -> int:
    if node.op not in ("call_function", "placeholder"): return 0
    return sum(t.numel() * t.element_size() for t in iter_tensors(node.meta.get("val")))

def allocation_bytes(node: fx.Node) -> int:
    if node.op == "call_function":
        if alias_in_schema(node): return 0
        if is_container(node) and all(is_getitem(u) for u in node.users): return 0

    if is_getitem_of_container(node):
        parent = node.all_input_nodes[0]
        if parent.op == "call_function" and alias_in_schema(parent): return 0
    return output_bytes(node)


def alias_target(parent: fx.Node, getitem: fx.Node) -> Optional[fx.Node]:
    """Return the input storage used by an aliasing ``getitem`` output."""
    if len(getitem.args) < 2 or not parent.args:  return None
    first = parent.args[0]
    if isinstance(first, fx.Node): return first
    idx = getitem.args[1]
    if (isinstance(first, (list, tuple)) and isinstance(idx, int) and 0 <= idx < len(first) and isinstance(first[idx], fx.Node)): 
        return first[idx]
    return None

class GraphProfiler(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.nodes: List[fx.Node] = list(self.module.graph.nodes)
        self.idx:   Dict[fx.Node, int] = {n: i for i, n in enumerate(self.nodes)}
        self.find_separators()
        self.assign_regions()
        self.classify_tensors()
        self.find_intermediates()
        self.compute_sizes()
        self.compute_aliases()
        self.reset_stats()

    def find_separators(self) -> None:
        self.sep_idx = self.sep_bwd_idx = self.opt_idx = -1
        first_foreach_after_bwd = -1
        for i, n in enumerate(self.nodes):
            if n.op != "call_function": 
                continue
            t = n.target
            if t == torch.ops.separator.sep.default:           
                self.sep_idx = i
            elif t == torch.ops.separator.sep_backward.default: 
                self.sep_bwd_idx = i
            elif t == torch.ops.aten._fused_adam.default: 
                self.opt_idx = i
            elif (first_foreach_after_bwd < 0 and self.sep_bwd_idx >= 0 and "_foreach_" in str(t)):
                first_foreach_after_bwd = i
        if self.opt_idx < 0:
            self.opt_idx = first_foreach_after_bwd

        assert self.sep_idx     >= 0, "no %sep marker in graph"
        assert self.sep_bwd_idx >= 0, "no %sep_backward marker in graph"

    def assign_regions(self) -> None:
        self.region: Dict[fx.Node, Region] = {}
        for i, n in enumerate(self.nodes):
            if 0 <= self.opt_idx <= i: self.region[n] = Region.OPTIMIZER
            elif i >= self.sep_bwd_idx: self.region[n] = Region.BACKWARD
            elif i >  self.sep_idx: self.region[n] = Region.LOSS
            else: self.region[n] = Region.FORWARD

    def classify_tensors(self) -> None:
        self.params: Set[fx.Node] = set()
        self.grads:  Set[fx.Node] = set()
        self.opt_states: Set[fx.Node] = set()

        fused = next((n for n in self.nodes if n.op == "call_function" and n.target == torch.ops.aten._fused_adam.default), None)
        if fused is not None:
            self.params = set(fused.args[0])
            self.grads  = set(fused.args[1])
            for arg in fused.args[2:]:
                if isinstance(arg, (list, tuple)):
                    self.opt_states.update(x for x in arg if isinstance(x, fx.Node))

        elif self.opt_idx >= 0:
            for n in self.nodes:
                if n.op == "placeholder":
                    user_idxs = [self.idx[u] for u in n.users if u in self.idx]
                    in_fwd = any(j < self.sep_idx for j in user_idxs)
                    in_opt = any(j >= self.opt_idx for j in user_idxs)
                    if in_fwd and in_opt:
                        self.params.add(n)
                    elif in_opt:
                        self.opt_states.add(n)
                elif (n.op == "call_function" and self.sep_bwd_idx <= self.idx[n] < self.opt_idx and any(self.idx[u] >= self.opt_idx for u in n.users if u in self.idx)):
                    self.grads.add(n)

        self.node_type: Dict[fx.Node, NodeType] = {
            n: (NodeType.PARAM if n in self.params else
                NodeType.GRAD  if n in self.grads  else
                NodeType.OPT_STATE if n in self.opt_states else
                NodeType.OTHER)
            for n in self.nodes
        }

    def find_intermediates(self) -> None:
        self.intermediates: List[Intermediate] = []
        for n in self.nodes:
            i = self.idx[n]
            if i >= self.sep_idx or n.op != "call_function" or n in self.params:
                continue
            user_idxs = [self.idx.get(u, -1) for u in n.users]
            if not any(j >= self.sep_bwd_idx for j in user_idxs):
                continue
            self.intermediates.append(Intermediate(
                node=n,
                last_fwd_idx =max((j for j in user_idxs if j <  self.sep_idx), default=i),
                first_bwd_idx=min((j for j in user_idxs if j >= self.sep_bwd_idx), default=-1),
            ))
            self.node_type[n] = NodeType.ACT

    def compute_sizes(self) -> None:
        self.node_logical_size_bytes: Dict[fx.Node, int] = {n: output_bytes(n) for n in self.nodes}
        self.node_size_bytes: Dict[fx.Node, int] = {n: allocation_bytes(n) for n in self.nodes}
        for inter in self.intermediates:
            inter.size_bytes = self.node_logical_size_bytes[inter.node]

    def compute_aliases(self) -> None:
        self.storage_owner: Dict[fx.Node, fx.Node] = {}
        self.aliases_by_owner: Dict[fx.Node, List[fx.Node]] = defaultdict(list)

        for n in self.nodes:
            owner = n
            if is_getitem_of_container(n):
                parent = n.all_input_nodes[0]
                if parent.op == "call_function" and alias_in_schema(parent):
                    target = alias_target(parent, n)
                    if target is not None and self.node_logical_size_bytes.get(target, 0) > 0:
                        owner = self.storage_owner.get(target, target)
            elif n.op == "call_function" and alias_in_schema(n):
                for inp in n.all_input_nodes:
                    if self.node_logical_size_bytes.get(inp, 0) > 0:
                        owner = self.storage_owner.get(inp, inp)
                        break
            self.storage_owner[n] = owner
            self.aliases_by_owner[owner].append(n)

    def reset_stats(self) -> None:
        self._runtimes_ms: Dict[str, List[float]] = defaultdict(list)
        self._memory_after_bytes: Dict[str, List[int]] = defaultdict(list)
        self._memory_delta_bytes: Dict[str, List[int]] = defaultdict(list)
        self._memory_peak_bytes: Dict[str, List[int]] = defaultdict(list)
        self._latencies_ms: List[float] = []

        self.avg_runtime_ms: Dict[str, float] = {}
        self.avg_memory_after_bytes: Dict[str, int] = {}
        self.avg_memory_delta_bytes: Dict[str, int] = {}
        self.avg_memory_peak_bytes: Dict[str, int] = {}
        self.avg_iter_latency_ms: float = 0.0

    def run(self, *args, **kwargs) -> Any:
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True),  torch.cuda.Event(enable_timing=True))
        start.record()
        result = super().run(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self._latencies_ms.append(start.elapsed_time(end))
        return result

    def run_node(self, n: fx.Node) -> Any:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        self._runtimes_ms[n.name].append(start.elapsed_time(end))
        self._memory_after_bytes[n.name].append(after)
        self._memory_delta_bytes[n.name].append(after - before)
        self._memory_peak_bytes[n.name].append(torch.cuda.max_memory_allocated())
        return result

    def aggregate_stats(self) -> None:
        self.avg_runtime_ms = {name: sum(vs) / len(vs) for name, vs in self._runtimes_ms.items()}
        self.avg_memory_after_bytes = {name: sum(vs) // len(vs) for name, vs in self._memory_after_bytes.items()}
        self.avg_memory_delta_bytes = {name: sum(vs) // len(vs) for name, vs in self._memory_delta_bytes.items()}
        self.avg_memory_peak_bytes = {name: sum(vs) // len(vs) for name, vs in self._memory_peak_bytes.items()}
        for inter in self.intermediates:
            inter.recompute_ms = self.avg_runtime_ms.get(inter.node.name, 0.0)
        if self._latencies_ms:
            self.avg_iter_latency_ms = (sum(self._latencies_ms) / len(self._latencies_ms))


    def memory_timeline_by_role(self) -> Dict[NodeType, List[int]]:
        n = len(self.nodes)
        timeline: Dict[NodeType, List[int]] = {nt: [0] * n for nt in NodeType}

        for owner, aliases in self.aliases_by_owner.items():
            size = self.node_size_bytes.get(owner, 0)
            if size == 0:
                continue
            role = max(self.node_type[a] for a in aliases)
            if owner.op == "placeholder":
                lo, hi = 0, n - 1
            else:
                lo = self.idx[owner]
                hi = lo
                for a in aliases:
                    hi = max(hi, self.idx[a], *(self.idx[u] for u in a.users if u in self.idx))
            for t in range(lo, min(hi + 1, n)):
                timeline[role][t] += size
        return timeline

    def peak_memory_bytes(self) -> int:
        n = len(self.nodes)
        if n == 0:
            return 0
        roles = self.memory_timeline_by_role()
        return max(sum(roles[nt][t] for nt in NodeType) for t in range(n))

    def iteration_latency_ms(self) -> float:
        return self.avg_iter_latency_ms

    def print_summary(self) -> None:
        timeline = self.memory_timeline_by_role()
        n = len(self.nodes)
        per_step = [sum(timeline[nt][t] for nt in NodeType) for t in range(n)]
        peak_step = max(range(n), key=per_step.__getitem__) if n else 0

        roles = Counter(self.node_type.values())
        regions = Counter(self.region.values())

        print(f"  Nodes: {n}  "
              f"(forward {regions[Region.FORWARD]}, "
              f"backward {regions[Region.BACKWARD]}, "
              f"optimizer {regions[Region.OPTIMIZER]})")
        print(f"  Intermediates: {len(self.intermediates)}\n")
        print(f"  Static peak (step {peak_step}):")
        print(f"    {'Role':<8} {'Nodes':>6} {'Bytes':>10}")
        for nt in DISPLAY_ROLES:
            print(f"    {nt.name:<8} {roles[nt]:>6}"
                  f" {timeline[nt][peak_step] / 1024**2:>7.2f} MB")
        print(f"    {'TOTAL':<8} {n:>6} {per_step[peak_step] / 1024**2:>7.2f} MB\n")
        print(f"  Iteration latency: {self.avg_iter_latency_ms:>7.2f} ms"
              f"  (avg of {len(self._latencies_ms)} runs)")
