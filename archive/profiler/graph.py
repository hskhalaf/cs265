from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch


class OpPhase(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    OPTIMIZER = auto()


class TensorRole(Enum):
    PARAMETER = auto()
    GRADIENT = auto()
    ACTIVATION = auto()
    OPTIMIZER_STATE = auto()
    OTHER = auto()


@dataclass
class TensorMeta:
    """Metadata for one logical tensor tracked by the profiler."""
    tensor_id: int          # stable synthetic id
    name: str               # human-readable label, e.g. "linear1.output"
    role: TensorRole
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    nbytes: int             # numel * element_size

    # Populated by compute_lifetimes(); indices into the topo-ordered node list
    first_use_op: Optional[int] = None
    last_use_op: Optional[int] = None


@dataclass
class ProfileResult:
    """Runtime measurements captured around one node execution."""
    wall_time_ms: float
    gpu_memory_bytes: int   # delta in torch.cuda.memory_allocated()
    cpu_memory_bytes: int   # delta from tracemalloc
    peak_gpu_bytes: int     # max_memory_allocated() over this op


@dataclass
class GraphNode:
    """One node in the computational graph — one module-level operation."""
    node_id: int
    op_name: str
    phase: OpPhase
    module_fqn: str                      # fully-qualified module name
    input_tensor_ids: List[int]
    output_tensor_ids: List[int]
    profile: Optional[ProfileResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Directed data-dependency edge between two nodes."""
    src_node_id: int
    dst_node_id: int
    tensor_id: int


class ComputationalGraph:
    """
    Mutable graph built incrementally by hooks; frozen for analysis.

    Nodes are stored in insertion order (= hook-capture order).
    After freeze(), nodes_in_topo_order() returns a valid topological sort.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._frozen: bool = False
        self._next_node_id: int = 0

    # Building API

    def next_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def add_node(self, node: GraphNode) -> None:
        if self._frozen:
            raise RuntimeError("Cannot add nodes to a frozen graph.")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if self._frozen:
            raise RuntimeError("Cannot add edges to a frozen graph.")
        self._edges.append(edge)

    # Query API

    def nodes_in_topo_order(self) -> List[GraphNode]:
        """Kahn's algorithm with min-heap tie-breaking by node_id.

        node_ids are assigned in hook-capture order, so the heap priority
        naturally preserves forward → backward → optimizer ordering when
        no edges constrain the sort (backward nodes carry no explicit incoming
        edges from the forward graph).
        """
        import heapq

        in_degree: Dict[int, int] = {nid: 0 for nid in self._nodes}
        children: Dict[int, set] = defaultdict(set)

        for edge in self._edges:
            src, dst = edge.src_node_id, edge.dst_node_id
            if src in in_degree and dst in in_degree and src != dst:
                if dst not in children[src]:
                    in_degree[dst] += 1
                    children[src].add(dst)

        heap = [nid for nid, deg in in_degree.items() if deg == 0]
        heapq.heapify(heap)
        order: List[GraphNode] = []

        while heap:
            nid = heapq.heappop(heap)
            order.append(self._nodes[nid])
            for child in sorted(children[nid]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    heapq.heappush(heap, child)

        # Safety net: append any stranded nodes if a cycle slips through
        if len(order) < len(self._nodes):
            seen = {n.node_id for n in order}
            for nid in sorted(self._nodes):
                if nid not in seen:
                    order.append(self._nodes[nid])

        return order

    def freeze(self) -> None:
        self._frozen = True

    @property
    def nodes(self) -> Dict[int, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> List[GraphEdge]:
        return self._edges
