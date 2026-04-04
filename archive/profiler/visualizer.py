from __future__ import annotations

from typing import Dict, List

import numpy as np

from .graph import ComputationalGraph, GraphNode, TensorRole
from .tensor_registry import TensorRegistry

# Stacked bar display order (bottom to top) and colors
_ROLE_ORDER = [
    TensorRole.PARAMETER,
    TensorRole.ACTIVATION,
    TensorRole.GRADIENT,
    TensorRole.OPTIMIZER_STATE,
    TensorRole.OTHER,
]

_ROLE_COLORS = {
    TensorRole.PARAMETER:       "#4e79a7",  # blue
    TensorRole.ACTIVATION:      "#59a14f",  # green
    TensorRole.GRADIENT:        "#f28e2b",  # orange
    TensorRole.OPTIMIZER_STATE: "#e15759",  # red
    TensorRole.OTHER:           "#b07aa1",  # purple
}


class MemoryVisualizer:
    """
    Stacked bar chart of memory by TensorRole over the execution timeline.

    'static'  — estimates live memory at each step from first/last_use_op lifetimes.
    'dynamic' — accumulates measured gpu_memory_bytes deltas from ProfileResult.
    """

    def __init__(self, graph: ComputationalGraph, registry: TensorRegistry) -> None:
        self._graph = graph
        self._registry = registry

    def plot_memory_timeline(self, output_path: str = "memory_breakdown.png", mode: str = "static") -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required: pip install matplotlib")

        nodes = self._graph.nodes_in_topo_order()
        n_steps = len(nodes)
        if n_steps == 0:
            print("Warning: empty graph — nothing to visualize.")
            return

        node_labels = [f"{n.op_name}\n({n.phase.name[0]})" for n in nodes]

        if mode == "static":
            data = self._build_static_timeline(nodes)
        elif mode == "dynamic":
            data = self._build_dynamic_timeline(nodes)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'static' or 'dynamic'.")

        fig, ax = plt.subplots(figsize=(max(14, n_steps * 0.7), 6))
        bottoms = np.zeros(n_steps)

        for role in _ROLE_ORDER:
            values_bytes = data.get(role, np.zeros(n_steps))
            values_mb = values_bytes / (1024 * 1024)
            ax.bar(range(n_steps), values_mb, bottom=bottoms / (1024 * 1024),
                   label=role.name, color=_ROLE_COLORS[role], alpha=0.85, width=0.7)
            bottoms += values_bytes

        # Mark peak
        totals_mb = bottoms / (1024 * 1024)
        peak_step = int(np.argmax(totals_mb))
        peak_val = totals_mb[peak_step]
        ax.axvline(x=peak_step, color="black", linestyle="--", linewidth=1.5, zorder=5)
        y_top = ax.get_ylim()[1]
        ax.annotate(
            f"Peak: {peak_val:.3f} MB",
            xy=(peak_step, peak_val),
            xytext=(peak_step + 0.3, y_top * 0.92),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
            va="top",
        )

        ax.set_xticks(range(n_steps))
        ax.set_xticklabels(node_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Operation (topological order)", fontsize=10)
        ax.set_ylabel("Memory (MB)", fontsize=10)
        mode_label = "Estimated (static lifetime analysis)" if mode == "static" else "Measured (dynamic)"
        ax.set_title(f"Peak Memory Breakdown by Tensor Role\n{mode_label}", fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Timeline builders

    def _build_static_timeline(self, nodes: List[GraphNode]) -> Dict[TensorRole, np.ndarray]:
        """Tensor is live at step t iff first_use_op <= t <= last_use_op; contribution = nbytes."""
        n = len(nodes)
        data: Dict[TensorRole, np.ndarray] = {role: np.zeros(n, dtype=np.float64) for role in TensorRole}
        for meta in self._registry._registry.values():
            if meta.first_use_op is None:
                continue
            first = max(0, min(meta.first_use_op, n - 1))
            last = max(0, min(meta.last_use_op if meta.last_use_op is not None else n - 1, n - 1))
            data[meta.role][first : last + 1] += meta.nbytes
        return data

    def _build_dynamic_timeline(self, nodes: List[GraphNode]) -> Dict[TensorRole, np.ndarray]:
        """Cumulative sum of per-op gpu_memory_bytes deltas, attributed to output tensor roles."""
        n = len(nodes)
        data: Dict[TensorRole, np.ndarray] = {role: np.zeros(n, dtype=np.float64) for role in TensorRole}
        for t, node in enumerate(nodes):
            if node.profile is None or node.profile.gpu_memory_bytes == 0:
                continue
            delta = node.profile.gpu_memory_bytes
            if node.output_tensor_ids:
                per_tensor = delta / len(node.output_tensor_ids)
                for tid in node.output_tensor_ids:
                    meta = self._registry._registry.get(tid)
                    if meta:
                        data[meta.role][t] += per_tensor
            else:
                data[TensorRole.OTHER][t] += delta
        for role in TensorRole:
            data[role] = np.cumsum(data[role])
        return data
