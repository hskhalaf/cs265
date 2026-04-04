"""
Memory Visualizer for CS265 Activation Checkpointing Project.

Produces a stacked bar chart showing GPU memory usage over the timeline of
graph execution, broken down by tensor role (PARAM, ACT, GRAD, OTHER).
The peak memory step is annotated with a dashed vertical line.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments.
import matplotlib.pyplot as plt
import numpy as np

from graph_prof import GraphProfiler, NodeType, Region


# Colours and display order (bottom to top).
_ROLE_STYLE = {
    NodeType.PARAM:          {"color": "#4C72B0", "label": "Parameter"},
    NodeType.ACT:            {"color": "#55A868", "label": "Activation"},
    NodeType.GRAD:           {"color": "#DD8452", "label": "Gradient"},
    NodeType.OPTIMIZER_STATE: {"color": "#C44E52", "label": "Optimizer State"},
    NodeType.OTHER:          {"color": "#8172B3", "label": "Other"},
}
_ROLE_ORDER = [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OPTIMIZER_STATE, NodeType.OTHER]


class MemoryVisualizer:
    """Generates memory breakdown charts from profiler data.

    Parameters
    ----------
    profiler : GraphProfiler
        A profiler that has already been run and had ``aggregate_stats()``
        called.
    """

    def __init__(self, profiler: GraphProfiler):
        self.profiler = profiler

    def plot_memory_timeline(
        self,
        path: str = "memory_breakdown.png",
        title: Optional[str] = None,
        figsize: tuple = (14, 6),
    ) -> None:
        """Generate and save a stacked bar chart of memory usage over time.

        Parameters
        ----------
        path : str
            Output file path (PNG).
        title : str, optional
            Chart title.  Defaults to "Memory Breakdown by Tensor Role".
        figsize : tuple
            Figure size in inches.
        """
        by_role = self.profiler._compute_live_memory_timeline_by_role()
        n_steps = len(self.profiler.node_list)

        if n_steps == 0:
            print("Warning: no nodes in graph; nothing to plot.")
            return

        # Convert bytes → MB.
        role_data = {
            role: np.array(by_role[role], dtype=np.float64) / (1024 * 1024)
            for role in _ROLE_ORDER
        }

        x = np.arange(n_steps)
        fig, ax = plt.subplots(figsize=figsize)

        bottom = np.zeros(n_steps)
        for role in _ROLE_ORDER:
            style = _ROLE_STYLE[role]
            values = role_data[role]
            ax.bar(
                x, values, bottom=bottom,
                color=style["color"], label=style["label"],
                width=1.0, edgecolor="none",
            )
            bottom += values

        # Mark the peak.
        total = sum(role_data[r] for r in _ROLE_ORDER)
        peak_step = int(np.argmax(total))
        peak_mb = total[peak_step]
        ax.axvline(
            x=peak_step, color="black", linestyle="--", linewidth=0.8,
            label=f"Peak ({peak_mb:.1f} MB @ step {peak_step})",
        )

        # Mark region boundaries.
        sep_idx = self.profiler.sep_index
        sep_bwd_idx = self.profiler.sep_bwd_index
        opt_idx = self.profiler.optimizer_index

        for boundary, label in [
            (sep_idx, "sep"),
            (sep_bwd_idx, "sep_bwd"),
        ]:
            if boundary >= 0:
                ax.axvline(
                    x=boundary, color="gray", linestyle=":",
                    linewidth=0.6, alpha=0.7,
                )
                ax.text(
                    boundary, ax.get_ylim()[1] * 0.95, f" {label}",
                    fontsize=7, color="gray", rotation=90,
                    verticalalignment="top",
                )

        ax.set_xlabel("Operation index (topological order)")
        ax.set_ylabel("Live memory (MB)")
        ax.set_title(title or "Memory Breakdown by Tensor Role")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(-0.5, n_steps - 0.5)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Memory breakdown chart saved to {path}")
