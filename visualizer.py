"""
Plotting helpers for Phase 1.

Two functions:

  * ``plot_memory_breakdown(profiler, path, title)`` — stacked-area chart
    of static live memory by tensor role over the iteration.
  * ``plot_peak_vs_batch(rows, path, title)`` — bar chart of static peak
    memory per batch size.  ``rows`` = ``[{"batch_size": int, "peak_mb": float}, ...]``.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_prof import GraphProfiler, NodeType


# Colour palette — same role, same colour, everywhere.
_ROLE_COLOR: Dict[NodeType, str] = {
    NodeType.PARAM: "#4C72B0",   # blue
    NodeType.OPT_STATE: "#8172B3",  # purple
    NodeType.ACT:   "#55A868",   # green
    NodeType.GRAD:  "#DD8452",   # orange
    NodeType.OTHER: "#999999",   # gray
}

# Stack order, bottom to top.
_STACK_ORDER: Tuple[NodeType, ...] = (
    NodeType.PARAM,
    NodeType.OPT_STATE,
    NodeType.OTHER,
    NodeType.ACT,
    NodeType.GRAD,
)


def plot_memory_breakdown(profiler: GraphProfiler,
                          path: str,
                          title: Optional[str] = None) -> None:
    """Stacked-area chart of static live memory by tensor role over the
    iteration.  Vertical guides mark the sep / sep_backward / optimizer
    boundaries and the peak step."""
    n = len(profiler.nodes)
    if n == 0:
        print("Warning: empty graph; nothing to plot.")
        return

    timeline = profiler.memory_timeline_by_role()
    x = np.arange(n)
    series = {role: np.array(timeline[role], dtype=float) / (1024 ** 2)
              for role in _STACK_ORDER}

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(
        x,
        *[series[r] for r in _STACK_ORDER],
        labels=[r.name for r in _STACK_ORDER],
        colors=[_ROLE_COLOR[r] for r in _STACK_ORDER],
        alpha=0.9, edgecolor="none",
    )

    total = sum(series[r] for r in _STACK_ORDER)
    peak_step = int(np.argmax(total))
    peak_mb   = float(total[peak_step])
    ax.axvline(peak_step, color="black", linestyle="--", linewidth=0.8,
               label=f"Peak {peak_mb:.1f} MB @ step {peak_step}")

    for boundary, label in [
        (profiler.sep_idx,     "sep"),
        (profiler.sep_bwd_idx, "sep_bwd"),
        (profiler.opt_idx,     "optimizer"),
    ]:
        if boundary >= 0:
            ax.axvline(boundary, color="gray", linestyle=":",
                       linewidth=0.6, alpha=0.6)
            ax.text(boundary, ax.get_ylim()[1] * 0.95, f" {label}",
                    fontsize=8, color="gray", rotation=90,
                    verticalalignment="top")

    ax.set_xlabel("Operation index (topological order)")
    ax.set_ylabel("Live memory (MB)")
    ax.set_title(title or "Memory Breakdown by Tensor Role")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_peak_vs_batch(rows: List[dict], path: str, title: str) -> None:
    """Single-series bar chart: one bar per batch size showing the static
    peak.  ``rows`` = ``[{"batch_size": int, "peak_mb": float}, ...]``."""
    rows = sorted(rows, key=lambda r: r["batch_size"])
    bs   = [r["batch_size"] for r in rows]
    vals = [r["peak_mb"]    for r in rows]

    x = np.arange(len(bs))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, vals, width=0.55, color="#4C72B0")
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
