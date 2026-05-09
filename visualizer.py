"""
Plotting helpers for the CS265 project.

Three top-level functions:

* ``plot_memory_breakdown(profiler, path, title=None)`` — Phase 1's stacked
  area chart of live memory by tensor role over the iteration.

* ``plot_peak_memory_vs_batch(rows, path, title)`` — deliverable plot:
  grouped bars comparing peak memory with vs. without activation
  checkpointing across batch sizes.

* ``plot_latency_vs_batch(rows, path, title)`` — same shape but for
  iteration latency.

The deliverable plots take ``rows``: a list of dicts of the form
``{"batch_size": int, "ac_off": float, "ac_on": float}`` (in MB or ms).
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_prof import GraphProfiler, NodeType


# Colour palette — kept stable across all charts so the same role looks the
# same wherever it appears.
_ROLE_COLOR: Dict[NodeType, str] = {
    NodeType.PARAM: "#4C72B0",   # blue
    NodeType.ACT:   "#55A868",   # green
    NodeType.GRAD:  "#DD8452",   # orange
    NodeType.OTHER: "#999999",   # gray
}

# Stack order, bottom to top.
_STACK_ORDER: Tuple[NodeType, ...] = (
    NodeType.PARAM,
    NodeType.OTHER,
    NodeType.ACT,
    NodeType.GRAD,
)


# --------------------------------------------------------------------------- #
# Phase 1 chart                                                               #
# --------------------------------------------------------------------------- #


def plot_memory_breakdown(profiler: GraphProfiler,
                          path: str,
                          title: Optional[str] = None) -> None:
    """Stacked area chart of live memory by role across the iteration.

    Vertical guides mark the forward/backward and backward/optimizer
    boundaries, plus the peak step.
    """
    timeline = profiler.memory_timeline_by_role(include_runtime_residual=True)
    n_steps = len(profiler.nodes)
    if n_steps == 0:
        print("Warning: empty graph; nothing to plot.")
        return

    x = np.arange(n_steps)
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

    measured = (
        getattr(profiler, "avg_measured_peak_memory", None)
        or getattr(profiler, "avg_measured_memory", None)
    )
    if measured:
        meas_mb = np.array(measured[:n_steps], dtype=float) / (1024 ** 2)
        ax.plot(x[:len(meas_mb)], meas_mb, color="red", linewidth=1.0,
                alpha=0.85,
                label=f"Measured node peak {meas_mb.max():.1f} MB")

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
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Memory breakdown chart saved to {path}")


# --------------------------------------------------------------------------- #
# Deliverable plots                                                           #
# --------------------------------------------------------------------------- #


def _grouped_bars(rows: List[dict], ylabel: str, path: str, title: str) -> None:
    """Two-bar group per batch size, AC-off vs. AC-on.

    Each row: ``{"batch_size": int, "ac_off": float, "ac_on": float}``.
    """
    rows = sorted(rows, key=lambda r: r["batch_size"])
    bs       = [r["batch_size"] for r in rows]
    ac_off   = [r["ac_off"]     for r in rows]
    ac_on    = [r["ac_on"]      for r in rows]

    x = np.arange(len(bs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, ac_off, width, label="without AC",
                color="#4C72B0")
    b2 = ax.bar(x + width / 2, ac_on,  width, label="with AC",
                color="#55A868")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs])
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_peak_memory_vs_batch(rows: List[dict], path: str, title: str) -> None:
    _grouped_bars(rows, "Peak memory (MB)", path, title)


def plot_latency_vs_batch(rows: List[dict], path: str, title: str) -> None:
    _grouped_bars(rows, "Iteration latency (ms)", path, title)


# --------------------------------------------------------------------------- #
# Phase 1 deliverable: peak vs batch size, single series (no AC)              #
# --------------------------------------------------------------------------- #


def plot_peak_vs_batch(rows: List[dict], path: str, title: str,
                       value_key: str = "peak_mb",
                       ylabel: str = "Peak memory (MB)") -> None:
    """Single-series bar chart: one bar per batch size.

    Each row: ``{"batch_size": int, value_key: float}``.  Used for the
    Phase 1 deliverable (peak memory vs batch size, no AC).
    """
    rows = sorted(rows, key=lambda r: r["batch_size"])
    bs   = [r["batch_size"]   for r in rows]
    vals = [r[value_key]      for r in rows]

    x = np.arange(len(bs))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, vals, width=0.55, color="#4C72B0")
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
