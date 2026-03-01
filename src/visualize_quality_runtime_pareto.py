#!/usr/bin/env python3
"""
Quality–runtime Pareto scatter: ARI vs mean runtime, sized by peak memory.

Uses benchmark_results_20260518_003517.csv (the only CSV with ARI columns).
One marker per (implementation × n_samples) cell.  Implementation centroids
are annotated with the display name.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_2026 = REPO_ROOT / "results" / "benchmark_results_20260518_003517.csv"
OUT_PATH = REPO_ROOT / "results" / "quality_runtime_pareto.png"

PALETTE: dict[str, str] = {
    "python": "#3776AB",
    "rust": "#CE422B",
    "rust_parallel": "#A0522D",
    "sklearn": "#F7931E",
}
LABELS: dict[str, str] = {
    "python": "Python",
    "rust": "Rust (serial)",
    "rust_parallel": "Rust (Parallel)",
    "sklearn": "scikit-learn",
}

# Marker scale: size = k * mean_memory_mb.  Tuned so the largest bubble
# (sklearn ~190 MB) lands around 300 pt² and the smallest (rust ~0.03 MB)
# is still visible at ≥ 10 pt².
_MEM_SCALE = 1.5
_MIN_SIZE = 10.0


def bubble_size(mem_mb: float) -> float:
    return max(_MIN_SIZE, mem_mb * _MEM_SCALE)


def main() -> None:
    df = pd.read_csv(CSV_2026)
    n_runs = len(df)

    # Per-(implementation, n_samples) aggregates.
    agg = (
        df.groupby(["implementation", "n_samples"])
        .agg(
            mean_runtime=("runtime", "mean"),
            mean_ari=("adjusted_rand_index", "mean"),
            mean_mem=("peak_memory_mb", "mean"),
        )
        .reset_index()
    )

    # Per-implementation centroid for annotation labels.
    centroid = (
        agg.groupby("implementation")
        .agg(cx=("mean_runtime", "mean"), cy=("mean_ari", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for impl, grp in agg.groupby("implementation"):
        color = PALETTE.get(str(impl), "#999999")
        sizes = grp["mean_mem"].apply(bubble_size)
        ax.scatter(
            grp["mean_runtime"],
            grp["mean_ari"],
            s=sizes,
            color=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            label=LABELS.get(str(impl), str(impl)),
            zorder=3,
        )

    # Annotate centroids — offset upward to avoid marker overlap.
    for _, row in centroid.iterrows():
        impl = str(row["implementation"])
        color = PALETTE.get(impl, "#999999")
        label = LABELS.get(impl, impl)
        ax.annotate(
            label,
            xy=(row["cx"], row["cy"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime  (s, lower is better)", fontsize=11)
    ax.set_ylabel("Mean Adjusted Rand Index  (higher is better)", fontsize=11)
    ax.set_title("Quality–runtime Pareto (ARI vs runtime)", fontsize=13, fontweight="bold")

    subtitle = f"{CSV_2026.name}  ·  {n_runs} runs  ·  marker area ∝ peak memory"
    ax.text(
        0.5, 1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="#666666",
    )

    # Legend for implementations (de-duplicated by matplotlib).
    # Add a separate size-legend for memory scale.
    handles, labels_leg = ax.get_legend_handles_labels()
    # Deduplicate while preserving order.
    seen: dict[str, int] = {}
    unique_handles, unique_labels = [], []
    for h, lb in zip(handles, labels_leg):
        if lb not in seen:
            unique_handles.append(h)
            unique_labels.append(lb)
            seen[lb] = 1

    legend_impl = ax.legend(
        unique_handles,
        unique_labels,
        title="Implementation",
        fontsize=8,
        title_fontsize=8,
        loc="lower right",
    )
    ax.add_artist(legend_impl)

    # Bubble-size guide — three representative memory values.
    guide_mems = [10.0, 50.0, 150.0]
    guide_handles = [
        plt.scatter([], [], s=bubble_size(m), color="#888888", alpha=0.6, label=f"{m:.0f} MB")
        for m in guide_mems
    ]
    ax.legend(
        guide_handles,
        [f"{m:.0f} MB" for m in guide_mems],
        title="Peak memory",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
    )

    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
