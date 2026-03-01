#!/usr/bin/env python3
"""
Two-panel memory chart:
  Left  — bar chart of MB / 1k samples per implementation (mean), annotated.
  Right — log-log line chart of peak_memory_mb vs n_samples (median per cell).

Uses benchmark_results_20250608_153059.csv (2025, three impls, 8 n_samples).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_2025 = REPO_ROOT / "results" / "benchmark_results_20250608_153059.csv"
OUT_PATH = REPO_ROOT / "results" / "memory_breakdown.png"

PALETTE: dict[str, str] = {
    "python": "#3776AB",
    "rust": "#CE422B",
    "sklearn": "#F7931E",
}
LABELS: dict[str, str] = {
    "python": "Python",
    "rust": "Rust",
    "sklearn": "scikit-learn",
}
# Consistent render order across both panels.
IMPL_ORDER = ["python", "rust", "sklearn"]


def main() -> None:
    df = pd.read_csv(CSV_2025)

    # Left panel: MB per 1k samples = peak_memory_mb / (n_samples / 1000).
    # Use the mean across all rows for each implementation.
    df["mem_per_1k"] = df["peak_memory_mb"] / (df["n_samples"] / 1000.0)
    bar_data = (
        df.groupby("implementation")["mem_per_1k"]
        .mean()
        .reindex(IMPL_ORDER)
        .dropna()
    )

    # Right panel: median peak_memory_mb per (implementation, n_samples).
    line_data = (
        df.groupby(["implementation", "n_samples"])["peak_memory_mb"]
        .median()
        .unstack("implementation")
    )

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ---- Left: bar chart ------------------------------------------------
    impls = list(bar_data.index)
    x = np.arange(len(impls))
    colors = [PALETTE[i] for i in impls]
    bars = ax_bar.bar(x, bar_data.values, color=colors, width=0.55, zorder=3)

    for bar, val in zip(bars, bar_data.values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + bar_data.values.max() * 0.02,
            f"{val:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABELS.get(i, i) for i in impls], fontsize=10)
    ax_bar.set_ylabel("Peak memory  (MB / 1k samples)", fontsize=10)
    ax_bar.set_title("Memory per 1 000 samples\n(lower is better)", fontsize=11)
    ax_bar.grid(axis="y", ls=":", alpha=0.5, zorder=0)

    # ---- Right: log-log line chart --------------------------------------
    for impl in IMPL_ORDER:
        if impl not in line_data.columns:
            continue
        series = line_data[impl].dropna()
        ax_line.plot(
            series.index,
            series.values,
            marker="o",
            markersize=5,
            lw=2,
            color=PALETTE[impl],
            label=LABELS.get(impl, impl),
        )

    ax_line.set_xscale("log")
    ax_line.set_yscale("log")
    ax_line.set_xlabel("n_samples", fontsize=10)
    ax_line.set_ylabel("Peak memory  (MB, median)", fontsize=10)
    ax_line.set_title("Memory scaling vs dataset size\n(log-log)", fontsize=11)
    ax_line.grid(True, which="both", ls=":", alpha=0.5)
    ax_line.legend(fontsize=9)

    # ---- Shared title + subtitle ----------------------------------------
    fig.suptitle("Memory footprint and scaling", fontsize=14, fontweight="bold", y=1.01)
    subtitle = f"Source: {CSV_2025.name}  ·  {len(df)} rows"
    fig.text(0.5, 1.0, subtitle, ha="center", va="bottom", fontsize=8, color="#666666")

    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
