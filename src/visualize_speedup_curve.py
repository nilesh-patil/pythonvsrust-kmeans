#!/usr/bin/env python3
"""
Speedup-vs-scale log-log curves.

Primary data: benchmark_results_20250608_153059.csv (8 n_samples points).
Augmented with Rust-Parallel from benchmark_results_20260518_003517.csv
wherever the n_samples grids overlap (1k–8k).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_2025 = REPO_ROOT / "results" / "benchmark_results_20250608_153059.csv"
CSV_2026 = REPO_ROOT / "results" / "benchmark_results_20260518_003517.csv"
OUT_PATH = REPO_ROOT / "results" / "speedup_curve.png"

# Unified palette — Python omitted from plot (it's the baseline denominator).
PALETTE: dict[str, str] = {
    "rust": "#CE422B",
    "rust_parallel": "#A0522D",
    "sklearn": "#F7931E",
}
LABELS: dict[str, str] = {
    "rust": "Rust (serial)",
    "rust_parallel": "Rust (Parallel)",
    "sklearn": "scikit-learn",
}


def median_runtime_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (n_samples × implementation) pivot of median runtimes."""
    return (
        df.groupby(["n_samples", "implementation"])["runtime"]
        .median()
        .unstack("implementation")
    )


def main() -> None:
    df25 = pd.read_csv(CSV_2025)
    df26 = pd.read_csv(CSV_2026)

    pivot25 = median_runtime_pivot(df25)
    python_baseline = pivot25["python"]

    # Compute speedup ratios for the two implementations present in 2025 CSV.
    speedup25: dict[str, pd.Series] = {}
    for impl in ("rust", "sklearn"):
        if impl in pivot25.columns:
            speedup25[impl] = python_baseline / pivot25[impl]

    # Rust-Parallel from 2026 CSV — only include n_samples that overlap with
    # the Python baseline from 2025 (both have 1k–8k, so overlap exists).
    pivot26 = median_runtime_pivot(df26)
    overlap = pivot25.index.intersection(pivot26.index)
    rust_par_speedup: pd.Series | None = None
    if "rust_parallel" in pivot26.columns and len(overlap) > 0:
        rust_par_speedup = python_baseline.loc[overlap] / pivot26.loc[overlap, "rust_parallel"]

    # --- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for impl, series in speedup25.items():
        ax.plot(
            series.index,
            series.values,
            marker="o",
            markersize=6,
            lw=2,
            color=PALETTE[impl],
            label=LABELS[impl],
        )

    if rust_par_speedup is not None:
        ax.plot(
            rust_par_speedup.index,
            rust_par_speedup.values,
            marker="s",
            markersize=6,
            lw=2,
            color=PALETTE["rust_parallel"],
            label=LABELS["rust_parallel"],
        )

    # Reference line: y = 1 means "same speed as Python".
    ax.axhline(1.0, ls="--", lw=1.2, color="#888888", alpha=0.8, label="Python baseline (1×)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_samples", fontsize=11)
    ax.set_ylabel("Speedup over pure-Python  (higher is better)", fontsize=11)
    ax.set_title("Speedup over pure-Python by dataset size", fontsize=13, fontweight="bold")

    subtitle = (
        f"Primary: {CSV_2025.name} ({len(df25)} rows) · "
        f"Rust-Parallel: {CSV_2026.name} ({len(df26)} rows)"
    )
    ax.text(
        0.5, 1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="#666666",
    )

    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
