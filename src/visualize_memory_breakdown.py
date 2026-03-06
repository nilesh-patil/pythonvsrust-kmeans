#!/usr/bin/env python3
"""Tufte-style sampled-RSS memory views from the newest benchmark CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from viz_style import (
        color,
        display_name,
        mpl_marker,
        ordered_implementations,
    )
except ImportError:
    from src.viz_style import (
        color,
        display_name,
        mpl_marker,
        ordered_implementations,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "memory_breakdown.png"


def latest_csv_with_columns(results_dir: Path, required_columns: set[str]) -> Path:
    """Return newest benchmark CSV containing every required column."""
    candidates = sorted(
        results_dir.glob("benchmark_results_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        columns = set(pd.read_csv(candidate, nrows=0).columns)
        if required_columns.issubset(columns):
            return candidate
    raise FileNotFoundError(
        f"No benchmark_results_*.csv in {results_dir} contains {sorted(required_columns)}"
    )


def add_workload_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "peak_rss_mb" not in out.columns:
        out["peak_rss_mb"] = out["peak_memory_mb"]
    if "k_max" not in out.columns:
        out["k_max"] = out["n_clusters"]
    out["k_sweep_sum_k"] = out["k_max"] * (out["k_max"] + 1) / 2
    out["nominal_work_units"] = out["n_samples"] * out["n_features"] * out["k_sweep_sum_k"]
    out["rss_mb_per_1k_samples"] = out["peak_rss_mb"] / (out["n_samples"] / 1000.0)
    return out


def main() -> None:
    csv_path = latest_csv_with_columns(
        REPO_ROOT / "results",
        {"peak_memory_mb", "implementation", "n_samples", "n_features", "n_clusters"},
    )
    df = add_workload_columns(pd.read_csv(csv_path))

    impl_order = ordered_implementations(df["implementation"].unique())
    largest_work = df["nominal_work_units"].max()
    dot_data = (
        df[df["nominal_work_units"] == largest_work]
        .groupby("implementation")["rss_mb_per_1k_samples"]
        .median()
        .sort_values()
    )

    # Right panel: median sampled RSS per matched nominal workload.
    line_data = (
        df.groupby(["implementation", "nominal_work_units"])["peak_rss_mb"]
        .median()
        .unstack("implementation")
        .sort_index()
    )

    fig, (ax_dot, ax_line) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ---- Left: dot chart ------------------------------------------------
    impls = list(dot_data.index)
    x = np.arange(len(impls))
    for idx, (impl, val) in enumerate(zip(impls, dot_data.values)):
        ax_dot.scatter(
            idx,
            val,
            s=90,
            marker=mpl_marker(impl),
            color=color(impl),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=display_name(impl),
        )
        ax_dot.text(
            idx,
            val * 1.12,
            f"{val:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_dot.set_xticks(x)
    ax_dot.set_xticklabels([display_name(i) for i in impls], fontsize=10)
    ax_dot.set_yscale("log", base=2)
    ax_dot.set_ylabel("Sampled RSS  (MB / 1k samples, log2)", fontsize=10)
    ax_dot.set_title("RSS per 1 000 samples at largest matched workload\n(lower is better)", fontsize=11)
    ax_dot.grid(axis="y", ls=":", alpha=0.5, zorder=0)

    # ---- Right: log-log line chart --------------------------------------
    for impl in impl_order:
        if impl not in line_data.columns:
            continue
        series = line_data[impl].dropna()
        ax_line.plot(
            series.index,
            series.values,
            marker=mpl_marker(impl),
            markersize=5,
            lw=2,
            color=color(impl),
            label=display_name(impl),
        )

    ax_line.set_xscale("log", base=2)
    ax_line.set_yscale("log", base=2)
    ax_line.set_xlabel("Nominal k-sweep work  (log2)", fontsize=10)
    ax_line.set_ylabel("Sampled RSS  (MB, median, log2)", fontsize=10)
    ax_line.set_title("Sampled RSS vs matched workload\n(log-log)", fontsize=11)
    ax_line.grid(True, which="both", ls=":", alpha=0.5)
    ax_line.legend(fontsize=9)

    # ---- Shared title + subtitle ----------------------------------------
    fig.suptitle("Memory footprint and scaling", fontsize=14, fontweight="bold", y=0.99)
    subtitle = f"Source: {csv_path.name}  ·  {len(df)} rows  ·  process RSS sampled every 10 ms"
    fig.text(0.5, 0.91, subtitle, ha="center", va="bottom", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0, 1, 0.84))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
