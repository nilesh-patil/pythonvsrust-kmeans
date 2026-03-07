#!/usr/bin/env python3
"""Speedup-vs-scale log-log curves from the newest benchmark CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from viz_style import (
        INK_FAINT,
        apply_mpl_style,
        color,
        display_name,
        end_label,
        mpl_linestyle,
        mpl_marker,
        ordered_implementations,
        si_log_axis,
        style_axes,
    )
except ImportError:
    from src.viz_style import (
        INK_FAINT,
        apply_mpl_style,
        color,
        display_name,
        end_label,
        mpl_linestyle,
        mpl_marker,
        ordered_implementations,
        si_log_axis,
        style_axes,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "speedup_curve.svg"


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
    if "wall_time_s" not in out.columns:
        out["wall_time_s"] = out["runtime"]
    if "k_max" not in out.columns:
        out["k_max"] = out["n_clusters"]
    out["k_sweep_sum_k"] = out["k_max"] * (out["k_max"] + 1) / 2
    out["nominal_work_units"] = out["n_samples"] * out["n_features"] * out["k_sweep_sum_k"]
    return out


def median_runtime_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (nominal workload x implementation) pivot of median runtimes."""
    return (
        df.groupby(["nominal_work_units", "implementation"])["wall_time_s"]
        .median()
        .unstack("implementation")
        .sort_index()
    )


def main() -> None:
    csv_path = latest_csv_with_columns(
        REPO_ROOT / "results",
        {"runtime", "implementation", "n_samples", "n_features", "n_clusters"},
    )
    df = add_workload_columns(pd.read_csv(csv_path))

    pivot = median_runtime_pivot(df)
    python_baseline = pivot["python"]

    speedups: dict[str, pd.Series] = {}
    for impl in ordered_implementations(pivot.columns):
        if impl != "python":
            speedups[str(impl)] = python_baseline / pivot[impl]

    # --- Plot -----------------------------------------------------------
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Nudge overlapping end-labels apart (Rust and Rust-Parallel converge).
    label_dy = {"rust": -6, "rust_parallel": 8, "sklearn": 0}
    for impl, series in speedups.items():
        ax.plot(
            series.index,
            series.values,
            marker=mpl_marker(impl),
            markersize=5,
            lw=1.8,
            ls=mpl_linestyle(impl),
            color=color(impl),
        )
        # Direct line-end label instead of a legend box.
        end_label(
            ax,
            series.index[-1],
            series.values[-1],
            display_name(impl),
            color(impl),
            dy=label_dy.get(impl, 0),
        )

    # Reference line: y = 1 means "same speed as Python".
    ax.axhline(1.0, ls=(0, (2, 2)), lw=1.0, color=INK_FAINT, alpha=0.7)
    ax.text(
        0.015, 1.0, "Python baseline (1×)",
        transform=ax.get_yaxis_transform(),
        va="bottom", ha="left", fontsize=8, color=INK_FAINT,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    si_log_axis(ax, "both")
    ax.set_xlabel("Nominal k-sweep work (n_samples x n_features x sum k)")
    ax.set_ylabel("Speedup over pure Python (higher is better)")
    subtitle = (
        f"Source: {csv_path.name} ({len(df)} rows) · "
        "end-to-end CLI k-sweep runtime · three paired repeats per workload"
    )
    fig.suptitle("Speedup over pure Python by matched workload", fontsize=11, y=0.98)
    fig.text(
        0.5, 0.925,
        subtitle,
        ha="center", va="bottom",
        fontsize=8, color=INK_FAINT,
    )

    style_axes(ax)
    ax.margins(x=0.12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
