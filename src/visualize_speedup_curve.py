#!/usr/bin/env python3
"""Speedup-vs-scale log-log curves from the newest benchmark CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from viz_style import color, display_name, mpl_marker, ordered_implementations
except ImportError:
    from src.viz_style import color, display_name, mpl_marker, ordered_implementations

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "speedup_curve.png"


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
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for impl, series in speedups.items():
        ax.plot(
            series.index,
            series.values,
            marker=mpl_marker(impl),
            markersize=6,
            lw=2,
            color=color(impl),
            label=display_name(impl),
        )

    # Reference line: y = 1 means "same speed as Python".
    ax.axhline(1.0, ls="--", lw=1.2, color="#888888", alpha=0.8, label="Python baseline (1×)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Nominal k-sweep work: n_samples x n_features x sum(k)  (log2)", fontsize=11)
    ax.set_ylabel("Speedup over pure-Python  (log2 scale, higher is better)", fontsize=11)
    subtitle = (
        f"Source: {csv_path.name} ({len(df)} rows) · "
        "end-to-end CLI k-sweep runtime · three paired repeats per workload"
    )
    fig.suptitle("Speedup over pure-Python by matched workload", fontsize=13, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.935,
        subtitle,
        ha="center", va="bottom",
        fontsize=8, color="#666666",
    )

    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
