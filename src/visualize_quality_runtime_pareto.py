#!/usr/bin/env python3
"""Quality-runtime frontier from the newest benchmark CSV.

Uses the newest benchmark_results_*.csv with ARI/NMI columns.
One marker per matched workload median. Memory is intentionally not encoded
as bubble area; it has a separate chart so small-memory implementations remain
legible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from viz_style import (
        INK_FAINT,
        PAPER,
        apply_mpl_style,
        color,
        display_name,
        mpl_marker,
        ordered_implementations,
        si_log_axis,
        style_axes,
    )
except ImportError:
    from src.viz_style import (
        INK_FAINT,
        PAPER,
        apply_mpl_style,
        color,
        display_name,
        mpl_marker,
        ordered_implementations,
        si_log_axis,
        style_axes,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "quality_runtime_pareto.svg"

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


def main() -> None:
    csv_path = latest_csv_with_columns(
        REPO_ROOT / "results",
        {"adjusted_rand_index", "normalized_mutual_info"},
    )
    df = add_workload_columns(pd.read_csv(csv_path))
    n_runs = len(df)

    # Per matched workload aggregates.
    agg = (
        df.groupby(["implementation", "nominal_work_units", "n_samples", "n_features", "n_clusters"])
        .agg(
            median_runtime=("wall_time_s", "median"),
            median_ari=("adjusted_rand_index", "median"),
        )
        .reset_index()
    )

    # Per-implementation centroid for annotation labels.
    centroid = (
        agg.groupby("implementation")
        .agg(cx=("median_runtime", "median"), cy=("median_ari", "median"))
        .reset_index()
    )

    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6.4))

    for impl in ordered_implementations(agg["implementation"].unique()):
        grp = agg[agg["implementation"] == impl]
        impl_color = color(str(impl))
        ax.scatter(
            grp["median_runtime"],
            grp["median_ari"],
            s=46,
            marker=mpl_marker(str(impl)),
            color=impl_color,
            alpha=0.75,
            edgecolors=PAPER,
            linewidths=0.6,
            label=display_name(str(impl)),
            zorder=3,
        )

    annotation_offsets = {
        "python": (-14, -16, "right", "top"),
        "rust": (-12, -12, "right", "top"),
        "rust_parallel": (12, -12, "left", "top"),
        "sklearn": (-10, -14, "right", "top"),
    }
    for _, row in centroid.iterrows():
        impl = str(row["implementation"])
        impl_color = color(impl)
        label = display_name(impl)
        dx, dy, ha, va = annotation_offsets.get(impl, (0, 10, "center", "bottom"))
        ax.annotate(
            label,
            xy=(row["cx"], row["cy"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9,
            color=impl_color,
            arrowprops=dict(arrowstyle="-", color=impl_color, lw=0.8, alpha=0.5),
        )

    ax.set_xscale("log", base=2)
    si_log_axis(ax, "x")
    ax.set_xlabel("Median runtime (s, lower is better)")
    ax.set_ylabel("Median Adjusted Rand Index (higher is better)")

    subtitle = f"{csv_path.name}  ·  {n_runs} rows  ·  medians by matched workload"
    fig.suptitle(
        "Quality vs runtime frontier (ARI vs runtime)",
        x=0.5,
        y=0.97,
        fontsize=11,
    )
    fig.text(
        0.5, 0.925,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8, color=INK_FAINT,
    )

    # Direct centroid labels stand in for a legend, so no legend box.
    style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
