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
    from viz_style import color, display_name, mpl_marker, ordered_implementations
except ImportError:
    from src.viz_style import color, display_name, mpl_marker, ordered_implementations

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "quality_runtime_pareto.png"

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
            edgecolors="white",
            linewidths=0.6,
            label=display_name(str(impl)),
            zorder=3,
        )

    annotation_offsets = {
        "python": (10, 0, "left", "center"),
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
            fontsize=8,
            fontweight="bold",
            color=impl_color,
            arrowprops=dict(arrowstyle="-", color=impl_color, lw=0.8, alpha=0.5),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Median runtime  (s, log2 scale, lower is better)", fontsize=11)
    ax.set_ylabel("Median Adjusted Rand Index  (higher is better)", fontsize=11)

    subtitle = f"{csv_path.name}  ·  {n_runs} rows  ·  medians by matched workload"
    fig.suptitle(
        "Quality–runtime Pareto (ARI vs runtime)",
        x=0.5,
        y=0.965,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.915,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8, color="#666666",
    )

    # Legend for implementations (de-duplicated by matplotlib).
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
        bbox_to_anchor=(0.985, 0.145),
        bbox_transform=fig.transFigure,
        frameon=True,
    )
    ax.add_artist(legend_impl)

    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout(rect=(0, 0, 0.82, 0.88))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
