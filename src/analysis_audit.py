#!/usr/bin/env python3
"""Generate a facts inventory for the current benchmark suite.

Run from the repo root:
    pixi run python src/analysis_audit.py > specs/analysis_facts.md

The report is derived from one benchmark CSV and is intended to keep docs,
dashboard claims, and quoted numbers tied to the same source artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CSV = RESULTS_DIR / "benchmark_results_20260609_112255.csv"

IMPL_ORDER = ["python", "rust", "rust_parallel", "sklearn"]
DISPLAY_NAMES = {
    "python": "Python",
    "rust": "Rust",
    "rust_parallel": "Rust-Parallel",
    "sklearn": "scikit-learn",
}


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_ints(values: pd.Series | np.ndarray | list[int]) -> str:
    return ", ".join(f"{int(v):,}" for v in values)


def latest_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("benchmark_results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No benchmark_results_*.csv files found in {results_dir}")
    return candidates[0]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wall_time_s" not in out.columns:
        out["wall_time_s"] = out["runtime"]
    if "peak_rss_mb" not in out.columns:
        out["peak_rss_mb"] = out["peak_memory_mb"]
    if "k_max" not in out.columns:
        out["k_max"] = out["n_clusters"]
    if "cpu_time_s" not in out.columns:
        out["cpu_time_s"] = np.nan
    if "effective_cores" not in out.columns:
        out["effective_cores"] = np.nan
    out["k_sweep_sum_k"] = out["k_max"] * (out["k_max"] + 1) / 2
    out["nominal_work_units"] = out["n_samples"] * out["n_features"] * out["k_sweep_sum_k"]
    out["rss_mb_per_1k_samples"] = out["peak_rss_mb"] / (out["n_samples"] / 1000.0)
    out["wall_seconds_per_1k_samples"] = out["wall_time_s"] / (out["n_samples"] / 1000.0)
    return out


def paired_runtime_pivot(df: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "dataset_id",
        "repeat_index",
        "n_samples",
        "n_features",
        "k_max",
        "init",
        "sklearn_n_init",
        "cluster_std",
        "cluster_separation",
    ]
    available_keys = [col for col in key_columns if col in df.columns]
    return (
        df.pivot_table(index=available_keys, columns="implementation", values="wall_time_s", aggfunc="median")
        .dropna(subset=["python"])
        .sort_index()
    )


def loglog_slope(df: pd.DataFrame, impl: str) -> tuple[float, float]:
    trend = (
        df[df["implementation"] == impl]
        .groupby("n_samples")["wall_time_s"]
        .median()
        .reset_index()
        .sort_values("n_samples")
    )
    if len(trend) < 2:
        return float("nan"), float("nan")
    x = np.log2(trend["n_samples"].to_numpy(dtype=float))
    y = np.log2(trend["wall_time_s"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return float(slope), float(r2)


def completeness_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["implementation", "n_samples"], observed=True)
        .size()
        .unstack("n_samples")
        .reindex(index=IMPL_ORDER)
        .sort_index(axis=1)
    )


def print_runtime_section(df: pd.DataFrame) -> None:
    print("## Runtime and paired speedups")
    print()
    print("| Implementation | Median wall time | IQR wall time | Median CPU time | Median effective cores |")
    print("|---|---:|---:|---:|---:|")
    for impl in IMPL_ORDER:
        sub = df[df["implementation"] == impl]
        iqr = sub["wall_time_s"].quantile(0.75) - sub["wall_time_s"].quantile(0.25)
        print(
            f"| {DISPLAY_NAMES[impl]} | {fmt(sub['wall_time_s'].median(), 3)} s | "
            f"{fmt(iqr, 3)} s | {fmt(sub['cpu_time_s'].median(), 3)} s | "
            f"{fmt(sub['effective_cores'].median(), 2)} |"
        )
    print()

    pivot = paired_runtime_pivot(df)
    print("| Comparison | Median paired speedup vs Python | Mean paired speedup vs Python |")
    print("|---|---:|---:|")
    for impl in ["rust", "rust_parallel", "sklearn"]:
        paired = (pivot["python"] / pivot[impl]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"| {DISPLAY_NAMES[impl]} | {fmt(paired.median(), 2)}x | {fmt(paired.mean(), 2)}x |")
    print()


def print_scaling_section(df: pd.DataFrame) -> None:
    print("## Scale factors")
    print()
    sample_sizes = sorted(df["n_samples"].unique())
    smallest = int(sample_sizes[0])
    largest = int(sample_sizes[-1])
    print(f"The sample axis is a log2 doubling sequence from {smallest:,} to {largest:,} rows.")
    print()
    print("| Implementation | Median runtime at smallest n | Median runtime at largest n | Large/small runtime factor | Log-log slope vs n | R2 |")
    print("|---|---:|---:|---:|---:|---:|")
    for impl in IMPL_ORDER:
        sub = df[df["implementation"] == impl]
        small_rt = sub.loc[sub["n_samples"] == smallest, "wall_time_s"].median()
        large_rt = sub.loc[sub["n_samples"] == largest, "wall_time_s"].median()
        slope, r2 = loglog_slope(df, impl)
        print(
            f"| {DISPLAY_NAMES[impl]} | {fmt(small_rt, 3)} s | {fmt(large_rt, 3)} s | "
            f"{fmt(large_rt / small_rt, 1)}x | {fmt(slope, 3)} | {fmt(r2, 3)} |"
        )
    print()


def print_memory_section(df: pd.DataFrame) -> None:
    print("## Memory and resource footprint")
    print()
    print("| Implementation | Median sampled RSS | Median RSS / 1k samples | Median wall sec / 1k samples |")
    print("|---|---:|---:|---:|")
    for impl in IMPL_ORDER:
        sub = df[df["implementation"] == impl]
        print(
            f"| {DISPLAY_NAMES[impl]} | {fmt(sub['peak_rss_mb'].median(), 1)} MB | "
            f"{fmt(sub['rss_mb_per_1k_samples'].median(), 2)} MB | "
            f"{fmt(sub['wall_seconds_per_1k_samples'].median(), 4)} s |"
        )
    print()

    largest_work = df["nominal_work_units"].max()
    largest = (
        df[df["nominal_work_units"] == largest_work]
        .groupby("implementation", observed=True)
        .agg(
            median_wall=("wall_time_s", "median"),
            median_rss=("peak_rss_mb", "median"),
            median_cpu=("cpu_time_s", "median"),
            median_cores=("effective_cores", "median"),
        )
        .reindex(IMPL_ORDER)
    )
    print(f"Largest matched workload: nominal work {int(largest_work):,}.")
    print()
    print("| Implementation | Median wall time | Median sampled RSS | Median CPU time | Median effective cores |")
    print("|---|---:|---:|---:|---:|")
    for impl, row in largest.iterrows():
        print(
            f"| {DISPLAY_NAMES[impl]} | {fmt(row['median_wall'], 2)} s | "
            f"{fmt(row['median_rss'], 0)} MB | {fmt(row['median_cpu'], 2)} s | "
            f"{fmt(row['median_cores'], 2)} |"
        )
    print()


def print_quality_section(df: pd.DataFrame) -> None:
    print("## Clustering quality")
    print()
    print("| Implementation | Median ARI | Mean ARI | Min ARI | Mean NMI | ARI >= 0.999 runs |")
    print("|---|---:|---:|---:|---:|---:|")
    for impl in IMPL_ORDER:
        sub = df[df["implementation"] == impl]
        perfect = int((sub["adjusted_rand_index"] >= 0.999).sum())
        print(
            f"| {DISPLAY_NAMES[impl]} | {fmt(sub['adjusted_rand_index'].median(), 3)} | "
            f"{fmt(sub['adjusted_rand_index'].mean(), 3)} | "
            f"{fmt(sub['adjusted_rand_index'].min(), 3)} | "
            f"{fmt(sub['normalized_mutual_info'].mean(), 3)} | {perfect}/{len(sub)} |"
        )
    print()

    print("| Implementation | Mean silhouette | Mean Davies-Bouldin | Mean Calinski-Harabasz |")
    print("|---|---:|---:|---:|")
    for impl in IMPL_ORDER:
        sub = df[df["implementation"] == impl]
        print(
            f"| {DISPLAY_NAMES[impl]} | {fmt(sub['silhouette_score'].mean(), 3)} | "
            f"{fmt(sub['davies_bouldin_index'].mean(), 3)} | "
            f"{fmt(sub['calinski_harabasz_index'].mean(), 1)} |"
        )
    print()


def print_completeness_section(df: pd.DataFrame, csv_path: Path) -> None:
    sample_sizes = sorted(int(v) for v in df["n_samples"].unique())
    feature_counts = sorted(int(v) for v in df["n_features"].unique())
    k_values = sorted(int(v) for v in df["k_max"].unique())
    repeats = sorted(int(v) for v in df["repeat_index"].unique()) if "repeat_index" in df.columns else []

    print("# K-Means analysis facts inventory")
    print()
    print(f"Source CSV: `results/{csv_path.name}`")
    print()
    print("All numbers below are generated by `src/analysis_audit.py` from the current benchmark CSV.")
    print("No historical benchmark snapshots are mixed into this report.")
    print()
    print("## Benchmark grid")
    print()
    print(f"- Rows: {len(df):,}")
    print(f"- Implementations: {', '.join(DISPLAY_NAMES[impl] for impl in IMPL_ORDER)}")
    print(f"- Sample sizes: {fmt_ints(sample_sizes)}")
    print(f"- Feature counts: {fmt_ints(feature_counts)}")
    print(f"- k_max values: {fmt_ints(k_values)}")
    if repeats:
        print(f"- Paired repeats: {fmt_ints(repeats)}")
    print(f"- Initialization policy: {df['init'].mode().iat[0]}, sklearn n_init={int(df['sklearn_n_init'].mode().iat[0])}")
    print()
    print("Rows per implementation and sample size:")
    print()
    print("| Implementation | " + " | ".join(f"{n:,}" for n in sample_sizes) + " |")
    print("|---|" + "|".join("---:" for _ in sample_sizes) + "|")
    counts = completeness_table(df)
    for impl in IMPL_ORDER:
        values = [int(counts.loc[impl, n]) for n in sample_sizes]
        print(f"| {DISPLAY_NAMES[impl]} | " + " | ".join(str(v) for v in values) + " |")
    print()
    failed = int((df["exit_code"] != 0).sum()) if "exit_code" in df.columns else 0
    missing_quality = int(df[["adjusted_rand_index", "normalized_mutual_info"]].isna().any(axis=1).sum())
    print(f"Exit-code failures: {failed}. Rows missing ARI/NMI: {missing_quality}.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate current benchmark facts inventory.")
    parser.add_argument("--input", type=Path, default=None, help="Benchmark CSV to audit; defaults to the current suite CSV.")
    args = parser.parse_args()

    csv_path = args.input or DEFAULT_CSV
    if not csv_path.exists():
        csv_path = latest_csv(RESULTS_DIR)
    df = normalize(pd.read_csv(csv_path))

    print_completeness_section(df, csv_path)
    print_runtime_section(df)
    print_scaling_section(df)
    print_memory_section(df)
    print_quality_section(df)

    print("## Caveats")
    print()
    print("- Runtime is end-to-end CLI subprocess wall time, including process launch, CSV read, the full k sweep, and output CSV writing.")
    print("- Memory is sampled process RSS polled during each subprocess; it is not a platform max-RSS measurement.")
    print("- The suite uses single-start k-means++ across all implementations, including scikit-learn `n_init=1`.")
    print("- ARI/NMI compare against generated ground-truth labels. Internal metrics are sampled at 10k rows for the largest workloads.")
    print("- The Rust and Rust-Parallel paths share the same clustering math; their quality rows should match for paired workloads.")


if __name__ == "__main__":
    main()
