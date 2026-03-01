"""
analysis_audit.py — Reproducible facts from both benchmark CSVs.

Run from the repo root:
    pixi run python src/analysis_audit.py

Outputs structured text that is the authoritative source for specs/analysis_facts.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file's location → repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
CSV_2025 = REPO_ROOT / "results" / "benchmark_results_20250608_153059.csv"
CSV_2026 = REPO_ROOT / "results" / "benchmark_results_20260518_003517.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df25 = pd.read_csv(CSV_2025)
df26 = pd.read_csv(CSV_2026)

IMPL_ORDER_25 = ["python", "sklearn", "rust"]
IMPL_ORDER_26 = ["python", "sklearn", "rust", "rust_parallel"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def src(csv: str, filt: str, agg: str) -> str:
    return f"Source: {csv}, filter `{filt}`, aggregation `{agg}`"


# ===========================================================================
# SECTION A — Headline runtime ratios (2025 CSV primary, 2026 for parallel)
# ===========================================================================
print("=" * 70)
print("## A. Headline runtime ratios")
print("=" * 70)

means_25 = {impl: df25.loc[df25["implementation"] == impl, "runtime"].mean()
            for impl in IMPL_ORDER_25}
counts_25 = {impl: (df25["implementation"] == impl).sum() for impl in IMPL_ORDER_25}

for impl in IMPL_ORDER_25:
    n = counts_25[impl]
    m = means_25[impl]
    print(f"- Mean runtime, {impl}: **{fmt(m)} s** (n={n} rows, 2025 CSV)")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', 'runtime.mean()')}")

py_m  = means_25["python"]
sk_m  = means_25["sklearn"]
ru_m  = means_25["rust"]

print()
print(f"- Rust / Python speed-up (mean ratio): **{fmt(py_m / ru_m, 3)}×**")
print(f"  {src(CSV_2025.name, 'all', 'mean(python_runtime) / mean(rust_runtime)')}")

print(f"- sklearn / Python speed-up (mean ratio): **{fmt(py_m / sk_m, 3)}×**")
print(f"  {src(CSV_2025.name, 'all', 'mean(python_runtime) / mean(sklearn_runtime)')}")

print(f"- Rust / sklearn speed-up (mean ratio): **{fmt(sk_m / ru_m, 3)}×**")
print(f"  {src(CSV_2025.name, 'all', 'mean(sklearn_runtime) / mean(rust_runtime)')}")

# Rust-Parallel from 2026 CSV — compare to rust on the same grid
rust26    = df26.loc[df26["implementation"] == "rust",          "runtime"].mean()
rp26      = df26.loc[df26["implementation"] == "rust_parallel", "runtime"].mean()
n_rust26  = (df26["implementation"] == "rust").sum()
n_rp26    = (df26["implementation"] == "rust_parallel").sum()

print()
impl_rust_filt = 'implementation=="rust"'
impl_rp_filt   = 'implementation=="rust_parallel"'
print(f"- Mean runtime, rust (2026 CSV):          **{fmt(rust26)} s** (n={n_rust26} rows)")
print(f"  {src(CSV_2026.name, impl_rust_filt, 'runtime.mean()')}")
print(f"- Mean runtime, rust_parallel (2026 CSV): **{fmt(rp26)} s** (n={n_rp26} rows)")
print(f"  {src(CSV_2026.name, impl_rp_filt, 'runtime.mean()')}")
print(f"- Rust-Parallel / Rust speed-up: **{fmt(rust26 / rp26, 3)}×**")
print(f"  {src(CSV_2026.name, 'both rust variants', 'mean(rust_runtime) / mean(rust_parallel_runtime)')}")

# Absolute median for robustness callout
medians_25 = {impl: df25.loc[df25["implementation"] == impl, "runtime"].median()
              for impl in IMPL_ORDER_25}
print()
for impl in IMPL_ORDER_25:
    print(f"- Median runtime, {impl}: **{fmt(medians_25[impl])} s**")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', 'runtime.median()')}")


# ===========================================================================
# SECTION B — Scaling behaviour
# ===========================================================================
print()
print("=" * 70)
print("## B. Scaling behaviour")
print("=" * 70)

smallest = df25["n_samples"].min()   # 1 000
largest  = df25["n_samples"].max()   # 128 000

for impl in IMPL_ORDER_25:
    sub = df25[df25["implementation"] == impl]
    rt_small = sub.loc[sub["n_samples"] == smallest, "runtime"].mean()
    rt_large = sub.loc[sub["n_samples"] == largest,  "runtime"].mean()
    scale_factor = rt_large / rt_small
    print(f"\n### {impl}")
    print(f"- Runtime @ n={smallest:,}:   **{fmt(rt_small)} s**")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r} & n_samples=={smallest}', 'runtime.mean()')}")
    print(f"- Runtime @ n={largest:,}: **{fmt(rt_large)} s**")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r} & n_samples=={largest}', 'runtime.mean()')}")
    print(f"- Scale factor (large / small): **{fmt(scale_factor, 1)}×**")

    # OLS slope of log(runtime) ~ log(n_samples)
    log_n = np.log(sub["n_samples"].values.astype(float))
    log_t = np.log(sub["runtime"].values.astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, r, p, se = stats.linregress(log_n, log_t)
    print(f"- Log-log OLS slope (complexity exponent): **{fmt(slope, 3)}** (R²={fmt(r**2, 3)})")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', 'linregress(log(n_samples), log(runtime)).slope')}")


# ===========================================================================
# SECTION C — Memory footprint
# ===========================================================================
print()
print("=" * 70)
print("## C. Memory footprint")
print("=" * 70)

df25["mb_per_1k"] = df25["peak_memory_mb"] / (df25["n_samples"] / 1000.0)

mem_mean_25 = {impl: df25.loc[df25["implementation"] == impl, "mb_per_1k"].mean()
               for impl in IMPL_ORDER_25}
mem_abs_25  = {impl: df25.loc[df25["implementation"] == impl, "peak_memory_mb"].mean()
               for impl in IMPL_ORDER_25}

for impl in IMPL_ORDER_25:
    print(f"- Mean peak_memory_mb, {impl}: **{fmt(mem_abs_25[impl])} MB** absolute")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', 'peak_memory_mb.mean()')}")
    print(f"- Mean MB per 1k samples, {impl}: **{fmt(mem_mean_25[impl])} MB/1k**")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', '(peak_memory_mb / (n_samples/1000)).mean()')}")
    print()

# Deltas
sk_rust_ratio = mem_abs_25["sklearn"] / mem_abs_25["rust"]
py_rust_ratio = mem_abs_25["python"]  / mem_abs_25["rust"]
print(f"- sklearn uses **{fmt(sk_rust_ratio, 2)}×** more memory than Rust (absolute mean)")
print(f"  {src(CSV_2025.name, 'all', 'mean(sklearn peak_memory_mb) / mean(rust peak_memory_mb)')}")
print(f"- Python uses **{fmt(py_rust_ratio, 2)}×** more memory than Rust (absolute mean)")
print(f"  {src(CSV_2025.name, 'all', 'mean(python peak_memory_mb) / mean(rust peak_memory_mb)')}")

# 2026 CSV memory
df26["mb_per_1k"] = df26["peak_memory_mb"] / (df26["n_samples"] / 1000.0)
mem_mean_26 = {impl: df26.loc[df26["implementation"] == impl, "mb_per_1k"].mean()
               for impl in IMPL_ORDER_26}
mem_abs_26  = {impl: df26.loc[df26["implementation"] == impl, "peak_memory_mb"].mean()
               for impl in IMPL_ORDER_26}
print()
print("  --- 2026 CSV memory (all 4 impls, smaller grid) ---")
for impl in IMPL_ORDER_26:
    print(f"- Mean peak_memory_mb, {impl} (2026): **{fmt(mem_abs_26[impl])} MB**")
    print(f"  {src(CSV_2026.name, f'implementation=={impl!r}', 'peak_memory_mb.mean()')}")
    print(f"- Mean MB per 1k samples, {impl} (2026): **{fmt(mem_mean_26[impl])} MB/1k**")
    print()


# ===========================================================================
# SECTION D — Cluster quality: internal metrics
# ===========================================================================
print()
print("=" * 70)
print("## D. Cluster quality — internal (silhouette, Davies-Bouldin)")
print("=" * 70)

print("\n### From 2025 CSV (n=336 per impl)")
for impl in IMPL_ORDER_25:
    sub = df25[df25["implementation"] == impl]
    sil = sub["silhouette_score"].mean()
    db  = sub["davies_bouldin_index"].mean()
    ch  = sub["calinski_harabasz_index"].mean()
    print(f"- {impl}: silhouette={fmt(sil)}, Davies-Bouldin={fmt(db)}, Calinski-Harabasz={fmt(ch, 1)}")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', 'mean of each metric')}")

print("\n### From 2026 CSV (n=12 per impl)")
for impl in IMPL_ORDER_26:
    sub = df26[df26["implementation"] == impl]
    sil = sub["silhouette_score"].mean()
    db  = sub["davies_bouldin_index"].mean()
    print(f"- {impl}: silhouette={fmt(sil)}, Davies-Bouldin={fmt(db)}")
    print(f"  {src(CSV_2026.name, f'implementation=={impl!r}', 'mean of each metric')}")

# Python silhouette delta vs sklearn
py_sil_25  = df25.loc[df25["implementation"] == "python",  "silhouette_score"].mean()
sk_sil_25  = df25.loc[df25["implementation"] == "sklearn", "silhouette_score"].mean()
ru_sil_25  = df25.loc[df25["implementation"] == "rust",    "silhouette_score"].mean()
print()
print(f"- Silhouette delta (sklearn − python): **{fmt(sk_sil_25 - py_sil_25, 3)}** points (2025 CSV)")
print(f"- Silhouette delta (sklearn − rust):   **{fmt(sk_sil_25 - ru_sil_25, 3)}** points (2025 CSV)")


# ===========================================================================
# SECTION E — Cluster quality: external (ARI / NMI) — 2026 CSV only
# ===========================================================================
print()
print("=" * 70)
print("## E. Cluster quality — external (ARI / NMI) — 2026 CSV only")
print("=" * 70)

for impl in IMPL_ORDER_26:
    sub = df26[df26["implementation"] == impl]
    ari = sub["adjusted_rand_index"].mean()
    nmi = sub["normalized_mutual_info"].mean()
    perfect = (sub["adjusted_rand_index"] >= 0.999).sum()
    total   = len(sub)
    print(f"- {impl}: mean ARI={fmt(ari, 4)}, mean NMI={fmt(nmi, 4)}, "
          f"ARI≈1.0 in {perfect}/{total} runs")
    print(f"  {src(CSV_2026.name, f'implementation=={impl!r}', 'mean(ARI), mean(NMI), count(ARI>=0.999)')}")

# Which impls always hit ARI=1.0
perfect_impls = [impl for impl in IMPL_ORDER_26
                 if (df26.loc[df26["implementation"] == impl, "adjusted_rand_index"] >= 0.999).all()]
print()
print(f"- Impls with ARI≈1.0 in ALL runs: {perfect_impls}")
print(f"  {src(CSV_2026.name, 'all', 'adjusted_rand_index >= 0.999 for every row per impl')}")

# Min ARI per impl
print()
for impl in IMPL_ORDER_26:
    min_ari = df26.loc[df26["implementation"] == impl, "adjusted_rand_index"].min()
    print(f"- {impl} min ARI: **{fmt(min_ari, 4)}**")
    print(f"  {src(CSV_2026.name, f'implementation=={impl!r}', 'adjusted_rand_index.min()')}")


# ===========================================================================
# SECTION F — Two-implementation deltas worth quoting
# ===========================================================================
print()
print("=" * 70)
print("## F. Notable two-implementation deltas")
print("=" * 70)

# 1. Memory: sklearn vs Rust
print(f"1. sklearn uses **{fmt(sk_rust_ratio, 2)}×** more peak memory than Rust (mean absolute, 2025 CSV)")

# 2. Silhouette: sklearn vs python gap
print(f"2. sklearn silhouette is **{fmt(sk_sil_25 - py_sil_25, 3)}** points higher than Python (2025 CSV)")

# 3. Rust speedup over Python at the largest dataset
sub_py_large = df25[(df25["implementation"] == "python") & (df25["n_samples"] == 128000)]["runtime"].mean()
sub_ru_large = df25[(df25["implementation"] == "rust")   & (df25["n_samples"] == 128000)]["runtime"].mean()
print(f"3. At n=128,000 samples, Rust is **{fmt(sub_py_large / sub_ru_large, 2)}×** faster than Python")
print(f"   {src(CSV_2025.name, 'n_samples==128000', 'mean(python) / mean(rust)')}")

# 4. rust_parallel vs rust at largest grid point in 2026 CSV
largest_26 = df26["n_samples"].max()
rp_large26 = df26[(df26["implementation"] == "rust_parallel") & (df26["n_samples"] == largest_26)]["runtime"].mean()
ru_large26 = df26[(df26["implementation"] == "rust")          & (df26["n_samples"] == largest_26)]["runtime"].mean()
print(f"4. At n={largest_26:,} (2026 CSV), rust_parallel is **{fmt(ru_large26 / rp_large26, 2)}×** faster than rust")
print(f"   {src(CSV_2026.name, f'n_samples=={largest_26}', 'mean(rust) / mean(rust_parallel)')}")

# 5. DB index: custom impls vs sklearn
py_db_25 = df25.loc[df25["implementation"] == "python",  "davies_bouldin_index"].mean()
sk_db_25 = df25.loc[df25["implementation"] == "sklearn", "davies_bouldin_index"].mean()
ru_db_25 = df25.loc[df25["implementation"] == "rust",    "davies_bouldin_index"].mean()
print(f"5. sklearn Davies-Bouldin: {fmt(sk_db_25)} vs Python: {fmt(py_db_25)}, Rust: {fmt(ru_db_25)} (lower=better, 2025 CSV)")

# 6. Median runtime ratio (more robust for heavy-tail distribution)
med_py = medians_25["python"]
med_ru = medians_25["rust"]
print(f"6. Median Rust/Python speed-up: **{fmt(med_py / med_ru, 3)}×** (vs {fmt(py_m / ru_m, 3)}× on means; 2025 CSV)")


# ===========================================================================
# SECTION G — Sample sizes and dimensionalities
# ===========================================================================
print()
print("=" * 70)
print("## G. Sample sizes and dimensionalities tested")
print("=" * 70)

n_samples_25  = sorted(df25["n_samples"].unique())
n_features_25 = sorted(df25["n_features"].unique())
n_clusters_25 = sorted(df25["n_clusters"].unique())
n_samples_26  = sorted(df26["n_samples"].unique())
n_features_26 = sorted(df26["n_features"].unique())
n_clusters_26 = sorted(df26["n_clusters"].unique())

n_samples_25_i  = [int(x) for x in n_samples_25]
n_features_25_i = [int(x) for x in n_features_25]
n_clusters_25_i = [int(x) for x in n_clusters_25]
n_samples_26_i  = [int(x) for x in n_samples_26]
n_features_26_i = [int(x) for x in n_features_26]
n_clusters_26_i = [int(x) for x in n_clusters_26]

print(f"### 2025 CSV (1008 rows)")
print(f"- n_samples:  {n_samples_25_i}  ({len(n_samples_25_i)} levels)")
print(f"- n_features: {n_features_25_i}  ({len(n_features_25_i)} levels)")
print(f"- n_clusters: {n_clusters_25_i}  ({len(n_clusters_25_i)} levels)")
print(f"- implementations: {IMPL_ORDER_25}  (3 × 8 × 7 × 6 = 1008 rows)")

print(f"\n### 2026 CSV (48 rows)")
print(f"- n_samples:  {n_samples_26_i}  ({len(n_samples_26_i)} levels)")
print(f"- n_features: {n_features_26_i}  ({len(n_features_26_i)} levels)")
print(f"- n_clusters: {n_clusters_26_i}  ({len(n_clusters_26_i)} levels)")
print(f"- implementations: {IMPL_ORDER_26}  (4 × 4 × 3 × 1 = 48 rows)")


# ===========================================================================
# SECTION H — Additional stats for richness
# ===========================================================================
print()
print("=" * 70)
print("## H. Additional stats")
print("=" * 70)

# Max runtime in 2025
max_row = df25.loc[df25["runtime"].idxmax()]
print(f"- Slowest single run (2025): {fmt(max_row['runtime'])} s, "
      f"impl={max_row['implementation']}, n={max_row['n_samples']}, "
      f"f={max_row['n_features']}, k={max_row['n_clusters']}")
print(f"  {src(CSV_2025.name, 'all rows', 'runtime.idxmax()')}")

# Min memory per impl
for impl in IMPL_ORDER_25:
    min_mem = df25.loc[df25["implementation"] == impl, "peak_memory_mb"].min()
    max_mem = df25.loc[df25["implementation"] == impl, "peak_memory_mb"].max()
    print(f"- {impl} memory range (2025): {fmt(min_mem)} – {fmt(max_mem)} MB")
    print(f"  {src(CSV_2025.name, f'implementation=={impl!r}', 'peak_memory_mb.min() / .max()')}")


# ===========================================================================
# SECTION I — Caveats summary
# ===========================================================================
print()
print("=" * 70)
print("## I. Caveats")
print("=" * 70)

# Count n_init values if present — not in 2025 CSV columns
# Check via stdout if available
print("- n_init is NOT a column in either CSV; cannot confirm n_init=10 for sklearn from data alone.")
print("  Runner script defaults should be verified against src/sklearn_impl/.")
print("- Custom Python/Rust impls use a single random seed per run (no multi-init averaging),")
print("  which inflates variance and can depress silhouette vs sklearn.")
print("- ARI/NMI are only available in the 2026 CSV (48 rows, narrow grid: n≤8k, f≤8, k=8 only).")
print("  Generalisability to other cluster counts is unknown.")
print("- The 2025 CSV runtime distribution is heavy-tailed (max/mean ratio ≫ 1 for python/sklearn),")
print("  so mean speedup ratios are sensitive to a handful of large-n, high-k, high-f runs.")
print("- peak_memory_mb is process-level RSS delta; it does not isolate algorithm overhead from")
print("  interpreter/runtime base footprint, which is ~20–50 MB for Python and ~5 MB for Rust.")
print("- rust_parallel exists only in the 2026 CSV (n≤8k), so parallel scaling to n=128k")
print("  is extrapolated, not measured.")
print("- Calinski-Harabasz index is not reported in analysis_facts.md because it is scale-dependent")
print("  and less interpretable without normalisation across configurations.")


# ===========================================================================
# SANITY CHECK — row counts
# ===========================================================================
print()
print("=" * 70)
print("## Sanity checks")
print("=" * 70)
for impl in IMPL_ORDER_25:
    n = (df25["implementation"] == impl).sum()
    print(f"2025 rows for {impl}: {n}")
for impl in IMPL_ORDER_26:
    n = (df26["implementation"] == impl).sum()
    print(f"2026 rows for {impl}: {n}")

print("\nDone.")
