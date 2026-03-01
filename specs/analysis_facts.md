# K-Means analysis — facts inventory

> Source CSVs:
> - `results/benchmark_results_20250608_153059.csv` (1008 rows, 3 impls: python, sklearn, rust)
> - `results/benchmark_results_20260518_003517.csv` (48 rows, 4 impls incl. rust_parallel)
>
> All numbers produced by `src/analysis_audit.py`. Run `pixi run python src/analysis_audit.py`
> to reproduce. No numbers were hand-estimated.

---

## A. Headline runtime ratios

Mean runtimes from the 2025 CSV (n=336 rows per implementation, the full 8×7×6 sweep):

- Mean runtime, **python**: **36.32 s** (n=336 rows)
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python'`, `runtime.mean()`

- Mean runtime, **sklearn**: **10.49 s** (n=336 rows)
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn'`, `runtime.mean()`

- Mean runtime, **rust**: **7.03 s** (n=336 rows)
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust'`, `runtime.mean()`

Pairwise speed-up ratios (higher = faster wins):

- Rust / Python (mean ratio): **5.169x**
 **Source:** `benchmark_results_20250608_153059.csv`, all rows, `mean(python_runtime) / mean(rust_runtime)`

- sklearn / Python (mean ratio): **3.462x**
 **Source:** `benchmark_results_20250608_153059.csv`, all rows, `mean(python_runtime) / mean(sklearn_runtime)`

- Rust / sklearn (mean ratio): **1.493x**
 **Source:** `benchmark_results_20250608_153059.csv`, all rows, `mean(sklearn_runtime) / mean(rust_runtime)`

Medians (more robust against the heavy tail at high-n, high-k, high-f combinations):

- Median runtime, python: **1.56 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python'`, `runtime.median()`

- Median runtime, sklearn: **3.73 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn'`, `runtime.median()`

- Median runtime, rust: **0.21 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust'`, `runtime.median()`

- Median Rust / Python speed-up: **7.323x**
 **Source:** `benchmark_results_20250608_153059.csv`, all, `median(python) / median(rust)`

### Rust-Parallel vs Rust (2026 CSV only -- smaller grid, n <= 8,000)

- Mean runtime, rust (2026 CSV): **0.06 s** (n=12 rows)
 **Source:** `benchmark_results_20260518_003517.csv`, `implementation=="rust"`, `runtime.mean()`

- Mean runtime, rust_parallel (2026 CSV): **0.10 s** (n=12 rows)
 **Source:** `benchmark_results_20260518_003517.csv`, `implementation=="rust_parallel"`, `runtime.mean()`

- Rust-Parallel / Rust ratio at this grid: **0.554x** (rust_parallel is ~1.8x *slower* than rust at n<=8k)
 **Source:** `benchmark_results_20260518_003517.csv`, both rust variants, `mean(rust_runtime) / mean(rust_parallel_runtime)`

> Note: rust_parallel is slower than rust on this small grid (n <= 8,000) because thread-spawn
> overhead dominates at these dataset sizes.

---

## B. Scaling behaviour

All numbers from the 2025 CSV. Runtime at smallest (n=1,000) and largest (n=128,000) dataset,
plus the OLS slope from `log(runtime) ~ log(n_samples)` (effective complexity exponent).

### Python
- Runtime @ n=1,000: **0.86 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python' & n_samples==1000`, `runtime.mean()`
- Runtime @ n=128,000: **186.74 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python' & n_samples==128000`, `runtime.mean()`
- Scale factor (128k / 1k): **216.8x**
- Log-log OLS slope (effective complexity exponent): **0.709** (R2=0.422)
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python'`, `linregress(log(n_samples), log(runtime)).slope`

### sklearn
- Runtime @ n=1,000: **3.21 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn' & n_samples==1000`, `runtime.mean()`
- Runtime @ n=128,000: **30.06 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn' & n_samples==128000`, `runtime.mean()`
- Scale factor (128k / 1k): **9.4x**
- Log-log OLS slope (effective complexity exponent): **0.286** (R2=0.242)
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn'`, `linregress(log(n_samples), log(runtime)).slope`

### Rust
- Runtime @ n=1,000: **0.05 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust' & n_samples==1000`, `runtime.mean()`
- Runtime @ n=128,000: **38.64 s**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust' & n_samples==128000`, `runtime.mean()`
- Scale factor (128k / 1k): **781.5x**
- Log-log OLS slope (effective complexity exponent): **1.057** (R2=0.571)
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust'`, `linregress(log(n_samples), log(runtime)).slope`

> Rust starts fastest at small n (0.05 s vs python 0.86 s vs sklearn 3.21 s) but scales
> super-linearly (exponent ~1.06), while sklearn's exponent (0.29) reflects BLAS vectorisation.
> At n=128,000 rust (38.64 s) is already slower than sklearn (30.06 s) on the mean.

---

## C. Memory footprint

`peak_memory_mb` is process-level RSS delta. All absolute means and per-1k-sample rates
from the 2025 CSV unless noted.

### 2025 CSV (absolute mean peak memory)

- python: **255.63 MB** absolute mean; **25.18 MB/1k samples**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python'`,
 `peak_memory_mb.mean()` and `(peak_memory_mb / (n_samples/1000)).mean()`

- sklearn: **280.74 MB** absolute mean; **47.88 MB/1k samples**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn'`, same aggregation

- rust: **23.33 MB** absolute mean; **0.83 MB/1k samples**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust'`, same aggregation

Memory ranges (min -- max across all configs, 2025 CSV):

- python: **67.92 -- 2,361.45 MB**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='python'`, `peak_memory_mb.min() / .max()`
- sklearn: **161.62 -- 1,075.39 MB**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='sklearn'`, same
- rust: **0.03 -- 234.47 MB**
 **Source:** `benchmark_results_20250608_153059.csv`, `implementation=='rust'`, same

Cross-implementation ratios:

- sklearn uses **12.03x** more peak memory than Rust (absolute mean)
 **Source:** `benchmark_results_20250608_153059.csv`, all, `mean(sklearn mb) / mean(rust mb)`
- Python uses **10.96x** more peak memory than Rust (absolute mean)
 **Source:** `benchmark_results_20250608_153059.csv`, all, `mean(python mb) / mean(rust mb)`

### 2026 CSV (all 4 impls, smaller grid)

- python: **81.56 MB** absolute mean; **36.77 MB/1k samples**
 **Source:** `benchmark_results_20260518_003517.csv`, `implementation=='python'`
- sklearn: **179.13 MB** absolute mean; **82.28 MB/1k samples**
 **Source:** `benchmark_results_20260518_003517.csv`, `implementation=='sklearn'`
- rust: **3.86 MB** absolute mean; **0.93 MB/1k samples**
 **Source:** `benchmark_results_20260518_003517.csv`, `implementation=='rust'`
- rust_parallel: **6.30 MB** absolute mean; **2.58 MB/1k samples**
 **Source:** `benchmark_results_20260518_003517.csv`, `implementation=='rust_parallel'`

---

## D. Cluster quality -- internal metrics

Silhouette score (higher = better), Davies-Bouldin index (lower = better).

### 2025 CSV (n=336 rows per impl -- primary source)

| Implementation | Silhouette | Davies-Bouldin |
|----------------|-----------|----------------|
| python | **0.64** | **1.83** |
| sklearn | **0.93** | **0.09** |
| rust | **0.62** | **1.96** |

**Source:** `benchmark_results_20250608_153059.csv`, per `implementation`, `mean()` of each column.

Silhouette deltas (2025 CSV):

- sklearn minus python: **+0.297 points**
 **Source:** `benchmark_results_20250608_153059.csv`, `mean(sklearn silhouette) - mean(python silhouette)`
- sklearn minus rust: **+0.316 points**
 **Source:** `benchmark_results_20250608_153059.csv`, `mean(sklearn silhouette) - mean(rust silhouette)`

### 2026 CSV (n=12 rows per impl -- confirmatory)

| Implementation | Silhouette | Davies-Bouldin |
|----------------|-----------|----------------|
| python | **0.67** | **0.96** |
| sklearn | **0.93** | **0.10** |
| rust | **0.60** | **1.08** |
| rust_parallel | **0.60** | **1.08** |

**Source:** `benchmark_results_20260518_003517.csv`, per `implementation`, `mean()` of each column.

---

## E. Cluster quality -- external metrics (ARI / NMI)

Available ONLY from the 2026 CSV (k=8 fixed, n <= 8,000, f <= 8).
ARI = Adjusted Rand Index; NMI = Normalized Mutual Information. Both in [0, 1]; 1.0 = perfect.

| Implementation | Mean ARI | Mean NMI | ARI~=1.0 runs | Min ARI |
|----------------|----------|----------|--------------|---------|
| python | **0.7434** | **0.8947** | 2 / 12 | **0.3831** |
| sklearn | **1.0000** | **1.0000** | 12 / 12 | **1.0000** |
| rust | **0.6624** | **0.8583** | 1 / 12 | **0.3526** |
| rust_parallel | **0.6624** | **0.8583** | 1 / 12 | **0.3526** |

**Source:** `benchmark_results_20260518_003517.csv`, per `implementation`,
`mean(adjusted_rand_index)`, `mean(normalized_mutual_info)`,
`count(adjusted_rand_index >= 0.999)`, `adjusted_rand_index.min()`.

- **Only sklearn achieves ARI ~= 1.0 in all 12 runs** (perfect ground-truth recovery every time).
- Custom implementations (python, rust, rust_parallel) achieve ARI < 0.40 in their worst runs,
 consistent with single-seed random initialisation occasionally landing in poor local optima.

---

## F. Notable two-implementation deltas worth quoting

1. **sklearn uses 12.03x more peak memory than Rust** (mean absolute, 2025 CSV)
 **Source:** `benchmark_results_20250608_153059.csv`, `mean(sklearn peak_mb) / mean(rust peak_mb)`

2. **sklearn silhouette is 0.297 points higher than Python** (2025 CSV)
 **Source:** `benchmark_results_20250608_153059.csv`, `mean(sklearn silhouette) - mean(python silhouette)`

3. **At n=128,000 samples, Rust is 4.83x faster than Python** (despite higher complexity exponent)
 **Source:** `benchmark_results_20250608_153059.csv`, `n_samples==128000`,
 `mean(python runtime) / mean(rust runtime)`

4. **At n=8,000 (2026 CSV), rust_parallel is 1.56x slower than rust**
 Thread-spawn overhead dominates at n <= 8k; parallel benefit expected only at larger n.
 **Source:** `benchmark_results_20260518_003517.csv`, `n_samples==8000`,
 `mean(rust runtime) / mean(rust_parallel runtime)`

5. **sklearn Davies-Bouldin = 0.09 vs Python 1.83 and Rust 1.96** (2025 CSV)
 The ~20x gap in DB index is the sharpest single-number quality contrast in the dataset.
 **Source:** `benchmark_results_20250608_153059.csv`, `mean(davies_bouldin_index)` per impl

6. **Median Rust/Python speed-up is 7.323x vs 5.169x on means** (2025 CSV)
 Mean is pulled down by outlier Rust runs at high-feature, high-cluster configurations;
 median better represents typical behaviour.
 **Source:** `benchmark_results_20250608_153059.csv`, `median(python) / median(rust)`

---

## G. Sample sizes and dimensionalities tested

### 2025 CSV (1008 rows = 3 impls x 8 x 7 x 6)
- **n_samples:** [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000] -- 8 levels
- **n_features:** [2, 4, 8, 16, 32, 64, 128] -- 7 levels
- **n_clusters:** [2, 4, 8, 16, 32, 64] -- 6 levels
- **implementations:** python, sklearn, rust

**Source:** `benchmark_results_20250608_153059.csv`, `df['col'].unique().tolist()` per dimension.

### 2026 CSV (48 rows = 4 impls x 4 x 3 x 1)
- **n_samples:** [1000, 2000, 4000, 8000] -- 4 levels
- **n_features:** [2, 4, 8] -- 3 levels
- **n_clusters:** [8] -- 1 level (k=8 only)
- **implementations:** python, sklearn, rust, rust_parallel

**Source:** `benchmark_results_20260518_003517.csv`, same.

### Additional extremes (2025 CSV)
- Slowest single run: **2,351.29 s** (python, n=128,000, f=128, k=64)
 **Source:** `benchmark_results_20250608_153059.csv`, all rows, `runtime.idxmax()`

---

## H. Caveats

- **Single-seed initialisation for custom impls.** Python and Rust implementations use one random
 seed per run with no multi-start averaging. sklearn defaults to n_init=10 (not a CSV column --
 verify against `src/sklearn_impl/`). This is the primary driver of the ARI and silhouette gap,
 not the underlying algorithm quality.

- **ARI/NMI narrow grid.** External quality metrics (Section E) come from 48 rows only: k=8 fixed,
 n <= 8,000, f <= 8. Generalisation to other cluster counts, higher dimensionalities, or larger n
 is unmeasured.

- **Heavy-tailed runtime distribution.** The 2025 CSV runtime maximum (2,351 s) is 64.7x the
 mean for python, inflating mean-based speed-up ratios. Prefer medians for typical-case claims.

- **Memory is process-level RSS delta, not algorithmic.** The ~20--50 MB Python interpreter
 base and ~5 MB Rust binary base are included in every measurement. At small n (e.g., 1,000
 samples) the base footprint dominates, inflating per-1k-sample rates.

- **rust_parallel measured only at n <= 8,000.** Parallel scaling to n=128,000 is extrapolated,
 not measured. At all measured sizes, rust_parallel is slower than rust (overhead-dominated).

- **2026 CSV is not a strict subset of 2025 CSV.** Different random seed, possibly different
 data generator version. Cross-CSV runtime comparisons (e.g., rust 2025 vs rust 2026) should
 not be used directly; only same-CSV ratios are reliable.

- **Calinski-Harabasz index omitted from headline facts.** Scale-dependent (grows with n);
 cross-configuration comparisons are misleading without normalisation. Values exist in both
 CSVs for single-configuration drill-downs.
