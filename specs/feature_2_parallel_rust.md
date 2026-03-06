# Feature 2 — Parallel Rust K-Means (Rayon)

## Why
The Rust README has long advertised "Parallel Processing: Can leverage Rust's fearless concurrency" — but the actual binary is **single-threaded** and `rayon` is not in `Cargo.toml`. This feature makes that claim real and produces a parallel-scaling visualization that is one of the most visually compelling stories in the project (linear speedup on assignment, sub-linear on update).

## Design
- Parallelize Lloyd's hot loop with `rayon`:
  - **Assignment step**: trivially data-parallel — each point independently finds its nearest centroid. Use `par_iter()`.
  - **Update step**: a per-thread accumulator pattern. Each rayon worker accumulates partial sums into its own `Vec<(Vec<f64>, usize)>` (sum, count) per cluster, then reduce across threads. Avoids contention on shared centroid buffers.
- Add `--threads` CLI flag (default `0` = use all available cores via `rayon::current_num_threads()`).
- Add `--parallel` boolean flag (default `false`) so the single-threaded path remains available for fair comparison and to make speedup curves meaningful.

## Rust API additions
- `KMeans::fit_parallel(&[DataPoint])` — separate method, OR a `parallel: bool` field selecting the impl. Prefer field on the struct so `fit()` dispatches; cleaner CLI.
- `assign_clusters_parallel` and `update_centroids_parallel` private methods.
- The parallel update uses `rayon::iter::ParallelIterator::fold` then `reduce` for the per-thread accumulator pattern.

## CLI additions
- `--parallel` — enables parallel path.
- `--threads N` — sets `RAYON_NUM_THREADS` indirectly via `rayon::ThreadPoolBuilder::new().num_threads(N).build_global().unwrap()` when N > 0.

## Tests (write first, `#[cfg(test)]` in lib.rs)
1. `test_parallel_matches_serial_labels`: same seed + same data, serial and parallel paths must produce **identical** labels at the end.
2. `test_parallel_matches_serial_centroids`: centroids within `1e-9` of each other (floating point accumulation order may differ slightly — bound the tolerance).
3. `test_parallel_completes_on_single_thread_pool`: rayon configured with 1 thread still works.

## Benchmark
- New script `src/bench_parallel_scaling.py` that:
  - Generates either one fixed dataset or a comma-separated sample-size scale sweep.
  - Runs the Rust binary with `--parallel --threads {1, 2, 4, 8}` up to the available logical CPU count reported by `os.cpu_count()`.
  - Records wall-clock time via `time.perf_counter`, 3 runs per thread count, takes median.
  - Saves the compatibility slice → `results/parallel_scaling.csv`.
  - For scale sweeps, saves one CSV per sample size → `results/parallel_scaling_n*.csv`.
- New visualization `src/visualize_parallel_scaling.py`:
  - For a single CSV, plots runtime, speedup, and efficiency against thread count.
  - For the scale sweep, plots log2 runtime vs sample count, best speedup vs sample count, and large-workload thread curves.
  - Saves → `results/parallel_scaling.png`.

## Acceptance criteria
- `cargo test` passes including the three new parallel tests.
- `cargo build --release` succeeds; `rayon` is a declared dependency.
- `results/parallel_scaling.csv`, `results/parallel_scaling_n*.csv`, and `results/parallel_scaling.png` exist and document the measured thread-scaling curve without a fixed speedup threshold; measurement variance is real.
- `src/rust_impl/README.md` documents `--parallel` and `--threads` and replaces the aspirational claim with measured numbers.

## Out of scope
- SIMD intrinsics — separate concern.
- Mini-batch K-Means — separate algorithmic direction.
- GPU offload — way out of scope.
