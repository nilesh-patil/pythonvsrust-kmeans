# Spec: Rust-Parallel Benchmark Split

**Date:** 2026-03-02  
**Branch:** feature/march-rust-parallel-bench

## Background

The Rust binary (`src/rust_impl/target/release/rust_impl`) gained `--parallel` support via
Rayon. Previously, `runner.py` invoked the binary once per experiment, labelling the row
`"rust"`. That single row conflates serial and parallel Rust performance and makes it
impossible to compare the two strategies side-by-side in the dashboard.

## Change Description

### New runner method: `run_rust_parallel_impl`

`BenchmarkRunner` gains a second Rust invocation method that mirrors `run_rust_impl` with
two additions to the subprocess command:

```
--parallel        # activates Rayon thread pool
--threads 0       # 0 = use all available cores (Rayon default)
```

The returned metrics dict carries `"implementation": "rust_parallel"` so every downstream
CSV row is distinguishable from the serial `"rust"` row.

### Updated `run_experiment`

`run_experiment` calls `run_rust_parallel_impl` immediately after `run_rust_impl`, reusing
the same dataset file that was already written for the serial run. Both result dicts receive
the same `n_samples` / `n_features` / `n_clusters` values before being appended to the
result list. Each experiment therefore produces **four rows** (python, sklearn, rust,
rust_parallel) instead of three.

### Dashboard additions (`src/build_dashboard.py`)

| Key | Value |
|-----|-------|
| `IMPL_COLORS["rust_parallel"]` | `"#A0522D"` (sienna — a darker rust shade) |
| display name | `"Rust - Parallel"` (via the `DISPLAY_NAMES` map) |

## Acceptance Criteria

1. `BenchmarkRunner.run_rust_parallel_impl` exists and is callable.
2. `IMPL_COLORS` contains `"rust_parallel"`.
3. The dashboard display-name map maps `"rust_parallel"` to `"Rust - Parallel"`.
4. All existing tests continue to pass.
5. The binary-not-found guard is preserved in the new method (same early-return shape).
