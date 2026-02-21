# Spec: `kmeans_fit_steps` WASM Export

**Date:** 2026-03-03  
**Branch:** `feature/march-wasm-step-api`  
**Crate:** `src/wasm_impl/`

## Purpose

Expose the full Lloyd's iteration history to JavaScript so the live demo can
animate K-Means convergence step-by-step. The existing `kmeans_fit` returns
only final labels. `kmeans_fit_steps` returns every intermediate state as a
single flat `Vec<f32>` buffer.

## Flat-buffer layout

```
Header (4 floats):
  [iter_count, converged, k, d]

  iter_count  — number of full Lloyd's iterations that ran (f32 cast of usize)
  converged   — 1.0 if the loop exited early (no label changes), 0.0 otherwise
  k           — cluster count (f32 cast)
  d           — feature dimensions (f32 cast)

Snapshots (iter_count + 1 blocks):
  snapshot 0  — centroids after init + initial assignment
  snapshot 1  — centroids + labels after iteration 1
  ...
  snapshot i  — centroids + labels after iteration i

Each snapshot is (k*d + n) floats:
  k*d centroid floats (row-major: centroid 0 first, then centroid 1, ...)
  n   label   floats  (integer cluster index stored as f32; JS uses Math.round)

Total buffer length: 4 + (iter_count + 1) * (k*d + n)
```

## Why flat f32 over a `#[wasm_bindgen]` struct

A struct-based API would require one `JsValue` round-trip per step to read
centroids and labels, adding ~1 µs of overhead per animation frame. The flat
buffer lets JS slice a single `Float32Array` view inside a `requestAnimationFrame`
loop with zero further WASM calls. This matches the existing `kmeans_fit`
plumbing (also a flat buffer across the boundary) and keeps the JS side simple.

## Equivalence requirement

Given identical `(xs, n, d, k, max_iter, seed, use_kpp)`, the final snapshot
of `kmeans_fit_steps` must produce the same centroid positions and label
assignments as `kmeans_fit`. Both functions use the same RNG seed path, the
same init strategy, and the same Lloyd's loop — the only difference is that
`kmeans_fit_steps` records state after each step.

## Acceptance criteria

1. Buffer header fields match expected `iter_count`, convergence flag, `k`, and `d`.
2. Each snapshot contains exactly `k*d` centroid floats followed by `n` label floats.
3. Total buffer length equals `4 + (iter_count + 1) * (k*d + n)`.
4. Final snapshot labels match `kmeans_fit` output byte-for-byte for the same seed.
5. `converged` is `1.0` when the algorithm exits before `max_iter`.
6. `kmeans_fit` remains unchanged and all pre-existing tests continue to pass.
