# Spec: Live Demo Overhaul (March 2026)

## Goal
Replace the one-shot animated demo (Gaussian blobs only) with a step-by-step animated
K-Means demo that supports multiple data distributions, playback controls, a live inertia
chart, and a WASM-vs-JS timing race.

## Distributions (6 total)

| ID | Name | Description |
|----|------|-------------|
| `blobs` | Gaussian blobs | k tight circular blobs arranged in a ring (existing) |
| `rings` | Concentric rings | 2 concentric annuli; inner/outer assigned alternately per k |
| `moons` | Two moons | Two interleaved crescent shapes (non-convex, hard for K-Means) |
| `anisotropic` | Anisotropic blobs | k elongated, tilted blobs — tests sensitivity to elliptical clusters |
| `uniform` | Uniform random | Points drawn uniformly over [0,1]² — no true cluster structure |
| `spiral` | Pinwheel / spiral | k arms radiating from center with added noise |

All generators take `(n, k, rng)` and return a normalized `Float32Array` of length `2n`
with values in `[0, 1]`. The canvas controller maps `[0,1]` → pixel space.

## Controls

| Control | Type | Range / Options | Default |
|---------|------|-----------------|---------|
| Points (n) | slider | 50–2000, step 50 | 400 |
| Clusters (k) | slider | 2–8, step 1 | 4 |
| Init | select | k-means++, random | k-means++ |
| Distribution | select | 6 options above | blobs |
| Max iterations | slider | 1–50, step 1 | 20 |
| Speed (fps) | slider | 0.5–8, step 0.5 | 3 |
| Generate | button | regenerates points | — |
| Run K-Means | button | starts animation | — |
| Step | button | advance one frame | — |
| Pause / Resume | button | toggle playback | — |
| Reset | button | return to idle | — |

## Animation Loop Strategy

1. `Run K-Means` calls `kmeans_fit_steps(xs, n, 2, k, maxIter, seed, useKpp)` synchronously.
2. The flat buffer is parsed into `snapshots[]` — each entry holds `{centroids: Float32Array(k*2), labels: Int32Array(n)}`.
3. Inertia for each snapshot is computed in JS: for each point, find its assigned centroid and sum squared Euclidean distances.
4. Animation state machine: `idle → playing → paused → done`.
   - `playing` advances `currentFrame` on each RAF tick, respecting `1000/fps` ms throttle.
   - `Step` advances exactly one frame (works in `paused` or `idle-after-load`).
   - `Pause` toggles between `playing` and `paused`.
   - `Reset` cancels RAF, clears state, re-runs `regen()`.
5. Each frame: clear canvas, draw points colored by label, draw centroid X markers.
6. Inertia chart redraws on each frame showing all accumulated inertia values up to current frame.

## WASM-vs-JS Race Design

- A pure-JS `kmeansJs(xs, n, d, k, maxIter, seed, useKpp)` mirrors the WASM algorithm:
  - Same Mulberry32 RNG for reproducibility.
  - k-means++ init: picks first center randomly, then each subsequent center with probability
    proportional to squared distance to nearest existing center.
  - Lloyd's iterations until convergence or `maxIter` reached.
- After each `Run K-Means` click: both WASM and JS run on identical data + seed.
- Timings via `performance.now()` displayed as:
  `WASM: X.X ms · JS: Y.Y ms (Z.Z× faster)`
- The race result is shown in a `.race-result` div below the main canvas.

## Inertia Chart

- Second `<canvas id="inertia-canvas" width="280" height="120">` rendered below the main canvas.
- Plain Canvas 2D API line chart; no external chart library.
- Y-axis auto-scaled to `[0, snapshots[0].inertia * 1.05]` (initial inertia as ceiling).
- X-axis: iteration index 0 … iter_count.
- Redraws every animation frame showing all points through `currentFrame`.
- Axes drawn with tick marks; line in `--accent` blue.

## Files Changed

- `specs/march_demo_overhaul.md` — this spec
- `docs/assets/js/demo.js` — full demo controller (new file, ES module)
- `docs/demo.md` — remove inline `<script>`, add HTML controls, import external module
- `docs/assets/css/style.css` — extend for new controls, inertia canvas, race result
