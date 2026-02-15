# Feature 3 — Lloyd's-iteration convergence animations

## Why
Static benchmark plots tell you *which* implementation is fast, but they don't show *why* random vs k-means++ produces such different inertia. An animation of Lloyd's iterations on 2-D data — centroids visibly chasing the points each iteration — is the most intuitive way to communicate the algorithm. These will be prominent on the GitHub Pages site.

## Design
Use the existing `KMeansClustering` (Python) and intercept the iteration loop to capture centroid history. Two ways to do this:
- (rejected) Modify `kmeans.py` to expose history — pollutes the production class.
- (chosen) **Inline reimplementation** in `src/animate_convergence.py` that mirrors the Lloyd's loop, recording centroids + labels at each step. Keeps the core impl clean.

## Output
- `results/animations/convergence_random.gif`     — k=4, random init, 2-D well-separated blobs
- `results/animations/convergence_kpp.gif`        — same data, k-means++ init (much faster convergence)
- `results/animations/convergence_pathological.gif` — k=4, random init on data where random fails (two centroids stuck in the same blob)

GIF format chosen for trivial embedding in Markdown/Jekyll. MP4 is also produced when ffmpeg is available, but GIF is the source of truth.

## Visual design
- Points colored by current label assignment (4 colors).
- Centroids drawn as large X markers with a thick border.
- A "trail" of past centroid positions fades behind each centroid (alpha 0.15 per step).
- Each frame shows iteration number and current inertia in the title.
- A short pause (~3 frames at the end) holds the converged state before the GIF loops.

## Tests (write first)
`tests/test_animate_convergence.py`:
1. `test_capture_history_matches_serial_kmeans`: the captured final labels and centroids match what `KMeansClustering(..., init="random")` produces on the same seed + data. This proves the animator is true to the algorithm.
2. `test_centroid_history_length`: history length equals `iterations_run + 1` (initial + after each iter).
3. `test_2d_only`: animator rejects data with `n_features != 2` (it would be a contract violation, since the animation is inherently 2-D).

## Acceptance criteria
- `pytest tests/test_animate_convergence.py -v` — all green.
- Three GIFs exist in `results/animations/` and are < 2 MB each.
- README updated with embedded thumbnails of the GIFs.

## Out of scope
- 3-D animation (would need projection choice, distracting).
- Animating the Rust binary (out of scope — Python is the visualization layer).
- Interactive widgets — that's Feature 4's domain.
