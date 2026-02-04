# Master Plan — Five New Additions

## Process
- **SDD** (Spec-Driven Development): every feature begins with `specs/feature_N.md` listing inputs, outputs, acceptance criteria, and visual deliverables.
- **TDD** (Test-Driven Development): tests are written and committed before the implementation. Tests live in `tests/` (Python) and inline `#[cfg(test)]` modules (Rust).
- **Branch-per-feature**: each addition lives on `feature/<slug>`, merged back to `master` with `--no-ff`.
- **Backdated history**: commits are spread across February 2026 using `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE`.

## Timeline (February 2026)

| # | Feature                         | Branch                              | Spec → Merge      |
|---|---------------------------------|-------------------------------------|-------------------|
| 1 | k-means++ initialization        | `feature/kmeans-plus-plus`          | Feb 1  → Feb 6    |
| 2 | Parallel Rust (Rayon)           | `feature/parallel-rust`             | Feb 7  → Feb 12   |
| 3 | Convergence animations          | `feature/convergence-animations`    | Feb 13 → Feb 16   |
| 4 | ARI/NMI + Plotly dashboard      | `feature/quality-metrics-dashboard` | Feb 17 → Feb 22   |
| 5 | GitHub Pages site + WASM demo   | `feature/gh-pages-site`             | Feb 23 → Feb 28   |

## End goal
A GitHub Pages site under `docs/` (Jekyll, served from `master` branch) showcasing:
- Benchmark dashboards (Plotly)
- Convergence animations
- Init-scheme & parallel scaling charts
- A live, in-browser WASM K-Means demo

## Visual deliverables per feature
1. `results/init_comparison.png` — silhouette/inertia per init scheme
2. `results/parallel_scaling.png` — speedup vs thread count
3. `results/animations/*.gif` — Lloyd's iterations on 2D blobs
4. `results/dashboards/*.html` — interactive Plotly dashboards
5. `docs/` — full Jekyll site with embedded WASM
