# CLAUDE.md

Project guidance for AI assistants and contributors working on `pythonvsrust-kmeans`.

## What this project is

A comparative study of K-Means clustering implementations in pure Python, Rust, and scikit-learn. The repo benchmarks all three on runtime, memory, and clustering quality, then surfaces the results as a Jekyll GitHub Pages site with a live in-browser WebAssembly demo.

## Layout

```
src/
  python_impl/         # NumPy-based reference, ~150 lines
  rust_impl/           # CLI binary + lib (Rayon-parallel; serial path preserved)
  sklearn_impl/        # Thin wrapper around sklearn.cluster.KMeans
  wasm_impl/           # Separate Cargo crate, builds to docs/wasm/ via wasm-pack
  generate_data.py     # Synthetic blob datasets + ground-truth labels
  animate_convergence.py     # Lloyd's-iteration GIFs
  visualize_*.py             # Static PNG charts for the README/site
  build_dashboard.py         # Interactive Plotly dashboard
  sync_assets.py             # Mirror results/* into docs/assets/*
  bench_parallel_scaling.py  # Rayon thread sweep

runner.py              # Orchestrator: subprocesses Python + sklearn + Rust + Rust-Parallel
tests/                 # pytest suite (~20 tests across init, animations, metrics, site, bench)
specs/                 # Spec-Driven Development docs (one per feature)
docs/                  # Jekyll site (served from this dir by GH Pages)
  └── assets/js/demo.js  # Live in-browser demo: 6 distributions, step animation, WASM-vs-JS race
results/               # Benchmark CSVs, PNGs, GIFs, dashboard HTML
data/                  # Generated datasets + _labels.npy (gitignored)
.claude-worktrees/     # Local-only git worktrees (gitignored)
```

## Implementation labels

- **`python`** → "Python" (pure-NumPy reference)
- **`sklearn`** → "scikit-learn"
- **`rust`** → "Rust" (serial Lloyd's via the CLI binary)
- **`rust_parallel`** → "Rust - Parallel" (Rayon-backed; same binary, `--parallel --threads 0`)

The runner emits separate rows for `rust` and `rust_parallel`; the dashboard renders both side by side. Implementation-name → color and display-name maps live in `src/viz_style.py` (`IMPL_COLORS`, `DISPLAY_NAMES`), which `runner.py`, `src/build_dashboard.py`, and every `src/visualize_*.py` script import for cross-artifact consistency.

## Dev workflow

We use **Spec-Driven Development + Test-Driven Development**:

1. Every new feature gets a `specs/feature_N_*.md` describing inputs, outputs, acceptance criteria, and visual deliverables.
2. Write tests first (red phase) and commit them.
3. Implement until green; commit the impl as a separate commit.
4. Update docs (README, sub-READMEs, site pages) in a final commit.
5. Merge to `master` with `--no-ff` so the branch graph stays readable.

## Environment

Pixi pins both Python (3.11) and Rust (1.87) toolchains. Common tasks:

```bash
pixi install                                   # set up environment
pixi run build-rust                            # cargo build --release in src/rust_impl/
pixi run python runner.py --quick              # ~36-experiment benchmark
pixi run test                                  # full pytest run
cd src/rust_impl && cargo test                 # Rust tests
cd src/wasm_impl && wasm-pack build --target web --out-dir ../../docs/wasm
```

## When working on this repo

- The **serial Rust path** must remain bit-identical to before each change — use it as the baseline for any new parallel or optimization work.
- The **default `--init` is `"random"`** to preserve historical benchmark comparability; `k-means++` is opt-in.
- `runner.py` is the single source of truth for benchmark metrics — add new metrics there, not in implementation CLIs.
- `src/sync_assets.py` is idempotent and is what copies new results into `docs/assets/`.
- Pyright import warnings against pixi-managed packages are expected (the IDE doesn't see `.pixi/envs/default/lib/...`). Runtime is fine; ignore them.

## Site

The Jekyll site lives in `docs/` and is built by GitHub Pages directly (no GH Actions step). To preview locally:

```bash
cd docs && bundle exec jekyll serve   # if you have Jekyll installed
# or simply open docs/index.md in a Markdown viewer
```

`docs/_config.yml` pins `baseurl: /pythonvsrust-kmeans` for hosted paths to resolve correctly.
