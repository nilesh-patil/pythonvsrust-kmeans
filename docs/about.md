---
title: About
---

<div class="site-content" markdown="1">

<span class="eyebrow">About</span>

# About this project

<p style="font-size: 1.15rem; color: var(--charcoal); max-width: 60ch;">
<code>pythonvsrust-kmeans</code> started as a single question: <em>how much speed do you actually gain by porting a textbook K-Means from Python to Rust, and how much does that gain hold up against scikit-learn's industrial-strength C-backed implementation?</em>
</p>

## What's in the repo

<div class="feature-grid">
  <div class="card">
    <h5>Pure-NumPy K-Means</h5>
    <p><a href="https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/python_impl/kmeans.py">src/python_impl/kmeans.py</a> — the readable reference.</p>
  </div>
  <div class="card">
    <h5>Rust K-Means</h5>
    <p><a href="https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/rust_impl">src/rust_impl/</a> — single-threaded CLI binary, with opt-in Rayon parallel path.</p>
  </div>
  <div class="card">
    <h5>sklearn wrapper</h5>
    <p><a href="https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/sklearn_impl/kmeans.py">src/sklearn_impl/kmeans.py</a> — thin wrapper around <code>sklearn.cluster.KMeans</code>.</p>
  </div>
  <div class="card">
    <h5>WASM K-Means</h5>
    <p><a href="https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/wasm_impl">src/wasm_impl/</a> — separate Cargo crate compiled to a 26.5 KB browser module.</p>
  </div>
</div>

## Reproducing the site locally

```bash
# Python + Rust toolchains via Pixi
pixi install
pixi run build-rust

# Run the fresh paired benchmark (writes results/benchmark_results_*.csv and labels)
pixi run python src/run_current_benchmark_suite.py \
  --output results/benchmark_results_20260609_112255.csv

# Refresh derived artifacts
pixi run python src/visualize_init_comparison.py
pixi run python src/visualize_speedup_curve.py
pixi run python src/visualize_memory_breakdown.py
pixi run python src/visualize_quality_runtime_pareto.py
pixi run python src/bench_parallel_scaling.py \
  --n_sample_grid 1000,2000,4000,8000,16000,32000,64000,128000,256000 \
  --n_features 32 --n_clusters 32 --k_max 32 --runs 3
pixi run python src/visualize_parallel_scaling.py
pixi run python src/animate_convergence.py
pixi run python src/build_dashboard.py --input results/benchmark_results_20260609_112255.csv
pixi run python src/sync_assets.py

# Build the WASM module
cd src/wasm_impl && wasm-pack build --target web --out-dir ../../docs/wasm

# Run the test suite
pixi run test
```

The site itself is served from the `docs/` directory by GitHub Pages — no CI build step is required.

## Tools and libraries

<div class="card-cream" markdown="1">
<ul style="margin: 0; padding-left: 1.2rem;">
  <li><strong>Python</strong>: NumPy, pandas, scikit-learn, matplotlib, Plotly, pytest, psutil.</li>
  <li><strong>Rust</strong>: clap, rand, csv, rayon, wasm-bindgen.</li>
  <li><strong>Site</strong>: Jekyll with the minima theme, Cormorant Garamond + Inter via Google Fonts, vanilla CSS, no JS frameworks.</li>
  <li><strong>Environment</strong>: <a href="https://pixi.sh/">Pixi</a> pins both Python and Rust toolchains.</li>
</ul>
</div>

## Acknowledgements

- Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*.
- The Rust `rayon` and `wasm-bindgen` teams for making the parallel and browser stories painless.
- The design system applied to this site is a homage to the [Mistral AI](https://mistral.ai) visual language.

</div>
