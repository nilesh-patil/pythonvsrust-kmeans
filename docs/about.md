---
title: About
---

# About this project

`pythonvsrust-kmeans` started as a single question: **how much speed do you actually gain by porting a textbook K-Means from Python to Rust, and how much does that gain hold up against scikit-learn's industrial-strength C-backed implementation?**

The repository contains:

- A pure-NumPy K-Means in [`src/python_impl/kmeans.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/python_impl/kmeans.py).
- A from-scratch Rust port in [`src/rust_impl/`](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/rust_impl), now Rayon-parallelized.
- A thin sklearn wrapper in [`src/sklearn_impl/kmeans.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/sklearn_impl/kmeans.py).
- A WebAssembly build of the Rust implementation in [`src/wasm_impl/`](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/wasm_impl).
- A benchmark harness, an interactive Plotly dashboard, and Lloyd's-iteration animations.

## Reproducing the site

```bash
# Python + Rust toolchains via Pixi
pixi install
pixi run build-rust

# Run the benchmark (writes results/benchmark_results_*.csv and labels)
pixi run python runner.py --quick

# Refresh derived artifacts
pixi run python src/visualize_init_comparison.py
pixi run python src/visualize_parallel_scaling.py
pixi run python src/animate_convergence.py
pixi run python src/build_dashboard.py
pixi run python src/sync_assets.py

# Build the WASM module
cd src/wasm_impl && wasm-pack build --target web --out-dir ../../docs/wasm

# Run the test suite
pixi run test
```

The site itself is served from the `docs/` directory by GitHub Pages — no CI build step is required.

## Tools and libraries

- **Python**: NumPy, pandas, scikit-learn, matplotlib, Plotly, pytest, psutil.
- **Rust**: clap, rand, csv, rayon, wasm-bindgen.
- **Site**: Jekyll with the minima theme, vanilla CSS, no JS frameworks.
- **Environment**: [Pixi](https://pixi.sh/) pins both Python and Rust toolchains for reproducibility.

## Acknowledgements

- Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*.
- The Rust `rayon` and `wasm-bindgen` teams for making the parallel and browser stories painless.
