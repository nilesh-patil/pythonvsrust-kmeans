---
title: About
---

One clustering algorithm, written four ways, with a harness to measure them against each other. The repository holds those four K-Means implementations and the tooling around them. The pure-NumPy version in [`src/python_impl/kmeans.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/python_impl/kmeans.py) is about 150 lines and exists to be read. The [Rust port](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/rust_impl) is a faithful translation of it, a single CLI binary with an opt-in Rayon parallel path behind a flag. The [scikit-learn wrapper](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/sklearn_impl/kmeans.py) is a thin shim over `sklearn.cluster.KMeans` that stands in for the production answer. And the [WASM crate](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/wasm_impl) compiles the same Rust math to a 26.5 KB browser module, which is what powers the live demo. `runner.py` ties them together: it runs each as a subprocess, times it end to end, and records runtime, memory, CPU, and clustering quality into one CSV.

## Toolchain

Pixi pins both toolchains so the benchmark is reproducible: Rust 1.87 and Python 3.11. On the Python side the work leans on NumPy, pandas, scikit-learn, matplotlib, Plotly, pytest, and psutil. On the Rust side it's clap, rand, csv, rayon, and wasm-bindgen. The site is plain Jekyll served straight from `docs/` by GitHub Pages with no build step, set in Newsreader with JetBrains Mono for code, math rendered by KaTeX, and no JavaScript framework anywhere except the demo's own module.

## What I'd do differently

Two things, both of which the [parallelism page]({{ '/parallel/' | relative_url }}) explains in more detail. The first is the data layout. Storing each row as `DataPoint { id: String, features: Vec<f64> }` means a vector of pointer-chasing structs, and the distance kernel pays for that in cache misses. A flat `n × d` matrix of `f64` would be friendlier to both the prefetcher and to Rayon, and it would probably do more for the parallel speedup than any amount of thread tuning. The second is the benchmark grid: the CLI fits every \\(k\\) from 1 to k_max, so most fits are tiny and Rayon's setup cost dominates them. Measuring a single large fit alongside the sweep would give the parallel path a fair test instead of burying it in startup overhead.

I'd also like to run the seeding ablation I keep gesturing at, comparing single-start against best-of-N restarts, rather than holding everything to one start for fairness. That's a different study, and an honest one would be its own page.

## How to reproduce it

Everything runs through pixi. Install the environment, build the Rust binary, run the suite, and rebuild the derived artifacts:

```bash
pixi install
pixi run build-rust

pixi run python src/run_current_benchmark_suite.py

pixi run python src/build_dashboard.py
pixi run python src/sync_assets.py

cd src/wasm_impl && wasm-pack build --target web --out-dir ../../docs/wasm

pixi run test
```

The full reproduction recipe, including the figure and animation scripts, lives alongside the methodology on the [benchmarks page]({{ '/benchmarks/' | relative_url }}).

## Acknowledgements

The k-means++ seeding and its \\(O(\log k)\\) guarantee are from Arthur and Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (SODA 2007). Thanks to the Rayon and wasm-bindgen teams for making the parallel and in-browser stories nearly painless. Source for everything is on [GitHub](https://github.com/nilesh-patil/pythonvsrust-kmeans).
