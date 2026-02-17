---
title: Overview
---

# Python vs Rust K-Means

A comparative study of K-Means clustering implementations in **pure Python**, **Rust**, and **scikit-learn** — measured on runtime, memory, and clustering quality.

This site is the live companion to the [GitHub repository](https://github.com/nilesh-patil/pythonvsrust-kmeans). All charts here are generated from the latest benchmark CSV; all animations are produced by the same Lloyd's-iteration tracer that backs the test suite.

## What's here

- **[Algorithms](algorithms/)** — k-means++ vs random initialization, animated.
- **[Parallelism](parallel/)** — adding Rayon to the Rust impl and what it bought us.
- **[Benchmarks](benchmarks/)** — interactive Plotly dashboard across all implementations.
- **[Live demo](demo/)** — run K-Means in your browser via the WASM-compiled Rust implementation.
- **[About](about/)** — project background, design choices, and acknowledgements.

## At a glance

![Implementation comparison](assets/images/analysis.png)

| Metric                | <span class="lang-py">Python</span> | <span class="lang-rust">Rust</span> | <span class="lang-sklearn">sklearn</span> |
|-----------------------|:---:|:---:|:---:|
| Mean runtime (large)  | slowest | **fastest** (5.2× over Python) | 3.5× over Python |
| Mean MB / 1k samples  | moderate | **lowest** (0.83) | 47.9 (highest) |
| Silhouette (k=k_true) | 0.67 | 0.61 | **0.93** |
| Adjusted Rand Index   | 0.74 | 0.66 | **1.00** |

(Numbers come from the most recent 36-run quick benchmark — see [benchmarks](benchmarks/) for the full sweep.)

## Why this project?

K-Means is one of the most widely deployed clustering algorithms. Its performance characteristics, however, depend heavily on language, data layout, init scheme, and number of restarts. This project quantifies the trade-offs across realistic workloads so you can predict which implementation will hold up in your own pipeline.
