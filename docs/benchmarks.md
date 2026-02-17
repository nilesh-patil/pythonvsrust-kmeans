---
title: Benchmarks
---

# Interactive benchmarks

All four metric tabs below are rendered from the latest [`results/benchmark_results_*.csv`](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/results) by [`src/build_dashboard.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/build_dashboard.py).

<iframe src="dashboard/index.html"
        style="width:100%;height:780px;border:1px solid #e5e7eb;border-radius:6px;"></iframe>

## How the metrics are computed

- **Runtime** — `time.perf_counter` around the subprocess call to each implementation's CLI. Excludes data loading from the implementation's reported time.
- **Memory** — `psutil.Process.memory_info().rss` polled every 10 ms while the implementation runs; we record the peak.
- **Silhouette / Davies-Bouldin / Calinski-Harabasz** — internal metrics from `sklearn.metrics`. Computed on the final `cluster_k` column where `k` equals the dataset's true cluster count.
- **Adjusted Rand Index (ARI) / Normalized Mutual Information (NMI)** — external metrics, computed against the ground-truth labels saved by `src/generate_data.py`. ARI ∈ [-1, 1] (1 = perfect recovery); NMI ∈ [0, 1].

## Reproducing the numbers

```bash
pixi run build-rust          # compile Rust binary in release mode
pixi run python runner.py --quick   # ~36 experiments, < 1 minute
pixi run python src/build_dashboard.py
```
