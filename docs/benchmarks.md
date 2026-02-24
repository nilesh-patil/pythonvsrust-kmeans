---
title: Benchmarks
---

<div class="site-content">

<span class="eyebrow">Benchmarks</span>

# Interactive Plotly dashboard

All four metric tabs below are rendered from the latest [`results/benchmark_results_*.csv`](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/results) by [`src/build_dashboard.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/build_dashboard.py).

<iframe src="{{ '/dashboard/index.html' | relative_url }}"
        style="width:100%;height:780px;"></iframe>

## How the metrics are computed

<div class="feature-grid">
  <div class="card">
    <h5>Runtime</h5>
    <p><code>time.perf_counter</code> around the subprocess call to each implementation's CLI. Excludes data loading from the implementation's reported time.</p>
  </div>
  <div class="card">
    <h5>Memory</h5>
    <p><code>psutil.Process.memory_info().rss</code> polled every 10 ms while the implementation runs; we record the peak.</p>
  </div>
  <div class="card">
    <h5>Internal quality</h5>
    <p>Silhouette, Davies-Bouldin, Calinski-Harabasz from <code>sklearn.metrics</code>. Computed at the dataset's true cluster count.</p>
  </div>
  <div class="card">
    <h5>External quality</h5>
    <p>Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), computed against the ground-truth labels saved by <code>src/generate_data.py</code>.</p>
  </div>
</div>

## Reproducing the numbers

```bash
pixi run build-rust                  # compile Rust binary in release mode
pixi run python runner.py --quick    # ~48 experiments (12 × 4 impls), < 1 minute
pixi run python src/build_dashboard.py
```

</div>
