---
title: Benchmarks
---

<div class="site-content" markdown="1">

<span class="eyebrow">Benchmarks</span>

# Interactive Plotly dashboard

The dashboard below is rendered from `results/benchmark_results_20260609_112255.csv` by [`src/build_dashboard.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/build_dashboard.py). It uses nominal workload units (`n_samples x n_features x sum(k=1..k_max)`) rather than connecting mixed feature/k settings on `n_samples` alone.

<iframe src="{{ '/dashboard/index.html' | relative_url }}"
        style="width:100%;height:780px;"></iframe>

## How the metrics are computed

<div class="feature-grid">
  <div class="card">
    <h5>Runtime</h5>
    <p><code>time.perf_counter</code> around each implementation's CLI subprocess, including startup, CSV loading, the k sweep, and output CSV writing.</p>
  </div>
  <div class="card">
    <h5>Memory</h5>
    <p><code>psutil.Process.memory_info().rss</code> polled every 10 ms while the implementation runs; this is a sampled process-RSS estimate, not a platform max-RSS measurement.</p>
  </div>
  <div class="card">
    <h5>CPU/resource use</h5>
    <p>Child-process CPU time from <code>resource.getrusage</code>, plus effective cores, context switches, sampled RSS, and normalized RSS/CPU per 1 000 samples.</p>
  </div>
  <div class="card">
    <h5>Quality</h5>
    <p>Adjusted Rand Index (ARI) and NMI use full ground-truth labels. Silhouette, Davies-Bouldin, and Calinski-Harabasz use the full dataset through 32k rows and a deterministic 10k-row sample for larger workloads.</p>
  </div>
</div>

## Reproducing the numbers

```bash
pixi run build-rust                  # compile Rust binary in release mode
pixi run python src/run_current_benchmark_suite.py \
  --output results/benchmark_results_20260609_112255.csv
pixi run python src/build_dashboard.py --input results/benchmark_results_20260609_112255.csv
```

The suite writes 648 rows over the log2 sample sequence from 1k through 256k. Each workload is generated from scratch and reused across implementations within the paired repeat. Every sample size uses feature counts 2, 8, and 32 with k_max values 8 and 32 over three paired repeats.

</div>
