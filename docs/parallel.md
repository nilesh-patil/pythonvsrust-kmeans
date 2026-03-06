---
title: Parallelism
---

<div class="site-content" markdown="1">

<span class="eyebrow">Parallelism</span>

# Adding Rayon to the Rust implementation

Pre-Rayon, the Rust implementation's README advertised "Parallel Processing: Can leverage Rust's fearless concurrency" — but the actual binary was single-threaded. Feature 2 of this project made that real.

## How

Lloyd's hot loop has two data-parallel steps and one essentially serial check:

<div class="feature-grid">
  <div class="card">
    <span class="badge badge-orange">Assignment</span>
    <h4 style="margin-top: 0.6rem;">Trivially parallel</h4>
    <p>Each point independently finds its nearest centroid. Use <code>par_iter()</code>.</p>
  </div>
  <div class="card">
    <span class="badge badge-cream">Update</span>
    <h4 style="margin-top: 0.6rem;">Per-thread accumulator</h4>
    <p>Naïve sharing causes contention; each rayon worker holds a private <code>Vec&lt;(sum, count)&gt;</code> per cluster, then a <code>reduce</code> merges them.</p>
  </div>
  <div class="card">
    <span class="badge badge-dark">Convergence</span>
    <h4 style="margin-top: 0.6rem;">Left serial</h4>
    <p>Comparing label vectors. Cheap; not worth parallelising.</p>
  </div>
</div>

```rust
let labels: Vec<usize> = data
    .par_iter()
    .map(|p| nearest_centroid(p, &centroids))
    .collect();

let merged = data
    .par_iter()
    .zip(labels.par_iter())
    .fold(zero_acc, |mut acc, (p, &c)| { add_to(&mut acc[c], p); acc })
    .reduce(zero_acc, merge);
```

## What it bought us

![parallel scaling](assets/images/parallel_scaling.png)

The current Rust-only scaling sweep uses k-means++ initialization on fresh `n x 32-feature` datasets with `k_max=32`. The sample axis follows the same log2 doubling sequence as the main benchmark suite, from 1k through 256k rows. Each scale is saved as `results/parallel_scaling_n*.csv`; `results/parallel_scaling.csv` mirrors the 32k slice for compatibility with older tooling.

| Samples | Serial runtime | Best threads | Best parallel runtime | Best speedup | 8-thread speedup |
|--------:|---------------:|-------------:|----------------------:|-------------:|-----------------:|
| 1k | 7.27 ms | 1 | 6.67 ms | 1.09× | 0.93× |
| 2k | 11.11 ms | 4 | 9.86 ms | 1.13× | 1.02× |
| 4k | 17.32 ms | 4 | 16.46 ms | 1.05× | 0.93× |
| 8k | 33.13 ms | 2 | 30.16 ms | 1.10× | 1.04× |
| 16k | 59.00 ms | 2 | 51.83 ms | 1.14× | 1.10× |
| 32k | 109.26 ms | 8 | 95.27 ms | 1.15× | 1.15× |
| 64k | 241.72 ms | 4 | 187.97 ms | 1.29× | 1.28× |
| 128k | 425.25 ms | 4 | 360.78 ms | 1.18× | 1.15× |
| 256k | 945.28 ms | 8 | 717.42 ms | 1.32× | 1.32× |

Honest answer: **useful but bounded** speedup. The parallel path is noise or a regression on small workloads, becomes helpful around the larger sample counts, and peaks at 1.32× on the 256k-row slice in this sweep.

## Why not more?

<div class="card-cream-soft" markdown="1">
<ol>
  <li><strong>Lloyd's is serial across iterations.</strong> Each iteration depends on the previous — you parallelise <em>within</em> an iteration, not across.</li>
  <li><strong>The data layout is pointer-chasing.</strong> <code>Vec&lt;DataPoint { id: String, features: Vec&lt;f64&gt; }&gt;</code>. A flat <code>Vec&lt;f64&gt;</code> of length <code>n × d</code> would be cache-friendlier — but it's a bigger refactor than this feature allowed.</li>
  <li><strong>Many small fits dominate.</strong> The CLI runs <code>k = 1..k_max</code>. Most of those fits are tiny and Rayon's per-fit setup eats the gain.</li>
</ol>
<p style="margin: 0;">A future direction: flatten the data layout and run only the <code>k = k_max</code> fit in benchmarks. That should push scaling closer to the ideal line.</p>
</div>

</div>
