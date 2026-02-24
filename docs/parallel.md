---
title: Parallelism
---

<div class="site-content">

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

200 000 points × 32 features × 16 clusters on an Apple M-series, 14 cores:

| Threads | Wall-clock | Speedup vs serial |
|--------:|-----------:|------------------:|
| serial  | 3.40 s | 1.00× |
| 1       | 3.18 s | 1.07× |
| 2       | 2.95 s | 1.15× |
| 4       | 2.84 s | 1.20× |
| 8       | 2.73 s | 1.24× |
| 14      | 2.74 s | 1.24× |

Honest answer: **modest** speedup, plateauing around 4–8 threads.

## Why not more?

<div class="card-cream-soft">
<ol>
  <li><strong>Lloyd's is serial across iterations.</strong> Each iteration depends on the previous — you parallelise <em>within</em> an iteration, not across.</li>
  <li><strong>The data layout is pointer-chasing.</strong> <code>Vec&lt;DataPoint { id: String, features: Vec&lt;f64&gt; }&gt;</code>. A flat <code>Vec&lt;f64&gt;</code> of length <code>n × d</code> would be cache-friendlier — but it's a bigger refactor than this feature allowed.</li>
  <li><strong>Many small fits dominate.</strong> The CLI runs <code>k = 1..k_max</code>. Most of those fits are tiny and Rayon's per-fit setup eats the gain.</li>
</ol>
<p style="margin: 0;">A future direction: flatten the data layout and run only the <code>k = k_max</code> fit in benchmarks. That should push scaling closer to the ideal line.</p>
</div>

</div>
