---
title: Overview
---

<section class="hero-band">
  <div class="hero-inner">
    <span class="hero-eyebrow">A K-Means clustering study</span>
    <h1 class="hero-display">Frontier clustering. In three languages.</h1>
    <p class="hero-subtitle">
      Pure Python, hand-rolled Rust, and scikit-learn — measured on
      runtime, memory, and clustering quality. With a live in-browser
      WebAssembly demo of the Rust implementation.
    </p>
    <div class="hero-actions">
      <a class="btn-dark"      href="{{ '/demo/'       | relative_url }}">Try the live demo</a>
      <a class="btn-secondary" href="{{ '/benchmarks/' | relative_url }}">View benchmarks</a>
    </div>
  </div>
</section>

<div class="site-content">

<span class="eyebrow">At a glance</span>

<div class="stat-row">
  <div class="stat-cell">
    <span class="stat-display">5.2×</span>
    <span class="stat-label">Rust over pure-Python, mean runtime</span>
  </div>
  <div class="stat-cell">
    <span class="stat-display">0.83</span>
    <span class="stat-label">MB / 1k samples — Rust, the memory champion</span>
  </div>
  <div class="stat-cell">
    <span class="stat-display">1.00</span>
    <span class="stat-label">Adjusted Rand Index — scikit-learn perfectly recovers truth</span>
  </div>
  <div class="stat-cell">
    <span class="stat-display">~24KB</span>
    <span class="stat-label">Rust → WebAssembly module shipped in the live demo</span>
  </div>
</div>

<hr>

## What you'll find here

<div class="feature-grid">
  <div class="card-feature">
    <h3><a href="{{ '/algorithms/' | relative_url }}">Algorithms</a></h3>
    <p>k-means++ vs random initialization, animated — including k-means' two classic failure modes (moons and rings).</p>
  </div>
  <div class="card-feature">
    <h3><a href="{{ '/parallel/' | relative_url }}">Parallelism</a></h3>
    <p>Adding Rayon to the Rust implementation: how the parallel update step works, and what it bought us on a 14-core machine.</p>
  </div>
  <div class="card-feature">
    <h3><a href="{{ '/benchmarks/' | relative_url }}">Benchmarks</a></h3>
    <p>Interactive Plotly dashboard across runtime, memory, internal quality (silhouette / Davies-Bouldin) and external quality (ARI / NMI).</p>
  </div>
  <div class="card-feature">
    <h3><a href="{{ '/demo/' | relative_url }}">Live demo</a></h3>
    <p>Run the Rust K-Means in your browser via WebAssembly — six point distributions, step-by-step animation, WASM-vs-JS speed race.</p>
  </div>
</div>

<hr>

## The headline numbers

![Implementation comparison](assets/images/analysis.png)

| Metric                    | <span class="lang-py">Python</span> | <span class="lang-rust">Rust</span> | <span class="lang-rust-par">Rust&nbsp;-&nbsp;Parallel</span> | <span class="lang-sklearn">scikit-learn</span> |
|---------------------------|:---:|:---:|:---:|:---:|
| Mean runtime              | slowest | **fastest** (5.2×) | comparable (parallel overhead at small k) | 3.5× |
| Mean MB / 1k samples      | moderate | **0.83** (lowest) | 1.3 | 47.9 (highest) |
| Silhouette @ k=k_true     | 0.67 | 0.61 | 0.61 | **0.93** |
| Adjusted Rand Index       | 0.74 | 0.66 | 0.66 | **1.00** |

(Numbers are from the latest 48-experiment quick benchmark. See the [Benchmarks](benchmarks/) page for the full sweep.)

<div class="cta-band-cream">
  <h2>The next chapter of K-Means is yours.</h2>
  <p>Open the live demo, generate two-moons data, watch Lloyd's algorithm fail in real time, then try the same data with k-means++ — all in your browser.</p>
  <div class="hero-actions" style="justify-content: center;">
    <a class="btn-primary" href="{{ '/demo/' | relative_url }}">Open the live demo</a>
    <a class="btn-cream"   href="https://github.com/nilesh-patil/pythonvsrust-kmeans">Star on GitHub</a>
  </div>
</div>

</div>
