---
title: Live demo
---

<div class="epigraph" markdown="1">
This is the Rust K-Means compiled to WebAssembly and running in your browser. Nothing you generate leaves your machine; the only network call is the one that fetches the 26.5 KB <code>.wasm</code> file. Generate two-moons data with a random start and watch it fail, then switch to k-means++ and watch it not.
</div>

<span class="newthought">The fastest way</span> to understand where K-Means works and where it gives up is to drive it yourself. Pick a distribution, set the controls, and hit <strong>Run K-Means</strong>: each Lloyd's iteration animates live on the canvas while the inertia chart below tracks the objective falling. A pure-JavaScript K-Means runs on the same data and seed right after, so you can read the WASM-versus-JS speedup directly off the page.

The run I'd try first is two moons with random initialization. The crescents aren't separable by straight Voronoi boundaries, so the centroids bisect both moons and the inertia chart flattens on a confidently wrong answer. Then switch the init to k-means++ and run it again. The seeding doesn't fix the shape problem (nothing does, K-Means draws convex cells), but it's the cleanest way to feel the difference seeding makes on the data where it *can* help.

<div class="demo-grid" markdown="0">
  <!-- LEFT: controls + status -->
  <aside class="demo-controls">
    <label>Distribution:
      <select id="ctl-dist">
        <option value="blobs" selected>Gaussian blobs</option>
        <option value="rings">Concentric rings</option>
        <option value="moons">Two moons</option>
        <option value="anisotropic">Anisotropic blobs</option>
        <option value="uniform">Uniform random</option>
        <option value="spiral">Pinwheel / spiral</option>
      </select>
    </label>
    <label>Points: <span id="lbl-n">400</span>
      <input id="ctl-n" type="range" min="50" max="100000" step="50" value="400">
    </label>
    <label>Clusters (k): <span id="lbl-k">4</span>
      <input id="ctl-k" type="range" min="2" max="8" step="1" value="4">
    </label>
    <label>Init:
      <select id="ctl-init">
        <option value="kpp" selected>k-means++</option>
        <option value="random">random</option>
      </select>
    </label>
    <label>Max iter: <span id="lbl-maxiter">20</span>
      <input id="ctl-maxiter" type="range" min="1" max="1000" step="1" value="20">
    </label>
    <label>Speed (fps): <span id="lbl-speed">3</span>
      <input id="ctl-speed" type="range" min="0.5" max="8" step="0.5" value="3">
    </label>
    <div class="demo-buttons">
      <button id="btn-generate">Generate</button>
      <button id="btn-fit">Run K-Means</button>
      <button id="btn-step" disabled>Step</button>
      <button id="btn-pause" disabled>Pause</button>
      <button id="btn-reset" disabled>Reset</button>
    </div>
    <p id="demo-status">Loading WebAssembly module…</p>
    <p id="race-result" class="race-result"></p>
  </aside>

  <!-- RIGHT: visual + inertia -->
  <section class="demo-visual">
    <div class="demo-canvas-wrap">
      <canvas id="demo-canvas" width="480" height="480"></canvas>
    </div>
    <div class="inertia-wrap">
      <h5>Inertia per iteration</h5>
      <canvas id="inertia-canvas" width="480" height="120"></canvas>
    </div>
  </section>
</div>

<script type="module" src="{{ '/assets/js/demo.js' | relative_url }}"></script>

## The controls

The distributions are sorted roughly from "K-Means handles this" to "K-Means can't." Gaussian and anisotropic blobs are the ideal case and the slightly-off case; the rest are there to break it. Concentric rings and the pinwheel spiral are non-convex, so K-Means slices them the wrong way every time. Two moons is the textbook failure. Uniform random has no structure at all and shows the algorithm inventing clusters that aren't there.

| Distribution | What it tests |
|---|---|
| Gaussian blobs | The ideal case: spherical, well-separated clusters |
| Concentric rings | Non-convex; K-Means splits rings across the wrong axis |
| Two moons | Crescents; the classic centroid-method failure |
| Anisotropic blobs | Elongated clusters; sensitivity to aspect ratio |
| Uniform random | No real structure; watch it over-fit |
| Pinwheel / spiral | Strongly non-convex; dramatic centroid drift |

Points, clusters, max iterations, and speed do what they say. Init switches between k-means++ and uniform random seeding, which is the lever the two-moons experiment turns. Generate makes a fresh dataset without fitting; Run K-Means fits and animates; Step, Pause, and Reset let you walk a single fit one iteration at a time.

## What's running underneath

Run K-Means calls `kmeans_fit_steps(xs, n, d, k, max_iter, seed, use_kpp)`, exported from `kmeans_wasm.js` and built from [`src/wasm_impl/src/lib.rs`](https://github.com/nilesh-patil/bench-kmeans-rust/blob/master/src/wasm_impl/src/lib.rs). It returns one flat `Float32Array` holding a header and a snapshot of every Lloyd's iteration, so the animation is just a replay of states the Rust code already computed. The typed array crosses the JS-to-WASM boundary without a copy, thanks to `wasm-bindgen`'s typed-array support, which is part of why the race comes out the way it does.

The module is built with a single command:

```bash
cd src/wasm_impl
wasm-pack build --target web --out-dir ../../docs/wasm
```

After each run, the pure-JS K-Means fits the same data with the same seed, both are timed with `performance.now()`, and the speedup factor lands above the controls. The inertia chart under the main canvas plots the within-cluster sum of squares per iteration, updating as the animation plays; convergence is the moment that curve goes flat.
