---
title: Live demo
---

<div class="site-content" markdown="1">

<span class="eyebrow">Live demo</span>

# Lloyd's, in your browser.

<p style="font-size: 1.1rem; color: var(--charcoal); max-width: 60ch;">
Watch K-Means converge step by step — the Rust implementation runs entirely in your browser via WebAssembly. No data leaves your machine; the only network call is the one that loads the ~24&nbsp;KB <code>.wasm</code> file.
</p>

<p>Choose a distribution, tune the controls, and hit <strong>Run K-Means</strong> to see each iteration animate live alongside a real-time inertia chart. A pure-JS K-Means runs in parallel so you can see the WASM speedup directly.</p>

<div class="demo-canvas-wrap">
  <canvas id="demo-canvas" width="480" height="480"></canvas>
</div>

<div class="demo-inertia-wrap">
  <p>Inertia per iteration</p>
  <canvas id="inertia-canvas" width="280" height="120"></canvas>
</div>

<p id="race-result" class="race-result"></p>

<div class="demo-controls">
  <label>Points: <span id="lbl-n">400</span>
    <input id="ctl-n" type="range" min="50" max="2000" step="50" value="400">
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
  <label>Max iter: <span id="lbl-maxiter">20</span>
    <input id="ctl-maxiter" type="range" min="1" max="50" step="1" value="20">
  </label>
  <label>Speed (fps): <span id="lbl-speed">3</span>
    <input id="ctl-speed" type="range" min="0.5" max="8" step="0.5" value="3">
  </label>
  <button id="btn-generate">Generate</button>
  <button id="btn-fit">Run K-Means</button>
  <button id="btn-step" disabled>Step</button>
  <button id="btn-pause" disabled>Pause</button>
  <button id="btn-reset" disabled>Reset</button>
</div>

<p id="demo-status">Loading WebAssembly module…</p>

<script type="module" src="{{ '/assets/js/demo.js' | relative_url }}"></script>

## What's running

**Run K-Means** calls `kmeans_fit_steps(xs, n, d, k, max_iter, seed, use_kpp)` exported from `kmeans_wasm.js` (built from [`src/wasm_impl/src/lib.rs`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/wasm_impl/src/lib.rs)). The function returns a flat `Float32Array` containing a header and every Lloyd's-iteration snapshot so the animation can replay each step.

The WASM module is compiled via:

```bash
cd src/wasm_impl
wasm-pack build --target web --out-dir ../../docs/wasm
```

**Distributions** let you explore how K-Means behaves on data it was and was not designed for:

| Distribution | What it tests |
|---|---|
| Gaussian blobs | Ideal case — spherical, well-separated |
| Concentric rings | Non-convex; K-Means splits rings incorrectly |
| Two moons | Crescent shapes; classic centroid-method failure mode |
| Anisotropic blobs | Elongated clusters; sensitivity to aspect ratio |
| Uniform random | No real structure; tests over-fitting tendency |
| Pinwheel / spiral | Highly non-convex; dramatic centroid drift visible |

**WASM-vs-JS race**: after each Run, a pure-JS K-Means implementation runs on the same data and seed. Both are timed with `performance.now()` and the speedup factor is displayed above the controls.

**Inertia chart**: the small canvas below the main plot shows the within-cluster sum-of-squared distances per iteration, updating in real time as the animation plays. Convergence is visible as the curve flattens.

Float32Arrays cross the JS↔WASM boundary without copying thanks to `wasm-bindgen`'s typed-array support.

</div>
