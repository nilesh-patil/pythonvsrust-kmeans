---
title: Live demo
---

# Live K-Means demo (Rust → WebAssembly)

Below is the Rust K-Means implementation, compiled to WebAssembly and running entirely in your browser. No data leaves your machine; the only network call is the one that loads the ~24 KB `.wasm` file.

<div class="demo-canvas-wrap">
  <canvas id="demo-canvas" width="480" height="480"></canvas>
</div>

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
  <button id="btn-generate">Generate blobs</button>
  <button id="btn-fit">Run K-Means</button>
</div>

<p id="demo-status">Loading WebAssembly module…</p>

<script type="module">
  import init, { init_panic_hook, kmeans_fit } from "{{ '/wasm/kmeans_wasm.js' | relative_url }}";

  const canvas  = document.getElementById("demo-canvas");
  const ctx     = canvas.getContext("2d");
  const status  = document.getElementById("demo-status");
  const ctlN    = document.getElementById("ctl-n");
  const ctlK    = document.getElementById("ctl-k");
  const ctlInit = document.getElementById("ctl-init");
  const lblN    = document.getElementById("lbl-n");
  const lblK    = document.getElementById("lbl-k");
  const btnGen  = document.getElementById("btn-generate");
  const btnFit  = document.getElementById("btn-fit");

  const palette = ["#0ea5e9","#dc2626","#16a34a","#a855f7","#f59e0b","#0891b2","#db2777","#65a30d"];

  let points  = null;  // Float32Array length 2n
  let labels  = null;  // Int32Array  length n
  let ready   = false;

  function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function generateBlobs(n, k) {
    const rng = mulberry32(Date.now() & 0xffffffff);
    const xs = new Float32Array(n * 2);
    const centers = [];
    for (let c = 0; c < k; c++) {
      const angle = (2 * Math.PI * c) / k;
      centers.push([0.5 + 0.32 * Math.cos(angle), 0.5 + 0.32 * Math.sin(angle)]);
    }
    for (let i = 0; i < n; i++) {
      const c = centers[i % k];
      xs[2 * i]     = c[0] + 0.04 * gaussian(rng);
      xs[2 * i + 1] = c[1] + 0.04 * gaussian(rng);
    }
    return xs;
  }

  function draw() {
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!points) return;
    const n = points.length / 2;
    for (let i = 0; i < n; i++) {
      const x = points[2 * i]     * canvas.width;
      const y = points[2 * i + 1] * canvas.height;
      ctx.fillStyle = labels ? palette[labels[i] % palette.length] : "#9ca3af";
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function regen() {
    const n = parseInt(ctlN.value, 10);
    const k = parseInt(ctlK.value, 10);
    points = generateBlobs(n, k);
    labels = null;
    draw();
    status.textContent = `${n} points generated. Click "Run K-Means".`;
  }

  function fit() {
    if (!ready || !points) return;
    const n = points.length / 2;
    const k = parseInt(ctlK.value, 10);
    const useKpp = ctlInit.value === "kpp";
    const t0 = performance.now();
    labels = kmeans_fit(points, n, 2, k, 100, (Math.random() * 2 ** 32) >>> 0, useKpp);
    const t1 = performance.now();
    draw();
    status.textContent = `Converged on n=${n}, k=${k}, init=${useKpp ? "k-means++" : "random"} in ${(t1 - t0).toFixed(1)} ms (WASM)`;
  }

  ctlN.addEventListener("input", () => { lblN.textContent = ctlN.value; });
  ctlK.addEventListener("input", () => { lblK.textContent = ctlK.value; });
  btnGen.addEventListener("click", regen);
  btnFit.addEventListener("click", fit);

  init().then(() => {
    init_panic_hook();
    ready = true;
    regen();
  }).catch(err => {
    status.textContent = "WebAssembly module failed to load: " + err.message
      + " — see About page for build instructions.";
  });
</script>

## What's running

The button calls `kmeans_fit(xs, n, d, k, max_iter, seed, use_kpp)` exported from [`src/wasm_impl/src/lib.rs`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/wasm_impl/src/lib.rs) and compiled via:

```bash
cd src/wasm_impl
wasm-pack build --target web --out-dir ../../docs/wasm
```

The WASM module is ~24 KB compressed. Float32Arrays cross the JS↔WASM boundary without copying thanks to `wasm-bindgen`'s typed-array support.
