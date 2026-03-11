/**
 * demo.js — K-Means live demo controller
 * ES module; imported by docs/demo.md via <script type="module">.
 *
 * Mar 4: distributions + JS extraction
 * Mar 5: animation loop, WASM-vs-JS race, inertia chart
 */

import init, { init_panic_hook, kmeans_fit_steps } from
  "/bench-kmeans-rust/wasm/kmeans_wasm.js";

// ---------------------------------------------------------------------------
// RNG helpers
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Distribution generators — all return Float32Array(2n) in [0,1]
// ---------------------------------------------------------------------------

/**
 * Gaussian blobs: k tight blobs arranged on a circle.
 */
function generateBlobs(n, k, rng) {
  const xs = new Float32Array(n * 2);
  const centers = [];
  for (let c = 0; c < k; c++) {
    const angle = (2 * Math.PI * c) / k;
    centers.push([0.5 + 0.32 * Math.cos(angle), 0.5 + 0.32 * Math.sin(angle)]);
  }
  for (let i = 0; i < n; i++) {
    const c = centers[i % k];
    xs[2 * i]     = Math.max(0, Math.min(1, c[0] + 0.04 * gaussian(rng)));
    xs[2 * i + 1] = Math.max(0, Math.min(1, c[1] + 0.04 * gaussian(rng)));
  }
  return xs;
}

/**
 * Concentric rings: 2 annuli, points alternated between them.
 * Hard for K-Means when k > 2.
 */
function generateRings(n, k, rng) {
  const xs = new Float32Array(n * 2);
  const radii = [0.18, 0.36];
  const noise = 0.025;
  for (let i = 0; i < n; i++) {
    const ring = i % 2;
    const r = radii[ring] + noise * gaussian(rng);
    const angle = rng() * 2 * Math.PI;
    xs[2 * i]     = Math.max(0, Math.min(1, 0.5 + r * Math.cos(angle)));
    xs[2 * i + 1] = Math.max(0, Math.min(1, 0.5 + r * Math.sin(angle)));
  }
  return xs;
}

/**
 * Two moons: two crescent shapes, interleaved.
 * Classic non-convex challenge for centroid-based clustering.
 */
function generateMoons(n, k, rng) {
  const xs = new Float32Array(n * 2);
  const noise = 0.03;
  for (let i = 0; i < n; i++) {
    const moon = i % 2;
    const t = rng() * Math.PI; // half-circle
    let x, y;
    if (moon === 0) {
      x = Math.cos(t);
      y = Math.sin(t);
    } else {
      x = 1 - Math.cos(t);
      y = 0.5 - Math.sin(t);
    }
    xs[2 * i]     = Math.max(0, Math.min(1, (x + 1) / 2.2 + 0.1 + noise * gaussian(rng)));
    xs[2 * i + 1] = Math.max(0, Math.min(1, (y + 1) / 2.2 + 0.1 + noise * gaussian(rng)));
  }
  return xs;
}

/**
 * Anisotropic blobs: k elongated, tilted blobs.
 * Tests sensitivity to non-spherical cluster shapes.
 */
function generateAnisotropic(n, k, rng) {
  const xs = new Float32Array(n * 2);
  const centers = [];
  for (let c = 0; c < k; c++) {
    const angle = (2 * Math.PI * c) / k;
    centers.push({
      cx: 0.5 + 0.30 * Math.cos(angle),
      cy: 0.5 + 0.30 * Math.sin(angle),
      tilt: angle + Math.PI / 4,   // rotation per blob
    });
  }
  for (let i = 0; i < n; i++) {
    const c = centers[i % k];
    const a = 0.10 * gaussian(rng); // long axis
    const b = 0.025 * gaussian(rng); // short axis
    const rx = a * Math.cos(c.tilt) - b * Math.sin(c.tilt);
    const ry = a * Math.sin(c.tilt) + b * Math.cos(c.tilt);
    xs[2 * i]     = Math.max(0, Math.min(1, c.cx + rx));
    xs[2 * i + 1] = Math.max(0, Math.min(1, c.cy + ry));
  }
  return xs;
}

/**
 * Uniform random: no cluster structure. Good baseline / stress test.
 */
function generateUniform(n, k, rng) {
  const xs = new Float32Array(n * 2);
  for (let i = 0; i < n * 2; i++) {
    xs[i] = rng();
  }
  return xs;
}

/**
 * Spiral / pinwheel: k arms radiating from center with added noise.
 * Visually dramatic; K-Means struggles since arms are not convex blobs.
 */
function generateSpiral(n, k, rng) {
  const xs = new Float32Array(n * 2);
  const turns = 1.5; // number of full rotations per arm
  const noise = 0.02;
  for (let i = 0; i < n; i++) {
    const arm = i % k;
    const t = rng(); // [0,1] along the arm
    const angle = (2 * Math.PI * arm) / k + turns * 2 * Math.PI * t;
    const r = 0.05 + 0.38 * t;
    xs[2 * i]     = Math.max(0, Math.min(1, 0.5 + r * Math.cos(angle) + noise * gaussian(rng)));
    xs[2 * i + 1] = Math.max(0, Math.min(1, 0.5 + r * Math.sin(angle) + noise * gaussian(rng)));
  }
  return xs;
}

const DISTRIBUTIONS = {
  blobs:       generateBlobs,
  rings:       generateRings,
  moons:       generateMoons,
  anisotropic: generateAnisotropic,
  uniform:     generateUniform,
  spiral:      generateSpiral,
};

// ---------------------------------------------------------------------------
// Pure-JS K-Means (mirrors the WASM algorithm for the race)
// ---------------------------------------------------------------------------

/**
 * kmeansJs: returns final Int32Array of labels.
 * Uses identical Mulberry32 RNG and k-means++ init logic to make timing fair.
 */
function kmeansJs(xs, n, d, k, maxIter, seed, useKpp) {
  const rng = mulberry32(seed);

  // --- init centroids ---
  const centroids = new Float32Array(k * d);
  if (useKpp) {
    // k-means++ init
    const first = Math.floor(rng() * n);
    for (let dim = 0; dim < d; dim++) {
      centroids[dim] = xs[first * d + dim];
    }
    for (let ci = 1; ci < k; ci++) {
      const dists = new Float64Array(n);
      let total = 0;
      for (let i = 0; i < n; i++) {
        let minD2 = Infinity;
        for (let cj = 0; cj < ci; cj++) {
          let dist2 = 0;
          for (let dim = 0; dim < d; dim++) {
            const diff = xs[i * d + dim] - centroids[cj * d + dim];
            dist2 += diff * diff;
          }
          if (dist2 < minD2) minD2 = dist2;
        }
        dists[i] = minD2;
        total += minD2;
      }
      let threshold = rng() * total;
      let chosen = n - 1;
      for (let i = 0; i < n; i++) {
        threshold -= dists[i];
        if (threshold <= 0) { chosen = i; break; }
      }
      for (let dim = 0; dim < d; dim++) {
        centroids[ci * d + dim] = xs[chosen * d + dim];
      }
    }
  } else {
    // random init
    const chosen = new Set();
    let ci = 0;
    while (ci < k) {
      const idx = Math.floor(rng() * n);
      if (!chosen.has(idx)) {
        chosen.add(idx);
        for (let dim = 0; dim < d; dim++) {
          centroids[ci * d + dim] = xs[idx * d + dim];
        }
        ci++;
      }
    }
  }

  // --- Lloyd's iterations ---
  const labels = new Int32Array(n);
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // assign
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < k; c++) {
        let dist2 = 0;
        for (let dim = 0; dim < d; dim++) {
          const diff = xs[i * d + dim] - centroids[c * d + dim];
          dist2 += diff * diff;
        }
        if (dist2 < bestDist) { bestDist = dist2; bestC = c; }
      }
      if (labels[i] !== bestC) { labels[i] = bestC; changed = true; }
    }

    if (!changed) break;

    // update
    const sums = new Float64Array(k * d);
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      for (let dim = 0; dim < d; dim++) {
        sums[c * d + dim] += xs[i * d + dim];
      }
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let dim = 0; dim < d; dim++) {
          centroids[c * d + dim] = sums[c * d + dim] / counts[c];
        }
      }
    }
  }

  return labels;
}

// ---------------------------------------------------------------------------
// Snapshot parser + inertia computer
// ---------------------------------------------------------------------------

/**
 * Parse the flat buffer returned by kmeans_fit_steps.
 * Returns { iterCount, converged, snapshots[] }
 * Each snapshot: { centroids: Float32Array(k*d), labels: Int32Array(n), inertia: number }
 */
function parseStepBuffer(buf, n, k, d, xs) {
  const iterCount  = Math.round(buf[0]);
  const converged  = buf[1] > 0.5;
  const snapCount  = iterCount + 1;       // initial + after each iter
  const snapshotLen = k * d + n;

  const snapshots = [];
  for (let s = 0; s < snapCount; s++) {
    const offset = 4 + s * snapshotLen;
    const centroids = buf.slice(offset, offset + k * d);
    const labelsF   = buf.slice(offset + k * d, offset + snapshotLen);
    const labels    = new Int32Array(n);
    for (let i = 0; i < n; i++) labels[i] = Math.round(labelsF[i]);

    // compute inertia for this snapshot
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      for (let dim = 0; dim < d; dim++) {
        const diff = xs[i * d + dim] - centroids[c * d + dim];
        inertia += diff * diff;
      }
    }
    snapshots.push({ centroids, labels, inertia });
  }

  return { iterCount, converged, snapshots };
}

// ---------------------------------------------------------------------------
// Canvas drawing helpers
// ---------------------------------------------------------------------------

// Muted, print-like categorical palette anchored on the implementation tokens
// (steel blue, rust, ochre …) — see DESIGN_BRIEF.md.
const PAPER = "#fffff8";
const PALETTE = ["#3d6b9e","#b7410e","#c98c1f","#5b7553","#7a2e0c","#6a5687","#9e5b3d","#4f7a8c"];

function drawFrame(canvas, ctx, points, snapshot) {
  const n = points.length / 2;
  const W = canvas.width, H = canvas.height;

  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, W, H);

  if (!points) return;

  // Batched Path2D render: one path per cluster colour, single fill() per cluster
  const radius = n > 5000 ? 1.5 : 3;
  const pathByCluster = new Map();
  for (let i = 0; i < n; i++) {
    const c = snapshot ? snapshot.labels[i] : -1;
    let path = pathByCluster.get(c);
    if (!path) { path = new Path2D(); pathByCluster.set(c, path); }
    const x = points[2 * i]     * W;
    const y = points[2 * i + 1] * H;
    path.moveTo(x + radius, y);
    path.arc(x, y, radius, 0, 2 * Math.PI);
  }
  for (const [c, path] of pathByCluster) {
    ctx.fillStyle = c === -1 ? "#9d9d92" : PALETTE[c % PALETTE.length];
    ctx.fill(path);
  }

  if (!snapshot) return;

  // draw centroid X markers — filled circle + cross
  const k = snapshot.centroids.length / 2;
  for (let c = 0; c < k; c++) {
    const cx = snapshot.centroids[2 * c]     * W;
    const cy = snapshot.centroids[2 * c + 1] * H;
    const color = PALETTE[c % PALETTE.length];
    const r = 9;

    // paper halo for contrast
    ctx.beginPath();
    ctx.arc(cx, cy, r + 2, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(255,255,248,0.78)";
    ctx.fill();

    // colored circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // X cross in paper
    ctx.strokeStyle = PAPER;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 5, cy - 5); ctx.lineTo(cx + 5, cy + 5);
    ctx.moveTo(cx + 5, cy - 5); ctx.lineTo(cx - 5, cy + 5);
    ctx.stroke();
  }
}

function drawInertia(iCanvas, iCtx, snapshots, currentFrame) {
  const W = iCanvas.width, H = iCanvas.height;
  const pad = { top: 10, right: 10, bottom: 22, left: 42 };
  const iW = W - pad.left - pad.right;
  const iH = H - pad.top - pad.bottom;

  iCtx.clearRect(0, 0, W, H);
  if (!snapshots || snapshots.length === 0) return;

  const maxInertia = snapshots[0].inertia * 1.05;
  const total = snapshots.length;

  // background
  iCtx.fillStyle = PAPER;
  iCtx.fillRect(0, 0, W, H);
  iCtx.strokeStyle = "#dcdcd4";
  iCtx.lineWidth = 1;
  iCtx.strokeRect(pad.left, pad.top, iW, iH);

  // axes labels
  iCtx.fillStyle = "#595959";
  iCtx.font = "10px 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace";
  iCtx.textAlign = "right";
  iCtx.fillText(maxInertia.toFixed(2), pad.left - 3, pad.top + 9);
  iCtx.fillText("0", pad.left - 3, pad.top + iH);
  iCtx.textAlign = "center";
  iCtx.fillText("Iteration →", pad.left + iW / 2, H - 4);
  iCtx.textAlign = "left";
  iCtx.fillText("Inertia", pad.left, pad.top - 1);

  // line through current frame
  const framesToDraw = Math.min(currentFrame + 1, snapshots.length);
  if (framesToDraw < 2) return;

  iCtx.beginPath();
  iCtx.strokeStyle = "#b7410e";
  iCtx.lineWidth = 2;
  for (let f = 0; f < framesToDraw; f++) {
    const x = pad.left + (f / (total - 1)) * iW;
    const y = pad.top + iH - (snapshots[f].inertia / maxInertia) * iH;
    if (f === 0) iCtx.moveTo(x, y);
    else         iCtx.lineTo(x, y);
  }
  iCtx.stroke();

  // dot at current frame
  const fx = pad.left + ((framesToDraw - 1) / (total - 1)) * iW;
  const fy = pad.top + iH - (snapshots[framesToDraw - 1].inertia / maxInertia) * iH;
  iCtx.beginPath();
  iCtx.arc(fx, fy, 4, 0, 2 * Math.PI);
  iCtx.fillStyle = "#b7410e";
  iCtx.fill();
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

// states: idle | playing | paused | done
let state = "idle";
let wasmReady = false;

let points   = null;  // Float32Array(2n)
let snapshots = null; // parsed step data
let currentFrame = 0;
let lastFrameTime = 0;
let rafHandle = null;
let runSeed = 0;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const canvas    = document.getElementById("demo-canvas");
const ctx       = canvas.getContext("2d");
const iCanvas   = document.getElementById("inertia-canvas");
const iCtx      = iCanvas.getContext("2d");
const status    = document.getElementById("demo-status");
const raceDiv   = document.getElementById("race-result");

const ctlN       = document.getElementById("ctl-n");
const ctlK       = document.getElementById("ctl-k");
const ctlInit    = document.getElementById("ctl-init");
const ctlDist    = document.getElementById("ctl-dist");
const ctlMaxIter = document.getElementById("ctl-maxiter");
const ctlSpeed   = document.getElementById("ctl-speed");

const lblN       = document.getElementById("lbl-n");
const lblK       = document.getElementById("lbl-k");
const lblMaxIter = document.getElementById("lbl-maxiter");
const lblSpeed   = document.getElementById("lbl-speed");

const btnGen     = document.getElementById("btn-generate");
const btnFit     = document.getElementById("btn-fit");
const btnStep    = document.getElementById("btn-step");
const btnPause   = document.getElementById("btn-pause");
const btnReset   = document.getElementById("btn-reset");

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function setStatus(msg) { status.textContent = msg; }

function updateButtonStates() {
  const hasPoints  = points !== null;
  const hasHistory = snapshots !== null;

  const atLastFrame = hasHistory && currentFrame >= snapshots.length - 1;
  btnFit.disabled   = !wasmReady || !hasPoints || state === "playing";
  btnStep.disabled  = !hasHistory || state === "playing" || atLastFrame;
  btnPause.disabled = !hasHistory || (state !== "playing" && state !== "paused");
  btnReset.disabled = state === "idle" && !hasHistory;

  btnPause.textContent = state === "paused" ? "Resume" : "Pause";
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

function regen() {
  cancelAnim();
  state = "idle";
  snapshots = null;
  currentFrame = 0;
  raceDiv.innerHTML = "";

  const n    = parseInt(ctlN.value, 10);
  const k    = parseInt(ctlK.value, 10);
  const dist = ctlDist.value;
  const seed = Date.now() & 0xffffffff;
  const rng  = mulberry32(seed);

  const gen = DISTRIBUTIONS[dist] || generateBlobs;
  points = gen(n, k, rng);

  drawFrame(canvas, ctx, points, null);
  iCtx.clearRect(0, 0, iCanvas.width, iCanvas.height);
  setStatus(`${n} points generated (${ctlDist.options[ctlDist.selectedIndex].text}). Click "Run K-Means".`);
  updateButtonStates();
}

// ---------------------------------------------------------------------------
// Fit & animation
// ---------------------------------------------------------------------------

function cancelAnim() {
  if (rafHandle !== null) {
    cancelAnimationFrame(rafHandle);
    rafHandle = null;
  }
}

function fitWithSteps() {
  if (!wasmReady || !points) return;
  cancelAnim();

  const n       = points.length / 2;
  const k       = parseInt(ctlK.value, 10);
  const maxIter = parseInt(ctlMaxIter.value, 10);
  const useKpp  = ctlInit.value === "kpp";
  runSeed       = (Math.random() * 2 ** 32) >>> 0;

  // WASM call — timed
  const t0wasm = performance.now();
  const buf = kmeans_fit_steps(points, n, 2, k, maxIter, runSeed, useKpp);
  const t1wasm = performance.now();

  // Parse step buffer
  const parsed = parseStepBuffer(buf, n, k, 2, points);
  snapshots = parsed.snapshots;
  currentFrame = 0;

  const wasmMs = (t1wasm - t0wasm).toFixed(1);

  // JS race — skip when n × max_iter > 5,000,000 (too slow to block main thread)
  if (n * maxIter > 5_000_000) {
    raceDiv.innerHTML =
      `<span class="rust">WASM: ${wasmMs} ms</span> · JS: skipped (n×iter too large)`;
  } else {
    const t0js = performance.now();
    kmeansJs(points, n, 2, k, maxIter, runSeed, useKpp);
    const t1js = performance.now();

    const jsMs  = (t1js - t0js).toFixed(1);
    const ratio = (t1js - t0js) / Math.max(t1wasm - t0wasm, 0.01);
    const verdict = ratio >= 1
      ? `WASM ${ratio.toFixed(1)}× faster`
      : `JS ${(1 / ratio).toFixed(1)}× faster — call overhead dominates at this size`;
    raceDiv.innerHTML =
      `<span class="rust">WASM: ${wasmMs} ms</span> · ` +
      `<span class="js">JS: ${jsMs} ms</span>` +
      ` (${verdict})`;
  }

  const convergenceNote = parsed.converged ? `converged in ${parsed.iterCount} iter` : `max ${parsed.iterCount} iter`;
  setStatus(`n=${n}, k=${k}, ${convergenceNote}. Playing…`);

  state = "playing";
  lastFrameTime = 0;
  updateButtonStates();
  rafHandle = requestAnimationFrame(animStep);
}

function animStep(ts) {
  if (state !== "playing") return;

  const fps  = parseFloat(ctlSpeed.value);
  const mspf = 1000 / fps;

  if (ts - lastFrameTime >= mspf) {
    lastFrameTime = ts;
    renderCurrentFrame();
    if (currentFrame < snapshots.length - 1) {
      currentFrame++;
    } else {
      // reached last frame
      state = "done";
      setStatus(status.textContent.replace("Playing…", "Done."));
      updateButtonStates();
      return;
    }
  }

  rafHandle = requestAnimationFrame(animStep);
}

function renderCurrentFrame() {
  if (!snapshots) return;
  const snap = snapshots[currentFrame];
  drawFrame(canvas, ctx, points, snap);
  drawInertia(iCanvas, iCtx, snapshots, currentFrame);

  const iter = currentFrame;
  setStatus(`Iteration ${iter} / ${snapshots.length - 1}  ·  inertia: ${snap.inertia.toFixed(4)}`);
}

function stepOne() {
  if (!snapshots) return;
  cancelAnim();
  if (state === "playing") state = "paused";
  if (currentFrame < snapshots.length - 1) currentFrame++;
  renderCurrentFrame();
  if (currentFrame >= snapshots.length - 1) state = "done";
  updateButtonStates();
}

function togglePause() {
  if (state === "playing") {
    cancelAnim();
    state = "paused";
  } else if (state === "paused") {
    state = "playing";
    lastFrameTime = 0;
    rafHandle = requestAnimationFrame(animStep);
  }
  updateButtonStates();
}

function reset() {
  cancelAnim();
  state = "idle";
  snapshots = null;
  currentFrame = 0;
  raceDiv.innerHTML = "";
  regen();
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

ctlN.addEventListener("input",       () => { lblN.textContent = ctlN.value; });
ctlK.addEventListener("input",       () => { lblK.textContent = ctlK.value; });
ctlMaxIter.addEventListener("input", () => { lblMaxIter.textContent = ctlMaxIter.value; });
ctlSpeed.addEventListener("input",   () => { lblSpeed.textContent = ctlSpeed.value; });

ctlDist.addEventListener("change",   regen);

btnGen.addEventListener("click",     regen);
btnFit.addEventListener("click",     fitWithSteps);
btnStep.addEventListener("click",    stepOne);
btnPause.addEventListener("click",   togglePause);
btnReset.addEventListener("click",   reset);

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

init().then(() => {
  init_panic_hook();
  wasmReady = true;
  regen();
}).catch(err => {
  setStatus("WebAssembly module failed to load: " + err.message +
    " — see About page for build instructions.");
  updateButtonStates();
});
