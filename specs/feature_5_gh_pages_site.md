# Feature 5 — GitHub Pages site + live WASM K-Means demo

## Why
All the visual artifacts produced by features 1–4 currently live in `results/` and need a human to navigate the repo to find them. A GitHub Pages site stitches everything into a discoverable narrative and adds a **live, interactive WASM K-Means demo** that runs entirely in the visitor's browser. This is the end-goal of the whole effort.

## Site structure (Jekyll, served from `docs/`)
```
docs/
├── _config.yml            # Jekyll config; theme = minima; markdown = kramdown
├── _layouts/
│   └── default.html       # Single layout with header + nav + content slot
├── assets/
│   ├── css/style.css      # Custom styling on top of minima
│   ├── images/            # Static PNGs copied from results/
│   └── animations/        # GIFs copied from results/animations/
├── wasm/
│   ├── kmeans_bg.wasm     # Compiled Rust → wasm32-unknown-unknown
│   ├── kmeans.js          # wasm-bindgen JS glue
│   └── demo.js            # Demo controller (uses kmeans.js)
├── index.md               # Landing page
├── benchmarks.md          # Plotly dashboard (iframed or embedded)
├── algorithms.md          # k-means++ vs random, with GIFs
├── parallel.md            # Parallel Rust scaling story
├── demo.md                # Live WASM demo
└── about.md               # Project background, links to GitHub
```

## WASM K-Means
- New crate at `src/wasm_impl/` (separate Cargo project so it doesn't drag wasm dependencies into the CLI binary):
  - `Cargo.toml` with `wasm-bindgen`, `getrandom = { version = "0.2", features = ["js"] }`, and `[lib] crate-type = ["cdylib"]`.
  - `src/lib.rs` exposing two `#[wasm_bindgen]` functions:
    - `init_panic_hook()` — `console_error_panic_hook::set_once()` for debuggability.
    - `kmeans_fit(xs: &[f32], n: usize, d: usize, k: usize, max_iter: usize, seed: u32) -> Vec<i32>` — returns flat label vector. f32 chosen because it crosses the wasm boundary efficiently.
- Build with `wasm-pack build --target web --out-dir ../../docs/wasm` from `src/wasm_impl/`.

## Demo page UX
- Canvas (~480×480px) showing 2-D points.
- Controls: `n_points` slider (50–2000), `k` slider (2–8), `init` (random/k-means++ — done in JS), `[Generate blobs]`, `[Run K-Means]`.
- Status line: "Ran in X ms (WASM)" — shows the timing payoff vs pure JS.
- Falls back gracefully if `kmeans_bg.wasm` fails to fetch (e.g., during local dev with `file://`): a one-line message linking to build instructions.

## Tests (write first)
`tests/test_docs_site.py`:
1. `test_required_pages_exist`: every file listed under "Site structure" exists.
2. `test_jekyll_config_has_site_metadata`: `_config.yml` parseable as YAML and contains `title`, `description`, `url`/`baseurl`.
3. `test_assets_synced`: every PNG in `results/` and every GIF in `results/animations/` is mirrored under `docs/assets/`.
4. `test_wasm_demo_present`: `docs/wasm/kmeans_bg.wasm` and `docs/wasm/kmeans.js` exist (built).
5. `test_demo_page_references_wasm`: `docs/demo.md` references `kmeans.js` and contains a canvas element.

A small `src/sync_assets.py` script copies new artifacts into `docs/assets/` and is idempotent — tested by running it twice and asserting no error.

## Acceptance criteria
- `pytest tests/test_docs_site.py -v` — green.
- `wasm-pack build` succeeds; `docs/wasm/` is committed.
- Opening `docs/index.md` previewed via Jekyll-compatible markdown renders correctly (we won't run a full Jekyll build locally — that's CI / GH Pages territory — but we verify markdown is valid).
- GH Pages config in `_config.yml` sets `baseurl: /bench-kmeans-rust` so paths work when hosted.

## Out of scope
- Custom theme work beyond a stylesheet override.
- GH Actions for automated site builds — GH Pages with `from: docs` is built-in.
- A WASM demo for the full feature set (high-d, sklearn comparison) — only 2-D blobs.
- Service worker / offline support.
