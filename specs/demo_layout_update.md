# Demo layout update — 2-column + bigger ranges

## Why
The current demo page stacks every element top-to-bottom: canvas, then inertia chart, then controls, then status. On wider screens this leaves the right side of the viewport empty and forces the user to scroll past the visual to reach the controls. Restructuring to a two-column layout (controls left, visuals right) puts every interactive surface in one glance.

The user also wants to push the demo into substantively larger workloads — up to 100 000 points and up to 1 000 iterations — which exposes new performance concerns the controls weren't dimensioned for.

## Layout contract (target)

```
┌─────────────────────────────────────────────────────────┐
│  HEADER  (eyebrow + headline + lead paragraph)           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────┬───────────────────────────────────┐
│                     │                                   │
│   CONTROLS          │   K-Means visual                  │
│   (single column,   │   (canvas, 480×480 or bigger)     │
│    left side)       │                                   │
│                     ├───────────────────────────────────┤
│   • Distribution    │   Inertia per iteration           │
│   • Points          │   (small canvas / chart)          │
│   • Clusters        │                                   │
│   • Init            │                                   │
│   • Max iter        │                                   │
│   • Speed           │                                   │
│   • Buttons         │                                   │
│                     │                                   │
│   Status line       │                                   │
└─────────────────────┴───────────────────────────────────┘
```

Below the columns: any prose / explanation sections continue full-width.

## Slider ranges (target)
- **Points (`#ctl-n`)**: `min=50`, `max=100000`, `step` scales — coarse at large N (1000 step ≥ 10000, 100 step from 1000–10000, 50 step under 1000).
- **Max iter (`#ctl-maxiter`)**: `min=1`, `max=1000`, `step=1`.

Default values stay sensible: points 400, max iter 20.

## Performance guardrails
- Rendering 100 000 circle arcs every frame at 60 fps is too many draw calls. Switch to a **batched-by-color** path: one `Path2D` per cluster colour, all points added with `moveTo`/`arc`, single `fill()` per cluster per frame.
- Pure-JS K-Means on 100 000 points × 1 000 iterations is too slow to block the main thread. When `n × max_iter > 5_000_000`, skip the JS race and show "(skipped — N×iter too large for synchronous JS)" instead.
- WASM is unchanged — it handles the bigger workloads natively.

## SDD acceptance criteria
1. `tests/test_demo_layout.py` (new) passes — see test plan below.
2. `tests/test_docs_site.py` (5 existing tests) still passes.
3. Live `/demo/` shows the new two-column layout on screens ≥ 900 px.
4. Below 900 px wide, the layout collapses gracefully to a single column (controls first, then visual + inertia).
5. Sliders accept the new range without breaking the WASM call.

## Test plan
`tests/test_demo_layout.py`:
1. `test_two_column_wrapper_exists` — assert `docs/demo.md` contains a `class="demo-grid"` (or equivalent) wrapper.
2. `test_points_slider_max_is_100000` — regex match `id="ctl-n"` line for `max="100000"`.
3. `test_maxiter_slider_max_is_1000` — regex match `id="ctl-maxiter"` line for `max="1000"`.
4. `test_inertia_canvas_present_below_main_canvas` — order check: `<canvas id="demo-canvas"` appears before `<canvas id="inertia-canvas"` in source.

## Out of scope
- Web-worker offload for JS K-Means (the synchronous-skip guardrail is sufficient).
- Plotly-style interactive inertia chart (keep canvas-based).
- Auto-tuning the `step` from JavaScript (we use HTML `step` attribute splits).
