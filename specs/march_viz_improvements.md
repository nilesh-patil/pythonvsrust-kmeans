# March 6 Visualization Improvements

**Author:** Viz agent  
**Date:** 2026-03-06  
**Branch:** feature/march-viz-improvements

## Unified Palette

Current implementation-level charts use the shared encodings in `src/viz_style.py`:

| Implementation   | Hex       |
|-----------------|-----------|
| python          | `#0072B2` |
| rust (serial)   | `#D55E00` |
| rust_parallel   | `#CC79A7` |
| sklearn         | `#009E73` |
| random_init     | `#d97706` |
| kpp_init        | `#0ea5e9` |

## Per-Script Changes

### 1. `src/visualize_init_comparison.py`
- Add ratio annotation above each bar pair showing `"X.X×"` (random inertia / kpp inertia).
- Confirm palette: random=`#d97706`, kpp=`#0ea5e9` (already correct).
- Add figure-level text at bottom: `"Lower is better · 10 seeds per bar"`.

### 2. `src/visualize_parallel_scaling.py`
- Keep runtime on a log2 y-axis.
- Avoid dual y-axes; the single-CSV view gives efficiency its own panel.
- When `results/parallel_scaling_n*.csv` exists, render the current scale-sweep view: runtime vs sample count, best speedup vs sample count, and large-workload thread curves.

### 3. `src/animate_convergence.py`
- Add inset axes (top-right) showing inertia vs iteration as a line plot, updated per frame.
- Add two new scenarios: `make_moons` (k=2) → `convergence_moons.gif`; `make_circles` (k=2) → `convergence_circles.gif`.
- All 5 GIFs must stay < 250 KB.

### 4. `runner.py:generate_plots()`
- Set log-y scale on runtime boxplot panel.
- Add ±1σ fill_between on the scalability line plot.
- Define local `RUNNER_PALETTE` dict matching unified palette; apply to scalability + quality bar colors.
- Use `DISPLAY_NAMES` mapping for axis / legend labels.

### 5. `src/build_dashboard.py`
- Wire `_display_name(impl)` into every `go.Scatter` and `go.Bar` `name=` argument so the dashboard legend shows "Rust", "Rust - Parallel", "scikit-learn", etc.
- Verify `_color(impl)` already used for line/marker colors (it is — no change needed there).

## Acceptance Criteria
- All 5 PNG/GIF outputs regenerate without error.
- `pixi run python -m pytest tests/ -q --no-header` passes completely.
- Docs and mirrored website assets are refreshed when benchmark inputs change.
