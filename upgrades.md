# Upgrades backlog

A page-by-page review of the site (visual and editorial), with proposed changes.
Ordered by priority within each section. Nothing here is urgent; the items at the
top are the ones I'd do first.

## Site-wide

1. **Nav has no active-page indication.** Every link in the header renders
   identically, with no `aria-current`. Style the current page's link (accent
   underline or darker ink) and set `aria-current="page"`.
2. **No prev/next footer links.** The six pages read as a sequence
   (Overview → Algorithms → Parallelism → Benchmarks → Demo → About) but the only
   way between them is the top nav. Add quiet "← Parallelism · Benchmarks →"
   links at the bottom of each page.
3. **No per-page `<h1>`.** Pages open at `<h2>`; the document outline starts at
   the site title link. Add a visible page title (or a visually-quiet h1) for
   accessibility, SEO, and orientation. Related: About opens with a plain
   paragraph while every other page opens with an italic epigraph — pick one.
4. **Sidenotes disappear between 760 and 1280 px.** The margin notes only float
   at ≥1280 px; below that they hide behind a ⊕ toggle that doesn't read as a
   footnote marker. Lower the breakpoint (sidenotes fit at ~1100 px) and style
   the toggle as a numbered superscript.
5. **Dark mode** (`prefers-color-scheme`) — the paper/ink palette would invert
   cleanly. **Print stylesheet** — expand sidenotes inline, hide nav/demo/iframe,
   keep figures whole. Both are natural fits for an essay site.

## Mobile (390 px)

6. **The thread-sweep table on Parallelism scrolls the whole page sideways.**
   The 6-column table renders wider than the viewport and drags `<main>` with
   it. Wrap wide tables in an `overflow-x: auto` container so they scroll
   within themselves.
7. **The embedded dashboard is unusable on phones.** At 310 px the Plotly
   x-ticks smear, the legend eats half the width, and a nested scrollbar
   appears. Below ~700 px, swap the iframe for an "open the dashboard ↗" link;
   on the dashboard itself, move the legend below the plot and set
   `displayModeBar: false`.
8. **The four-panel overview chart is illegible at 310 px** (4.6× downscale).
   Serve a stacked single-column variant on narrow screens, or split it into
   four figures with the existing single-panel styling.
9. **Display math is full-bleed on mobile.** KaTeX blocks span edge to edge
   while body text is inset; give display math the same inset and let it
   scroll inside its own box.

## Per page

10. **Parallelism / About: code blocks clip their most important lines.** At the
    666 px column, the `fold`/`reduce` closing lines and the `wasm-pack` command
    hide behind a horizontal scrollbar. Let `<pre>` blocks break out to the
    906 px figure width (or wrap long lines).
11. **Demo: the default run converges in one iteration** — nothing to watch.
    Default to two moons with random init (the case the intro text itself
    recommends) so the first click shows a real animation and a falling
    inertia curve.
12. **Demo: the canvas leaves half the page empty at desktop widths.** The grid
    is ~713 px in a 1440 px window. Let the visual column grow; the 100k-point
    render deserves the space.
13. **Algorithms: the GIF grid leaves an unbalanced empty cell** (5 items in a
    3-column grid). Center the last row. The inertia insets inside the GIFs are
    illegible at 285 px — consider dropping them from the animation and letting
    the page's own inertia discussion carry that detail.
14. **Overview: line-end labels collide** in the throughput panel
    ("Rust-Parallel"/"Rust") and the memory panel ("Python"/"scikit-learn").
    Nudge the label anchors apart in the figure scripts.
15. **Dashboard standalone page is a dead end** — no link back to the site. Add
    a small "← K-Means, four ways" header link.
16. **Overview/README: the hero image is a timestamped filename**
    (`benchmark_plots_20260609_112255.svg`). Give the canonical figure a stable
    name in the sync step so the published pages don't rot when the suite is
    re-run.

## Content ideas (questions a reader will ask that the site doesn't answer)

17. **Why is scikit-learn slow at small n?** The pages show the crossover but
    never name the fixed cost (interpreter start, import, dispatch, thread-pool
    spin-up). One sidenote on the Overview would close the most predictable
    open question and reinforce the end-to-end thesis.
18. **scikit-learn isn't running Lloyd's.** Its default is the Elkan variant
    with chunked BLAS — so the comparison is "my Lloyd's vs sklearn's smarter
    algorithm." One honest sidenote on Benchmarks would tighten the
    like-for-like framing.
19. **float32.** Everything here is `f64`. Half the memory and better SIMD
    density is the obvious untested lever; worth a line in About's "what I'd do
    differently."
20. **The flat-matrix fix is promised three times** (Overview, Parallelism,
    About) **but never quantified.** Either attempt it and report, or say
    explicitly that the expected gain is unmeasured, once.
21. **WASM-vs-JS race framing.** On tiny in-browser workloads the JS path can
    win because the call overhead dominates — the readout now says so honestly,
    but the demo could also time the kernel only, or warm up the module, to
    make the comparison fairer to WASM.
