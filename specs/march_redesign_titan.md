# Mar 8 (evening) — Pivot to Titan monochrome system

## Why
Morning's Mistral-style redesign shipped at 20:00. After review, the visual direction is being pivoted to the Titan "monochrome financial-ledger" system documented in `DESIGN.md` (parent directory). The two systems sit at opposite ends of the spectrum: Mistral is editorial/serif/sunset; Titan is monochrome/sans/pill — Linear/Stripe/Wealthfront family.

## What changes (vs the Mistral version)

| Aspect | Mistral (replaced) | Titan (new) |
|---|---|---|
| Palette | Saturated orange + sunset gradient + cream | Black + white + off-white sage + faded stone + soft orange accent only |
| Display font | Cormorant Garamond (near-serif) | Geist 700 (fallback: Inter 700) |
| UI font | Inter | Geist (fallback: Inter) |
| Card radius | 12px | 32px |
| Button radius | 8px | **160px** (pill) |
| Nav-link radius | — | **140px** (pill) |
| Signature | Sunset-stripe gradient band | None — just spacious whitespace |
| Hero | Full-bleed gradient sky + mountain SVG | Two-column split: bold headline left, illustration right |
| Decoration | Heavy | Almost none — content-first |

## Concrete tokens
```
--color-midnight-ink:    #111111  primary text
--color-canvas-white:    #ffffff  page bg
--color-off-white-sage:  #f3efeb  card bg
--color-faded-stone:     #e9eaeb  dividers, navbar
--color-gunmetal-gray:   #615e5b  secondary text
--color-soft-concrete:   #d8d3cc  subtle borders
--color-action-black:    #000000  primary button bg
--color-highlight-orange:#ff9900  subtle accent only
```

Spacing rhythm: 4px base; section gap 80px; card padding 28px; element gap 24px.

## Pages touched (same files as morning redesign)
1. `docs/assets/css/style.css` — full rewrite with Titan tokens
2. `docs/_layouts/default.html` — drop sunset stripe; drop Cormorant; restyle nav as pill-radius links; load Geist fallback
3. `docs/index.md` — replace editorial hero band with Titan two-column hero (headline + CTA + monospace stat strip)
4. `docs/algorithms.md` `docs/parallel.md` `docs/benchmarks.md` `docs/demo.md` `docs/about.md` — drop eyebrows; replace card-cream with feature-card on Off-White Sage

## Tests
`tests/test_docs_site.py` (5 tests) must remain green — no structural file changes; demo.md still references `kmeans_wasm` and `<canvas`.

## Out of scope
- Custom Geist font hosting (use Inter as fallback per DESIGN.md substitute spec)
- Detailed illustration set — `DESIGN.md` describes "fine linework cityscape" illustrations; we keep the existing GIFs and PNGs in their own gallery cards rather than commissioning illustrations
