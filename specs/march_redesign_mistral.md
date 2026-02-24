# Mar 8 — Mistral-style redesign of the GH Pages site

## Why
The existing site (post-February) used minima's defaults plus a thin custom stylesheet. With the demo + dashboard now substantive, the visual presentation lagged the content. This redesign applies the Mistral AI design system documented in `DESIGN-new.md` — editorial near-serif display type, cream-yellow form surfaces, saturated-orange CTAs, and the signature **sunset stripe band** at the foot of every page.

## Palette (concrete hex values for the Mistral tokens)
- `--primary`           `#FA500F`  Mistral orange
- `--primary-deep`      `#E54400`  pressed state
- `--sunshine-700`      `#F26B1F`
- `--sunshine-500`      `#F89B30`
- `--sunshine-300`      `#FCC56B`
- `--yellow-saturated`  `#FFD740`
- `--cream`             `#FBF6E6`
- `--cream-soft`        `#FDFAEF`
- `--cream-deeper`      `#F4EBC5`
- `--beige-deep`        `#E8DDB5`
- `--canvas`            `#FFFFFF`
- `--surface-cream`     `#FAF4E0`
- `--surface-code`      `#1B1A18`
- `--hairline`          `#E5E2D6`
- `--hairline-soft`     `#EDEAE0`
- `--hairline-strong`   `#B5AE94`
- `--ink`               `#1B1A18`
- `--charcoal`          `#4A4842`
- `--slate`             `#6B695F`
- `--steel`             `#8F8C7E`

## Typography pairing
- **Display**: Cormorant Garamond (Google Fonts) — near-serif editorial, the closest free analog to PP Editorial Old.
- **UI**: Inter (Google Fonts).
- **Code**: JetBrains Mono.
- Hero display 64–84px / 400 weight / -1.5px tracking / 1.05 line-height.
- Body 16px Inter 400 / 1.55 line-height.

## Critical components to ship
- **`.sunset-stripe`** — multi-stop gradient band (orange → sunshine-500 → yellow → cream) above the footer on every page.
- **`.hero-band`** — full-bleed editorial hero with gradient sky + SVG mountain silhouette.
- **`.cta-band-cream`** — page-bottom call-to-action on cream surface.
- **`.btn-primary` / `.btn-dark` / `.btn-cream` / `.btn-secondary`** — `{rounded.md}` (8px), 10px×20px padding.
- **`.card`** — 12px radius, cream border or `--hairline-soft`.
- **`.card-cream`** — for form panels / feature callouts.

## Pages updated
1. `docs/_layouts/default.html` — top nav restyle + sunset stripe footer integration
2. `docs/assets/css/style.css` — complete rewrite using Mistral tokens
3. `docs/index.md` — hero, stat row, feature cards
4. `docs/algorithms.md` / `docs/parallel.md` — section eyebrows, editorial H2s
5. `docs/benchmarks.md` — iframe in a card-cream container
6. `docs/demo.md` — demo-controls panel in cream
7. `docs/about.md` — footer-style content with cream callouts

## Acceptance
- All 5 tests in `tests/test_docs_site.py` still pass (no structural file changes).
- Live site renders correctly after Pages rebuild.
- Sunset stripe present at the foot of every page.
- All buttons are 8px radius, no pills (badges still pill).
