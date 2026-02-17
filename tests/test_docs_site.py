"""Tests for the Jekyll site under docs/."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS      = REPO_ROOT / "docs"

REQUIRED_PAGES = [
    DOCS / "_config.yml",
    DOCS / "_layouts" / "default.html",
    DOCS / "assets"   / "css"  / "style.css",
    DOCS / "index.md",
    DOCS / "benchmarks.md",
    DOCS / "algorithms.md",
    DOCS / "parallel.md",
    DOCS / "demo.md",
    DOCS / "about.md",
]


def test_required_pages_exist():
    missing = [p for p in REQUIRED_PAGES if not p.exists()]
    assert not missing, f"missing pages: {missing}"


def test_jekyll_config_has_site_metadata():
    cfg = yaml.safe_load((DOCS / "_config.yml").read_text())
    for key in ("title", "description", "baseurl"):
        assert key in cfg, f"_config.yml missing key: {key}"


def test_assets_synced():
    """Every PNG in results/ and GIF in results/animations/ must be mirrored under docs/assets/."""
    src_pngs = sorted((REPO_ROOT / "results").glob("*.png"))
    src_gifs = sorted((REPO_ROOT / "results" / "animations").glob("*.gif"))

    for src in src_pngs:
        mirror = DOCS / "assets" / "images" / src.name
        assert mirror.exists(), f"missing mirror for {src.name}"

    for src in src_gifs:
        mirror = DOCS / "assets" / "animations" / src.name
        assert mirror.exists(), f"missing mirror for {src.name}"


def test_wasm_demo_present():
    wasm = DOCS / "wasm" / "kmeans_wasm_bg.wasm"
    js   = DOCS / "wasm" / "kmeans_wasm.js"
    assert wasm.exists(), "compiled wasm missing — run wasm-pack build --target web"
    assert js.exists(),   "wasm-bindgen JS glue missing"


def test_demo_page_references_wasm():
    body = (DOCS / "demo.md").read_text()
    assert "kmeans_wasm.js" in body or "kmeans_wasm" in body
    assert "<canvas" in body
