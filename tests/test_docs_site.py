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
    """Every SVG/PNG in results/ and GIF in results/animations/ must be mirrored under docs/assets/."""
    src_images = sorted(
        (REPO_ROOT / "results").glob("*.svg")
    ) + sorted((REPO_ROOT / "results").glob("*.png"))
    src_gifs = sorted((REPO_ROOT / "results" / "animations").glob("*.gif"))

    for src in src_images:
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


def test_benchmark_page_names_cli_subprocess_scope():
    """The benchmark page must describe timing as the full CLI subprocess, not the
    clustering kernel in isolation, and must qualify memory as a sampled process-RSS
    estimate rather than a platform max-RSS reading.

    The exact prose is allowed to evolve; these assertions pin the *claims*, not a
    single literal sentence. Each claim is checked via a small set of phrases the
    rewritten page actually uses.
    """
    body = (DOCS / "benchmarks.md").read_text()
    lower = body.lower()

    # Claim 1: timing is the whole end-to-end CLI subprocess, not a function call.
    assert "subprocess" in lower
    assert "end to end" in lower or "end-to-end" in lower
    # ...and that scope explicitly includes process startup plus disk I/O.
    assert "process launch" in lower or "launch" in lower
    assert "csv" in lower  # reading/writing the CSV is named as part of the timed span

    # Claim 2: the timer wraps more than the clustering kernel — the page must say so.
    assert "kernel" in lower
    assert ("not the clustering kernel" in lower
            or "rather than the cost of its inner loop" in lower
            or "not just the kernel" in lower)

    # Claim 3: memory is a sampled process-RSS estimate, not platform max-RSS.
    assert "sampled process-rss estimate" in lower
    assert "max-rss" in lower
    assert "not a platform max-rss" in lower

    # Regression guard: the page must not revert to claiming I/O is excluded.
    assert "excludes data loading" not in lower
    assert "fit-only" not in lower
