"""TDD red phase for the two-column demo layout update."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_MD   = REPO_ROOT / "docs" / "demo.md"


def _read():
    return DEMO_MD.read_text()


def test_two_column_wrapper_exists():
    """A grid container must wrap the controls + visual columns."""
    body = _read()
    assert 'class="demo-grid"' in body or "class='demo-grid'" in body, (
        "expected a div with class=\"demo-grid\" wrapping the controls + visual columns"
    )


def test_points_slider_max_is_100000():
    body = _read()
    match = re.search(r'id="ctl-n"[^>]*\bmax="(\d+)"', body)
    assert match, "could not find #ctl-n slider"
    assert int(match.group(1)) == 100_000, f"points max was {match.group(1)}, expected 100000"


def test_maxiter_slider_max_is_1000():
    body = _read()
    match = re.search(r'id="ctl-maxiter"[^>]*\bmax="(\d+)"', body)
    assert match, "could not find #ctl-maxiter slider"
    assert int(match.group(1)) == 1_000, f"max-iter max was {match.group(1)}, expected 1000"


def test_inertia_canvas_appears_after_main_canvas():
    """In the new layout the inertia chart sits below the main visual."""
    body = _read()
    i_main    = body.find('id="demo-canvas"')
    i_inertia = body.find('id="inertia-canvas"')
    assert i_main    != -1, "main demo canvas missing"
    assert i_inertia != -1, "inertia canvas missing"
    assert i_main < i_inertia, "main canvas must appear before inertia canvas in source"
