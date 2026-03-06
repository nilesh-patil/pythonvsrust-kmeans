"""Tests for Rust-Parallel benchmark split (spec: march_rust_parallel_bench.md).

TDD red phase — these must FAIL before the implementation is in place.
"""

import sys
from pathlib import Path

# Ensure both the project root and src/ are importable.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest


# ---------------------------------------------------------------------------
# Test 1 — BenchmarkRunner has run_rust_parallel_impl
# ---------------------------------------------------------------------------

def test_runner_has_run_rust_parallel_impl():
    """BenchmarkRunner must expose run_rust_parallel_impl as a callable method."""
    from runner import BenchmarkRunner

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    assert hasattr(runner, "run_rust_parallel_impl"), (
        "BenchmarkRunner is missing run_rust_parallel_impl"
    )
    assert callable(runner.run_rust_parallel_impl), (
        "run_rust_parallel_impl must be callable"
    )


# ---------------------------------------------------------------------------
# Test 2 — dashboard display-name map includes rust_parallel
# ---------------------------------------------------------------------------

def test_dashboard_label_map_includes_rust_parallel():
    """build_dashboard.DISPLAY_NAMES must map 'rust_parallel' -> 'Rust - Parallel'."""
    from build_dashboard import DISPLAY_NAMES

    assert "rust_parallel" in DISPLAY_NAMES, (
        "DISPLAY_NAMES in build_dashboard is missing 'rust_parallel'"
    )
    assert DISPLAY_NAMES["rust_parallel"] == "Rust - Parallel", (
        f"Expected 'Rust - Parallel', got {DISPLAY_NAMES['rust_parallel']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — dashboard colour palette includes rust_parallel
# ---------------------------------------------------------------------------

def test_dashboard_color_map_includes_rust_parallel():
    """IMPL_COLORS in build_dashboard must contain a hex entry for 'rust_parallel'."""
    from build_dashboard import IMPL_COLORS

    assert "rust_parallel" in IMPL_COLORS, (
        "IMPL_COLORS in build_dashboard is missing 'rust_parallel'"
    )
    color = IMPL_COLORS["rust_parallel"]
    assert color.startswith("#") and len(color) == 7, (
        f"Expected a 7-char hex color, got {color!r}"
    )
