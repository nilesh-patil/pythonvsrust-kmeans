"""Shared visual encodings for implementation-level charts."""

from __future__ import annotations

from math import ceil, floor, isclose, log2
from typing import Iterable, Sequence

IMPLEMENTATION_ORDER: tuple[str, ...] = ("python", "rust", "rust_parallel", "sklearn")

DISPLAY_NAMES: dict[str, str] = {
    "python": "Python",
    "rust": "Rust",
    "rust_parallel": "Rust - Parallel",
    "sklearn": "scikit-learn",
}

# Colorblind-aware, high-separation categorical palette.
IMPL_COLORS: dict[str, str] = {
    "python": "#0072B2",
    "rust": "#D55E00",
    "rust_parallel": "#CC79A7",
    "sklearn": "#009E73",
}

IMPL_SYMBOLS_PLOTLY: dict[str, str] = {
    "python": "circle",
    "rust": "diamond",
    "rust_parallel": "square",
    "sklearn": "triangle-up",
}

IMPL_MARKERS_MPL: dict[str, str] = {
    "python": "o",
    "rust": "D",
    "rust_parallel": "s",
    "sklearn": "^",
}

IMPL_HATCHES_MPL: dict[str, str] = {
    "python": "",
    "rust": "//",
    "rust_parallel": "xx",
    "sklearn": "..",
}

IMPL_PATTERNS_PLOTLY: dict[str, str] = {
    "python": "",
    "rust": "/",
    "rust_parallel": "x",
    "sklearn": ".",
}


def display_name(impl: str) -> str:
    return DISPLAY_NAMES.get(impl.lower(), impl)


def implementation_from_display(display: str) -> str:
    for impl, name in DISPLAY_NAMES.items():
        if display == name:
            return impl
    return display.lower().replace(" ", "_").replace("-", "_")


def color(impl: str) -> str:
    return IMPL_COLORS.get(impl.lower(), "#666666")


def plotly_symbol(impl: str) -> str:
    return IMPL_SYMBOLS_PLOTLY.get(impl.lower(), "circle-open")


def mpl_marker(impl: str) -> str:
    return IMPL_MARKERS_MPL.get(impl.lower(), "o")


def mpl_hatch(impl: str) -> str:
    return IMPL_HATCHES_MPL.get(impl.lower(), "")


def plotly_pattern(impl: str) -> str:
    return IMPL_PATTERNS_PLOTLY.get(impl.lower(), "")


def ordered_implementations(values: Iterable[str]) -> list[str]:
    seen = set(values)
    ordered = [impl for impl in IMPLEMENTATION_ORDER if impl in seen]
    ordered.extend(sorted(seen.difference(IMPLEMENTATION_ORDER)))
    return ordered


def log2_tick_values(values: Sequence[float] | Iterable[float]) -> list[float]:
    positives = [float(value) for value in values if float(value) > 0]
    if not positives:
        return [1.0]

    start = floor(log2(min(positives)))
    end = ceil(log2(max(positives)))
    return [2.0 ** exp for exp in range(start, end + 1)]


def log2_tick_text(tick_values: Sequence[float]) -> list[str]:
    labels: list[str] = []
    for value in tick_values:
        exponent = round(log2(float(value))) if value > 0 else 0
        if value > 0 and isclose(float(value), 2.0 ** exponent, rel_tol=1e-9):
            labels.append(f"2^{exponent}")
        elif value >= 1:
            labels.append(f"{value:g}")
        else:
            labels.append(f"{value:.4f}".rstrip("0").rstrip("."))
    return labels
