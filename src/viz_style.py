"""Shared visual encodings and Tufte editorial styling for the charts.

All implementation-level matplotlib and Plotly artifacts pull their colors,
fonts, and axis chrome from here so the figure family stays coherent: paper
off-white background, serif type, left+bottom spines only, ultra-light y-grid,
direct line-end labels, and Rust-Parallel rendered as a dashed line of the same
rust family.
"""

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

# Tufte editorial implementation palette (DESIGN_BRIEF tokens).
IMPL_COLORS: dict[str, str] = {
    "python": "#3d6b9e",        # steel blue
    "rust": "#b7410e",          # rust
    "rust_parallel": "#7a2e0c",  # dark rust (dashed in charts)
    "sklearn": "#c98c1f",       # muted ochre
}

# Editorial paper/ink tokens shared across every chart.
PAPER: str = "#fffff8"
INK: str = "#151515"
INK_FAINT: str = "#595959"
SPINE: str = "#999999"
GRID: str = "#eeeeee"
RULE: str = "#dcdcd4"
ACCENT: str = "#a32015"

# Rust-Parallel is the only implementation drawn dashed.
IMPL_LINESTYLES_MPL: dict[str, str | tuple] = {
    "python": "-",
    "rust": "-",
    "rust_parallel": (0, (5, 2)),
    "sklearn": "-",
}

IMPL_DASH_PLOTLY: dict[str, str] = {
    "python": "solid",
    "rust": "solid",
    "rust_parallel": "dash",
    "sklearn": "solid",
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


def mpl_linestyle(impl: str):
    """Matplotlib linestyle; Rust-Parallel is the only dashed line."""
    return IMPL_LINESTYLES_MPL.get(impl.lower(), "-")


def plotly_dash(impl: str) -> str:
    """Plotly line dash style; Rust-Parallel is the only dashed line."""
    return IMPL_DASH_PLOTLY.get(impl.lower(), "solid")


# ── matplotlib editorial style ────────────────────────────────────────────────

MPL_RC: dict[str, object] = {
    "figure.facecolor": PAPER,
    "figure.edgecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.edgecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.family": "serif",
    "font.serif": ["Newsreader", "Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "text.color": INK,
    "xtick.color": INK_FAINT,
    "ytick.color": INK_FAINT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.titleweight": "normal",
    # ── SVG output ────────────────────────────────────────────────────────────
    # Keep text as real <text> elements (not paths) so the site's Newsreader /
    # JetBrains Mono webfonts render the labels and they stay selectable. The
    # font-family attribute matplotlib writes is the resolved family name; the
    # serif stack above degrades gracefully when Newsreader is unavailable.
    "svg.fonttype": "none",
    # Stable element ids across regenerations keep diffs small and reviewable.
    "svg.hashsalt": "pythonvsrust-kmeans",
}


def apply_mpl_style() -> None:
    """Install the shared Tufte rcParams. Call once at the top of each script."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(MPL_RC)


def style_axes(*axes, ygrid: bool = True) -> None:
    """Left+bottom spines only; an ultra-light dashed y-grid behind the data."""
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE)
        ax.spines["bottom"].set_color(SPINE)
        ax.set_axisbelow(True)
        if ygrid:
            ax.grid(True, axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=1.0)
            ax.grid(False, axis="x")
        else:
            ax.grid(False)


def end_label(ax, x, y, text: str, impl_color: str, *, dx: int = 6, dy: int = 0, **kwargs) -> None:
    """Direct line-end label in the implementation color, replacing a legend entry."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=kwargs.pop("fontsize", 9),
        color=impl_color,
        clip_on=False,
        **kwargs,
    )


# ── SI tick labels for log axes (1k / 10k / 100k, not 10^3) ───────────────────

def si_label(value: float) -> str:
    """Format a number as a plain magnitude label: 1k, 10k, 1M, 0.5, etc."""
    v = float(value)
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e9:
        s, suffix = v / 1e9, "B"
    elif av >= 1e6:
        s, suffix = v / 1e6, "M"
    elif av >= 1e3:
        s, suffix = v / 1e3, "k"
    else:
        # Sub-thousand: trim trailing zeros, keep small fractions readable.
        text = f"{v:.4f}".rstrip("0").rstrip(".")
        return text if text else "0"
    text = f"{s:.1f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def si_labels(values: Sequence[float]) -> list[str]:
    return [si_label(v) for v in values]


def si_log_axis(ax, which: str = "both") -> None:
    """Replace 10^n / 2^n tick labels with plain magnitudes (1k, 10k, 1M, 0.5).

    Uses a FuncFormatter so labels stay attached to whatever ticks matplotlib
    chooses for the log locator, even after relayout.
    """
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _pos: si_label(v) if v > 0 else "")
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _pos: ""))
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _pos: ""))


# ── Plotly editorial layout ───────────────────────────────────────────────────

PLOTLY_FONT_FAMILY = "Newsreader, Georgia, 'Times New Roman', serif"


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
