#!/usr/bin/env python3
"""Visualize Rust parallel scaling data produced by bench_parallel_scaling.py."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from viz_style import (
        INK_FAINT,
        apply_mpl_style,
        color,
        display_name,
        log2_tick_text,
        log2_tick_values,
        mpl_linestyle,
        mpl_marker,
        si_label,
        style_axes,
    )
except ImportError:
    from src.viz_style import (
        INK_FAINT,
        apply_mpl_style,
        color,
        display_name,
        log2_tick_text,
        log2_tick_values,
        mpl_linestyle,
        mpl_marker,
        si_label,
        style_axes,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = REPO_ROOT / "results" / "parallel_scaling.csv"
CSV_PATH = DEFAULT_CSV_PATH
SCALE_DIR = REPO_ROOT / "results"
SCALE_PATTERN = "parallel_scaling_n*.csv"
OUT_PATH = REPO_ROOT / "results" / "parallel_scaling.svg"


def _sample_size_from_path(path: Path) -> int:
    match = re.fullmatch(r"parallel_scaling_n(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"cannot infer sample size from {path}")
    return int(match.group(1))


def _style_axes(*axes: plt.Axes) -> None:
    style_axes(*axes)


def _format_sample_ticks(ax: plt.Axes, samples: pd.Series) -> None:
    ticks = [int(value) for value in sorted(samples.unique())]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([si_label(value) for value in ticks])


def _format_log2_y(ax: plt.Axes, values: pd.Series | list[float]) -> None:
    ticks = log2_tick_values(values)
    ax.set_yscale("log", base=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([si_label(value) for value in ticks])


def _load_scale_sweep(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["n_samples"] = _sample_size_from_path(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _plot_scale_sweep(df: pd.DataFrame) -> None:
    df = df.sort_values(["n_samples", "threads"]).reset_index(drop=True)
    serial = df[df["threads"] == 0].sort_values("n_samples")
    parallel = df[df["threads"] > 0].copy()
    best = (
        parallel.sort_values(["n_samples", "median_s", "threads"])
        .groupby("n_samples", as_index=False)
        .first()
        .sort_values("n_samples")
    )

    apply_mpl_style()
    fig, (ax_t, ax_s, ax_threads) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Runtime by scale: the main "doubling work" story.
    ax_t.plot(
        serial["n_samples"],
        serial["median_s"],
        marker=mpl_marker("rust"),
        color=color("rust"),
        lw=1.8,
        ls=mpl_linestyle("rust"),
        label=display_name("rust"),
    )
    ax_t.plot(
        best["n_samples"],
        best["median_s"],
        marker=mpl_marker("rust_parallel"),
        color=color("rust_parallel"),
        lw=1.8,
        ls=mpl_linestyle("rust_parallel"),
        label=f"{display_name('rust_parallel')} best",
    )
    ax_t.fill_between(
        best["n_samples"],
        best["min_s"],
        best["max_s"],
        color=color("rust_parallel"),
        alpha=0.18,
        linewidth=0,
    )
    _format_sample_ticks(ax_t, serial["n_samples"])
    _format_log2_y(
        ax_t,
        list(serial["median_s"])
        + list(best["median_s"])
        + list(best["min_s"])
        + list(best["max_s"]),
    )
    ax_t.set_xlabel("Samples (doubling sequence)")
    ax_t.set_ylabel("Wall-clock runtime (s)")
    ax_t.set_title("Runtime grows with sample scale")
    ax_t.annotate(
        "Serial Rust",
        xy=(serial["n_samples"].iloc[-1], serial["median_s"].iloc[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        color=color("rust"),
        fontsize=9,
    )
    ax_t.annotate(
        "Best parallel",
        xy=(best["n_samples"].iloc[-1], best["median_s"].iloc[-1]),
        xytext=(5, -11),
        textcoords="offset points",
        va="center",
        color=color("rust_parallel"),
        fontsize=9,
    )

    # Best speedup by scale: small gain, shown honestly around the 1x baseline.
    ax_s.axhline(1.0, ls=(0, (2, 2)), color=INK_FAINT, lw=1.0, alpha=0.7)
    ax_s.plot(
        best["n_samples"],
        best["speedup"],
        marker=mpl_marker("rust_parallel"),
        color=color("rust_parallel"),
        ls=mpl_linestyle("rust_parallel"),
        lw=1.8,
    )
    _format_sample_ticks(ax_s, best["n_samples"])
    ax_s.set_ylim(0.95, max(1.35, float(best["speedup"].max()) * 1.05))
    ax_s.set_xlabel("Samples (doubling sequence)")
    ax_s.set_ylabel("Best speedup vs serial Rust")
    ax_s.set_title("Parallel speedup remains bounded")
    ax_s.annotate(
        f"peak {best['speedup'].max():.2f}x",
        xy=(
            best.loc[best["speedup"].idxmax(), "n_samples"],
            best["speedup"].max(),
        ),
        xytext=(4, 8),
        textcoords="offset points",
        color=color("rust_parallel"),
        fontsize=9,
    )

    # Thread-count resource use on the larger workloads.
    selected_scales = [value for value in (64_000, 128_000, 256_000) if value in set(df["n_samples"])]
    # Scale categories (not implementations): a muted ink-to-rust ramp.
    scale_styles = {
        64_000: ("#9a8f80", "o"),
        128_000: ("#b7410e", "s"),
        256_000: ("#7a2e0c", "^"),
    }
    for n_samples in selected_scales:
        subset = parallel[parallel["n_samples"] == n_samples].sort_values("threads")
        line_color, marker = scale_styles[n_samples]
        ax_threads.plot(
            subset["threads"],
            subset["speedup"],
            marker=marker,
            color=line_color,
            lw=1.8,
        )
        ax_threads.annotate(
            f"{n_samples // 1000}k",
            xy=(subset["threads"].iloc[-1], subset["speedup"].iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=line_color,
            fontsize=9,
        )
    if selected_scales:
        max_thread = int(parallel["threads"].max())
        ax_threads.plot([1, max_thread], [1, max_thread], ls=(0, (1, 3)), color=INK_FAINT, lw=1.0, alpha=0.6)
    ax_threads.axhline(1.0, ls=(0, (2, 2)), color=INK_FAINT, lw=1.0, alpha=0.7)
    ax_threads.set_xlabel("Rayon threads (log2 scale)")
    ax_threads.set_ylabel("Speedup vs serial Rust")
    ax_threads.set_title("Large-scale thread sweep")
    ax_threads.set_xscale("log", base=2)
    thread_ticks = sorted(int(value) for value in parallel["threads"].unique())
    ax_threads.set_xticks(thread_ticks)
    ax_threads.set_xticklabels([str(value) for value in thread_ticks])
    ax_threads.set_ylim(0.65, max(1.45, float(parallel["speedup"].max()) * 1.05))

    _style_axes(ax_t, ax_s, ax_threads)
    fig.suptitle("Rust k-means parallel scaling across sample sizes", y=1.03, fontsize=12)
    fig.text(
        0.5,
        0.0,
        "Source: results/parallel_scaling_n*.csv · Rust-only k-sweep · n_features=32 · k_max=32",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK_FAINT,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


def _plot_single_sweep(df: pd.DataFrame) -> None:
    serial = df[df["threads"] == 0].iloc[0]
    par = df[df["threads"] > 0].sort_values("threads").reset_index(drop=True)

    # Speedup is computed two ways:
    #   * vs serial-Rust baseline (the honest end-to-end comparison)
    #   * vs parallel-1-thread (isolates pure parallel scaling, excluding rayon overhead)
    par["speedup_vs_serial"] = serial["median_s"] / par["median_s"]
    par_1t = par[par["threads"] == 1]["median_s"].iloc[0]
    par["speedup_vs_parallel1"] = par_1t / par["median_s"]

    apply_mpl_style()
    fig, (ax_t, ax_s, ax_eff) = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- Runtime panel
    ax_t.plot(
        par["threads"],
        [serial["median_s"]] * len(par),
        marker=mpl_marker("rust"),
        color=color("rust"),
        ls="--",
        lw=1.5,
        label=f"{display_name('rust')} baseline ({serial['median_s']:.2f}s)",
    )
    ax_t.plot(
        par["threads"],
        par["median_s"],
        marker=mpl_marker("rust_parallel"),
        lw=2,
        color=color("rust_parallel"),
        label=display_name("rust_parallel"),
    )
    ax_t.fill_between(par["threads"], par["min_s"], par["max_s"],
                      color=color("rust_parallel"), alpha=0.22, label="min/max")
    ax_t.set_xlabel("Threads")
    ax_t.set_ylabel("Wall-clock runtime (s, log2 scale)")
    ax_t.set_title("Runtime vs thread count")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log", base=2)
    ax_t.set_xticks(par["threads"])
    ax_t.set_xticklabels([str(t) for t in par["threads"]])
    ax_t.grid(True, ls=":", alpha=0.6)
    ax_t.legend()

    # --- Speedup panel
    max_t = int(par["threads"].max())
    ax_s.plot([1, max_t], [1, max_t], ls="--", color="#9ca3af",
              label="ideal (linear)")
    ax_s.plot(
        par["threads"],
        par["speedup_vs_serial"],
        marker=mpl_marker("rust"),
        lw=2,
        color=color("rust"),
        label=f"vs {display_name('rust')}",
    )
    ax_s.plot(
        par["threads"],
        par["speedup_vs_parallel1"],
        marker=mpl_marker("rust_parallel"),
        lw=2,
        color=color("rust_parallel"),
        label=f"vs {display_name('rust_parallel')} @ 1 thread",
    )

    # os.cpu_count() reports the available logical CPU count.
    cpu_count = os.cpu_count() or 1
    ax_s.axvline(cpu_count, ls=":", color="#6b7280", lw=1.4)
    ymax = max(float(par["speedup_vs_serial"].max()), float(par["speedup_vs_parallel1"].max()))
    ax_s.annotate(
        "available CPU count",
        xy=(cpu_count, ymax),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6b7280",
        arrowprops=dict(arrowstyle="-", color="#6b7280", lw=0.8),
    )

    # Efficiency curve (speedup / threads) gets its own panel; no dual y-axis.
    par["efficiency"] = par["speedup_vs_parallel1"] / par["threads"]
    ax_eff.plot(par["threads"], par["efficiency"], marker="^", lw=1.5,
                color="#6b7280", ls="--", label="efficiency")
    ax_eff.set_xlabel("Threads")
    ax_eff.set_ylabel("Efficiency (speedup / threads)")
    ax_eff.set_title("Parallel efficiency")
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_xticks(par["threads"])
    ax_eff.set_xticklabels([str(t) for t in par["threads"]])
    ax_eff.set_ylim(0, 1.4)
    ax_eff.axhline(1.0, ls=":", color="#6b7280", lw=0.8, alpha=0.5)
    ax_eff.grid(True, ls=":", alpha=0.6)
    ax_eff.legend(fontsize=8)

    ax_s.set_xlabel("Threads")
    ax_s.set_ylabel("Speedup")
    ax_s.set_title("Speedup vs thread count")
    ax_s.set_xscale("log", base=2)
    ax_s.set_yscale("log", base=2)
    ax_s.set_xticks(par["threads"])
    ax_s.set_xticklabels([str(t) for t in par["threads"]])
    ax_s.grid(True, ls=":", alpha=0.6)
    ax_s.legend(fontsize=8)

    fig.suptitle("Rust k-means parallel scaling (Rayon)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


def main() -> None:
    scale_paths = sorted(
        SCALE_DIR.glob(SCALE_PATTERN),
        key=_sample_size_from_path,
    )
    if Path(CSV_PATH) == DEFAULT_CSV_PATH and scale_paths:
        _plot_scale_sweep(_load_scale_sweep(scale_paths))
        return
    if not CSV_PATH.exists():
        sys.exit(f"missing {CSV_PATH} — run src/bench_parallel_scaling.py first")
    _plot_single_sweep(pd.read_csv(CSV_PATH))


if __name__ == "__main__":
    main()
