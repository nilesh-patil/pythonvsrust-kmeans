#!/usr/bin/env python3
"""
Animate Lloyd's K-Means iterations on 2-D data.

Reimplements Lloyd's loop here (rather than reusing KMeansClustering's fit
directly) so we can record centroid + label history at each step without
polluting the production class. The traced run mirrors KMeansClustering's
control flow exactly, and a test asserts both produce identical final states.

The five GIFs are styled through src/viz_style.py so they sit in the same
editorial family as the static charts: paper off-white background, serif type,
a muted print-like categorical palette (the demo.js PALETTE family), accent
centroid crosshairs, and an unobtrusive inertia inset.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.datasets import make_blobs, make_circles, make_moons

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "python_impl"))

from kmeans import KMeansClustering  # noqa: E402

from viz_style import (  # noqa: E402
    ACCENT,
    INK,
    INK_FAINT,
    PAPER,
    RULE,
    SPINE,
    apply_mpl_style,
)

# Muted, print-like categorical palette — kept byte-identical to docs/assets/js/
# demo.js so the GIFs and the live in-browser demo color clusters the same way:
# steel blue, rust, ochre, sage, dark rust, muted purple, terracotta, slate.
PALETTE = [
    "#3d6b9e", "#b7410e", "#c98c1f", "#5b7553",
    "#7a2e0c", "#6a5687", "#9e5b3d", "#4f7a8c",
]

# Rendering geometry. ~1050 px square at dpi=150 reads crisply at any size the
# 5-across strip (or a click-through) puts the GIF at — comfortably past the
# ~720 px readable target and roughly double the previous effective resolution.
FIG_INCHES = 7.0
DPI = 150


@dataclass
class LloydsTrace:
    centroid_history: list[np.ndarray] = field(default_factory=list)
    labels_history:   list[np.ndarray] = field(default_factory=list)
    inertia_history:  list[float]      = field(default_factory=list)
    iterations_run:   int              = 0
    init:             str              = "random"
    seed:             int              = 0


def _inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    diffs = X - centroids[labels]
    return float(np.einsum("ij,ij->", diffs, diffs))


def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diffs = X[:, None, :] - centroids[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
    return np.argmin(d2, axis=1)


def _update(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centroids = np.empty((k, X.shape[1]))
    for c in range(k):
        members = X[labels == c]
        centroids[c] = members.mean(axis=0) if len(members) else X[np.random.randint(len(X))]
    return centroids


def run_lloyds(X: np.ndarray, k: int, init: str = "random",
               seed: int = 42, max_iter: int = 300) -> LloydsTrace:
    """Run Lloyd's K-Means recording state per iteration."""
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("animator requires 2-D input (X.shape[1] == 2)")

    # Delegate initialisation to KMeansClustering so the trace's seed matches
    # the production class — that's what the equivalence test relies on.
    km = KMeansClustering(n_clusters=k, init=init, random_state=seed, max_iter=1)
    rs = np.random.RandomState(seed)
    if init == "k-means++":
        # KMeansClustering.__init__ already seeded its own RandomState, but
        # we need to re-derive the initial centroids without running fit() — so
        # mirror the dispatch.
        centroids = km._initialize_centroids_kpp(X)  # noqa: SLF001
    else:
        indices = rs.choice(X.shape[0], k, replace=False)
        centroids = X[indices].copy()

    trace = LloydsTrace(init=init, seed=seed)
    labels = _assign(X, centroids)
    trace.centroid_history.append(centroids.copy())
    trace.labels_history.append(labels.copy())
    trace.inertia_history.append(_inertia(X, centroids, labels))

    for it in range(max_iter):
        new_centroids = _update(X, labels, k)
        new_labels    = _assign(X, new_centroids)

        trace.centroid_history.append(new_centroids.copy())
        trace.labels_history.append(new_labels.copy())
        trace.inertia_history.append(_inertia(X, new_centroids, new_labels))
        trace.iterations_run = it + 1

        if np.array_equal(new_labels, labels):
            break
        centroids = new_centroids
        labels    = new_labels

    return trace


def animate(trace: LloydsTrace, X: np.ndarray, k: int, out_path: Path,
            title: str | None = None, fps: int = 2, hold: int = 5) -> None:
    """Render a trace to a GIF.

    `hold` repeats the converged final frame so the answer lingers; identical
    repeated frames cost almost nothing in the GIF since they compress away.
    """
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    colors = [PALETTE[c % len(PALETTE)] for c in range(k)]
    scatter = ax.scatter(X[:, 0], X[:, 1], s=22, alpha=0.7, edgecolor="none", zorder=2)
    # Centroid crosshairs: an ink-outlined accent "X" so the moving markers stay
    # legible over any cluster color.
    centroid_dots = ax.scatter([], [], s=420, marker="X", facecolor=ACCENT,
                               edgecolor=INK, linewidth=1.6, zorder=6)
    trail_lines = [ax.plot([], [], "-", color=INK_FAINT,
                           alpha=0.45, lw=1.1, zorder=5)[0] for _ in range(k)]

    pad = 1.0
    ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad)
    ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    # Minimal chrome: drop every spine, the scatter carries the frame itself.
    for spine in ax.spines.values():
        spine.set_visible(False)
    base_title = title or f"Lloyd's K-Means — init={trace.init}, k={k}"

    # Inertia-vs-iteration inset, top-right, kept editorial: paper face, a hair-
    # line frame, a faint full curve for context, and an accent progress line.
    ax_inset = inset_axes(ax, width="34%", height="26%", loc="upper right",
                          borderpad=1.3)
    ax_inset.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax_inset.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_inset.spines[side].set_color(SPINE)
        ax_inset.spines[side].set_linewidth(0.7)
    ax_inset.tick_params(labelsize=7, length=2, colors=INK_FAINT)
    ax_inset.set_xlabel("iteration", fontsize=7, color=INK_FAINT)
    ax_inset.set_title("inertia", fontsize=8, color=INK, pad=3)
    all_iters = list(range(len(trace.inertia_history)))
    ax_inset.plot(all_iters, trace.inertia_history, color=RULE, lw=1.0, zorder=1)
    (inset_line,) = ax_inset.plot([], [], color=ACCENT, lw=1.6, zorder=2)
    (inset_head,) = ax_inset.plot([], [], "o", color=ACCENT, ms=3.5, zorder=3)

    n_steps = len(trace.centroid_history)
    n_frames = n_steps + hold

    def frame(i: int):
        step = min(i, n_steps - 1)
        centroids = trace.centroid_history[step]
        labels    = trace.labels_history[step]
        inertia   = trace.inertia_history[step]

        scatter.set_color([colors[c] for c in labels])
        centroid_dots.set_offsets(centroids)

        for c in range(k):
            xs = [trace.centroid_history[s][c, 0] for s in range(step + 1)]
            ys = [trace.centroid_history[s][c, 1] for s in range(step + 1)]
            trail_lines[c].set_data(xs, ys)

        inset_line.set_data(all_iters[: step + 1], trace.inertia_history[: step + 1])
        inset_head.set_data([all_iters[step]], [trace.inertia_history[step]])

        status = "converged" if step == trace.iterations_run else f"iteration {step}"
        ax.set_title(f"{base_title}\n{status}  ·  inertia = {inertia:,.1f}",
                     fontsize=13, color=INK)
        return [scatter, centroid_dots, *trail_lines, inset_line, inset_head]

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=1000 / fps,
                         blit=False, repeat=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=DPI,
              savefig_kwargs={"facecolor": PAPER})
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "results" / "animations")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args(argv)

    # Well-separated 2D blobs.
    X_easy, _ = make_blobs(n_samples=400, n_features=2, centers=4,
                           cluster_std=0.8, random_state=42)

    # seed=2 (the historical pick) actually strands two centroids in one blob —
    # it looks identical to the pathological panel and undercuts the strip's
    # good→bad→fixed contrast. seed=11 converges cleanly to all four blobs in
    # four iterations (ARI 1.0), which is what the "random init" caption means
    # by "converging from a random initialization on Gaussian blobs".
    trace_rand = run_lloyds(X_easy, k=4, init="random",     seed=11, max_iter=30)
    trace_kpp  = run_lloyds(X_easy, k=4, init="k-means++",  seed=42, max_iter=30)

    animate(trace_rand, X_easy, k=4,
            out_path=args.output_dir / "convergence_random.gif",
            title="Lloyd's K-Means — random init", fps=args.fps)
    animate(trace_kpp,  X_easy, k=4,
            out_path=args.output_dir / "convergence_kpp.gif",
            title="Lloyd's K-Means — k-means++ init", fps=args.fps)

    # Pathological seed for random init on the random_state=10 blobs: this seed
    # strands two centroids in one blob (blob 2) while a real cluster (blob 1)
    # goes unseeded, and grinds for nine iterations before settling there —
    # exactly the failure the algorithms.md caption describes. (The dataset seed
    # stays fixed; only the init seed selects this trajectory.)
    X_path, _ = make_blobs(n_samples=400, n_features=2, centers=4,
                           cluster_std=0.8, random_state=10)
    trace_bad = run_lloyds(X_path, k=4, init="random", seed=5, max_iter=30)
    animate(trace_bad, X_path, k=4,
            out_path=args.output_dir / "convergence_pathological.gif",
            title="random init — pathological seed", fps=args.fps)

    # make_moons: two interleaved crescent shapes — k-means bisects them
    # incorrectly because they aren't convex.  k=2 since there are two moons.
    X_moons, _ = make_moons(n_samples=300, noise=0.10, random_state=42)
    trace_moons = run_lloyds(X_moons, k=2, init="random", seed=7, max_iter=15)
    animate(trace_moons, X_moons, k=2,
            out_path=args.output_dir / "convergence_moons.gif",
            title="make_moons — k-means failure mode", fps=args.fps)

    # make_circles: two concentric rings — k-means pie-slices them instead of
    # separating inner from outer, a classic non-convex failure mode.
    X_circles, _ = make_circles(n_samples=260, noise=0.06, factor=0.5, random_state=42)
    trace_circles = run_lloyds(X_circles, k=2, init="random", seed=7, max_iter=15)
    animate(trace_circles, X_circles, k=2,
            out_path=args.output_dir / "convergence_circles.gif",
            title="make_circles — k-means failure mode", fps=args.fps)


if __name__ == "__main__":
    main()
