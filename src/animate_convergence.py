#!/usr/bin/env python3
"""
Animate Lloyd's K-Means iterations on 2-D data.

Reimplements Lloyd's loop here (rather than reusing KMeansClustering's fit
directly) so we can record centroid + label history at each step without
polluting the production class. The traced run mirrors KMeansClustering's
control flow exactly, and a test asserts both produce identical final states.
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
from sklearn.datasets import make_blobs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "python_impl"))

from kmeans import KMeansClustering  # noqa: E402

PALETTE = ["#0ea5e9", "#dc2626", "#16a34a", "#a855f7",
           "#f59e0b", "#0891b2", "#db2777", "#65a30d"]


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
            title: str | None = None, fps: int = 2) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    colors = [PALETTE[c % len(PALETTE)] for c in range(k)]
    scatter = ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.55, edgecolor="none")
    centroid_dots = ax.scatter([], [], s=320, marker="X",
                                edgecolor="black", linewidth=1.8, zorder=5)
    trail_lines = [ax.plot([], [], "-", color=colors[c],
                            alpha=0.35, lw=1)[0] for c in range(k)]

    pad = 1.0
    ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad)
    ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    base_title = title or f"Lloyd's K-Means — init={trace.init}, k={k}"

    # Hold the converged state on the last few frames for readability.
    hold = 3
    n_frames = len(trace.centroid_history) + hold

    def frame(i: int):
        step = min(i, len(trace.centroid_history) - 1)
        centroids = trace.centroid_history[step]
        labels    = trace.labels_history[step]
        inertia   = trace.inertia_history[step]

        scatter.set_color([colors[c] for c in labels])
        centroid_dots.set_offsets(centroids)
        centroid_dots.set_color(colors[:k])

        for c in range(k):
            xs = [trace.centroid_history[s][c, 0] for s in range(step + 1)]
            ys = [trace.centroid_history[s][c, 1] for s in range(step + 1)]
            trail_lines[c].set_data(xs, ys)

        status = "converged" if step == trace.iterations_run else f"iter {step}"
        ax.set_title(f"{base_title}\n{status} · inertia = {inertia:,.1f}",
                     fontsize=11)
        return [scatter, centroid_dots, *trail_lines]

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=1000 / fps,
                         blit=False, repeat=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
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

    trace_rand = run_lloyds(X_easy, k=4, init="random",     seed=2,  max_iter=30)
    trace_kpp  = run_lloyds(X_easy, k=4, init="k-means++",  seed=42, max_iter=30)

    animate(trace_rand, X_easy, k=4,
            out_path=args.output_dir / "convergence_random.gif",
            title="Lloyd's K-Means — random init", fps=args.fps)
    animate(trace_kpp,  X_easy, k=4,
            out_path=args.output_dir / "convergence_kpp.gif",
            title="Lloyd's K-Means — k-means++ init", fps=args.fps)

    # Pathological seed for random init: known to put two centroids in the
    # same blob and require many iterations to recover.
    X_path, _ = make_blobs(n_samples=400, n_features=2, centers=4,
                           cluster_std=0.8, random_state=10)
    trace_bad = run_lloyds(X_path, k=4, init="random", seed=1, max_iter=30)
    animate(trace_bad, X_path, k=4,
            out_path=args.output_dir / "convergence_pathological.gif",
            title="random init — pathological seed", fps=args.fps)


if __name__ == "__main__":
    main()
