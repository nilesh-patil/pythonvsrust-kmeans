#!/usr/bin/env python3
"""
Visualize inertia: random vs k-means++ initialization
across three dataset sizes, using the pure-Python KMeansClustering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "python_impl"))

from kmeans import KMeansClustering  # noqa: E402


DATASETS = [
    ("small",  1_000,  2,  4),
    ("medium", 8_000,  8, 16),
    ("large", 32_000, 16, 32),
]
N_RUNS = 10
BASE_SEED = 42


def run_one(X: np.ndarray, k: int, init: str, seed: int) -> float:
    """Fit once and return the inertia."""
    km = KMeansClustering(n_clusters=k, init=init, random_state=seed)
    km.fit(X)
    diffs = X - km.centroids[km.labels]
    return float(np.einsum("ij,ij->", diffs, diffs))


def main() -> None:
    results: dict[str, dict[str, list[float]]] = {}

    for name, n_samples, n_features, k in DATASETS:
        print(f"[{name}] n={n_samples} d={n_features} k={k}")
        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=k,
            cluster_std=1.0,
            random_state=BASE_SEED,
        )
        rand_inertias = [run_one(X, k, "random",     BASE_SEED + i) for i in range(N_RUNS)]
        kpp_inertias  = [run_one(X, k, "k-means++",  BASE_SEED + i) for i in range(N_RUNS)]
        results[name] = {"random": rand_inertias, "k-means++": kpp_inertias}
        print(f"  random   mean inertia: {np.mean(rand_inertias):,.1f}")
        print(f"  k-means++ mean inertia: {np.mean(kpp_inertias):,.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [d[0] for d in DATASETS]
    x = np.arange(len(names))
    width = 0.35

    rand_means = [np.mean(results[n]["random"])    for n in names]
    rand_stds  = [np.std (results[n]["random"])    for n in names]
    kpp_means  = [np.mean(results[n]["k-means++"]) for n in names]
    kpp_stds   = [np.std (results[n]["k-means++"]) for n in names]

    ax.bar(x - width / 2, rand_means, width, yerr=rand_stds,
           label="random", color="#d97706", capsize=4)
    ax.bar(x + width / 2, kpp_means,  width, yerr=kpp_stds,
           label="k-means++", color="#0ea5e9", capsize=4)

    # Annotate the ratio (random / k-means++) above each bar pair.
    # On a log scale the "top" of a bar is its mean value; we place the
    # annotation just above the taller of the two bars.
    for i, (rm, km) in enumerate(zip(rand_means, kpp_means)):
        ratio = rm / km if km > 0 else float("nan")
        top = max(rm, km)
        ax.annotate(
            f"{ratio:.1f}×",
            xy=(x[i], top),
            xytext=(x[i], top * 1.25),  # slightly above on the log axis
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={s:,}, d={d}, k={k})"
                        for (n, s, d, k) in DATASETS])
    ax.set_ylabel("Mean inertia (lower is better)")
    ax.set_title(f"K-Means initialization comparison · {N_RUNS} seeds per bar")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Footer note — placed in figure coordinates so it sits below the axes.
    fig.text(0.5, 0.01, "Lower is better · 10 seeds per bar",
             ha="center", va="bottom", fontsize=8, color="#888")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = REPO_ROOT / "results" / "init_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
