"""Tests for src/animate_convergence.py — Lloyd's iteration capture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "python_impl"))

from animate_convergence import LloydsTrace, run_lloyds  # noqa: E402
from kmeans import KMeansClustering  # noqa: E402


def _blobs_2d(seed: int = 42, k: int = 4, n: int = 200):
    X, _ = make_blobs(n_samples=n, n_features=2, centers=k,
                      cluster_std=0.8, random_state=seed)
    return X


def test_capture_history_matches_serial_kmeans():
    """The traced run must produce the same final state as production KMeansClustering."""
    X = _blobs_2d(seed=7, k=4)
    trace = run_lloyds(X, k=4, init="random", seed=7, max_iter=300)

    reference = KMeansClustering(n_clusters=4, init="random", random_state=7,
                                 max_iter=300).fit(X)

    np.testing.assert_array_equal(trace.labels_history[-1], reference.labels)
    np.testing.assert_allclose(trace.centroid_history[-1], reference.centroids,
                               atol=1e-10)


def test_centroid_history_length():
    """history length should equal iterations_run + 1 (initial state + after each iter)."""
    X = _blobs_2d(seed=11, k=4)
    trace = run_lloyds(X, k=4, init="k-means++", seed=11, max_iter=20)

    assert len(trace.centroid_history) == trace.iterations_run + 1
    assert len(trace.labels_history)   == trace.iterations_run + 1


def test_2d_only():
    """Must reject non-2D inputs."""
    X = np.random.RandomState(0).randn(50, 3)
    with pytest.raises(ValueError, match="2-D"):
        run_lloyds(X, k=3, init="random", seed=0)


def test_trace_dataclass_has_inertia_per_step():
    """LloydsTrace must include an inertia value per recorded state, for animation titles."""
    X = _blobs_2d(seed=3, k=4, n=100)
    trace = run_lloyds(X, k=4, init="random", seed=3, max_iter=50)
    assert isinstance(trace, LloydsTrace)
    assert len(trace.inertia_history) == len(trace.centroid_history)
    # inertia should be monotonically non-increasing under Lloyd's
    diffs = np.diff(trace.inertia_history)
    assert (diffs <= 1e-9).all(), f"inertia increased somewhere: {diffs}"
