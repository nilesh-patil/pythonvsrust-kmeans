"""Tests for k-means++ centroid initialization (Python impl).

Red phase: these tests are written before the implementation exists.
All four cases target KMeansClustering.init and _initialize_centroids_kpp.
"""

import sys
import pathlib

import numpy as np
import pytest
from sklearn.datasets import make_blobs  # available in the pixi env

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "python_impl"))
from kmeans import KMeansClustering


# ---------------------------------------------------------------------------
# Test 1: shape and membership
# ---------------------------------------------------------------------------

def test_kpp_returns_k_distinct_points() -> None:
    """_initialize_centroids_kpp must return k rows, each a row that exists in X."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)

    km = KMeansClustering(n_clusters=5, random_state=7, init="k-means++")
    centroids = km._initialize_centroids_kpp(X)

    assert centroids.shape == (5, 4), f"Expected (5, 4), got {centroids.shape}"

    # Every centroid must be an exact row of X (selection, not synthesis)
    for centroid in centroids:
        assert any(np.allclose(centroid, x) for x in X), (
            "Centroid is not a row from X"
        )

    # All centroids must be distinct
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            assert not np.allclose(centroids[i], centroids[j]), (
                f"Centroids {i} and {j} are identical"
            )


# ---------------------------------------------------------------------------
# Test 2: reproducibility
# ---------------------------------------------------------------------------

def test_kpp_seed_reproducibility() -> None:
    """Same random_state must yield bit-identical centroids on repeated calls."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 3)

    km_a = KMeansClustering(n_clusters=6, random_state=42, init="k-means++")
    km_b = KMeansClustering(n_clusters=6, random_state=42, init="k-means++")

    c_a = km_a._initialize_centroids_kpp(X)
    c_b = km_b._initialize_centroids_kpp(X)

    np.testing.assert_array_equal(c_a, c_b, err_msg="k-means++ not reproducible with same seed")

    # Different seed → different result (probabilistically; 3-sigma safe with these params)
    km_c = KMeansClustering(n_clusters=6, random_state=99, init="k-means++")
    c_c = km_c._initialize_centroids_kpp(X)
    assert not np.array_equal(c_a, c_c), "Different seeds produced identical centroids"


# ---------------------------------------------------------------------------
# Test 3: inertia comparison on well-separated blobs
# ---------------------------------------------------------------------------

def _inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    total = 0.0
    for k in range(centroids.shape[0]):
        pts = X[labels == k]
        if len(pts):
            total += float(np.sum((pts - centroids[k]) ** 2))
    return total


def test_kpp_beats_random_inertia_on_separated_blobs() -> None:
    """Over 20 independent runs, mean k-means++ inertia <= mean random inertia."""
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.5, random_state=0)

    n_runs = 20
    inertias_kpp = []
    inertias_random = []

    for seed in range(n_runs):
        km_kpp = KMeansClustering(n_clusters=4, random_state=seed, init="k-means++")
        km_kpp.fit(X)
        inertias_kpp.append(_inertia(X, km_kpp.labels, km_kpp.centroids))

        km_rand = KMeansClustering(n_clusters=4, random_state=seed, init="random")
        km_rand.fit(X)
        inertias_random.append(_inertia(X, km_rand.labels, km_rand.centroids))

    mean_kpp = float(np.mean(inertias_kpp))
    mean_rand = float(np.mean(inertias_random))

    print(f"\n  mean inertia k-means++: {mean_kpp:.4f}")
    print(f"  mean inertia random:    {mean_rand:.4f}")

    assert mean_kpp <= mean_rand, (
        f"Expected k-means++ ({mean_kpp:.4f}) <= random ({mean_rand:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: k == n_samples edge case
# ---------------------------------------------------------------------------

def test_kpp_handles_k_equals_n() -> None:
    """When k == n_samples, each data point should be selected exactly once."""
    rng = np.random.RandomState(5)
    X = rng.randn(10, 2)

    km = KMeansClustering(n_clusters=10, random_state=3, init="k-means++")
    # fit() shortcuts when n_clusters >= n_samples, so call _initialize_centroids_kpp directly
    centroids = km._initialize_centroids_kpp(X)

    assert centroids.shape == (10, 2)

    # Sort both and compare — every point appears exactly once
    X_sorted = X[np.lexsort(X.T)]
    c_sorted = centroids[np.lexsort(centroids.T)]
    np.testing.assert_array_almost_equal(
        X_sorted, c_sorted,
        err_msg="k==n_samples: not every point selected exactly once",
    )
