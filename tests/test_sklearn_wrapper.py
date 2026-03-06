"""Regression tests for the scikit-learn CLI wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKLEARN_WRAPPER = PROJECT_ROOT / "src" / "sklearn_impl" / "kmeans.py"


def _load_sklearn_wrapper():
    spec = importlib.util.spec_from_file_location("sklearn_wrapper_under_test", SKLEARN_WRAPPER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sklearn_wrapper_fits_once_per_k(monkeypatch):
    """fit_predict already fits; the wrapper must not call fit separately."""
    module = _load_sklearn_wrapper()
    calls: list[tuple[str, int]] = []

    class FakeKMeans:
        def __init__(self, n_clusters: int, random_state: int, n_init: int):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.n_init = n_init

        def fit(self, X):
            calls.append(("fit", self.n_clusters))
            return self

        def fit_predict(self, X):
            calls.append(("fit_predict", self.n_clusters))
            return np.full(X.shape[0], self.n_clusters)

    monkeypatch.setattr(module, "KMeans", FakeKMeans)

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    results = module.run_kmeans_multiple_k(X, k_max=3, random_state=42)

    assert calls == [("fit_predict", 1), ("fit_predict", 2), ("fit_predict", 3)]
    assert set(results) == {1, 2, 3}
