"""Tests for Feature 4: Ground-truth quality metrics (ARI/NMI) + Plotly dashboard."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root so we can import runner helpers
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Test 1 — generator writes a _labels.npy alongside the CSV
# ---------------------------------------------------------------------------

def test_generator_writes_labels_file(tmp_path):
    """generate_data.py must produce both the CSV and a sibling _labels.npy."""
    out_csv = tmp_path / "tiny.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "generate_data.py"),
            "--n_rows", "60",
            "--n_features", "3",
            "--n_clusters", "3",
            "--output", str(out_csv),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Generator failed:\n{result.stderr}"

    labels_file = tmp_path / "tiny_labels.npy"
    assert labels_file.exists(), "_labels.npy not written next to the CSV"

    labels = np.load(labels_file)
    assert labels.shape == (60,), f"Expected shape (60,), got {labels.shape}"
    assert np.issubdtype(labels.dtype, np.integer), f"Expected integer dtype, got {labels.dtype}"


# ---------------------------------------------------------------------------
# Test 2 — runner attaches ARI/NMI when labels file exists
# ---------------------------------------------------------------------------

def test_runner_attaches_ari_nmi_when_labels_exist(tmp_path):
    """calculate_clustering_metrics returns ARI in [-1,1] and NMI in [0,1]."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from runner import BenchmarkRunner

    rng = np.random.default_rng(0)
    n = 120
    n_clusters = 3

    # Build a tidy CSV (ID + features)
    true_labels = np.repeat(np.arange(n_clusters), n // n_clusters)
    # Each cluster has a distinct mean — KMeans should recover them
    features = true_labels[:, None] * 10.0 + rng.standard_normal((n, 2)) * 0.1
    df_data = pd.DataFrame(features, columns=["feature_1", "feature_2"])
    df_data.insert(0, "ID", range(1, n + 1))

    csv_path = tmp_path / "data.csv"
    df_data.to_csv(csv_path, index=False)

    # Save ground-truth labels next to the CSV
    labels_path = tmp_path / "data_labels.npy"
    np.save(labels_path, true_labels.astype(np.int32))

    # Build a results DataFrame with (near-perfect) cluster assignments
    df_results = pd.DataFrame({"cluster_3": true_labels})

    runner = BenchmarkRunner()
    metrics = runner.calculate_clustering_metrics(str(csv_path), df_results, n_clusters)

    ari = metrics.get("adjusted_rand_index")
    nmi = metrics.get("normalized_mutual_info")

    assert ari is not None, "adjusted_rand_index missing from metrics"
    assert nmi is not None, "normalized_mutual_info missing from metrics"
    assert not np.isnan(ari), "ARI is NaN even though labels file exists"
    assert not np.isnan(nmi), "NMI is NaN even though labels file exists"
    assert -1.0 <= ari <= 1.0, f"ARI={ari} out of range [-1, 1]"
    assert 0.0 <= nmi <= 1.0, f"NMI={nmi} out of range [0, 1]"


# ---------------------------------------------------------------------------
# Test 3 — runner gracefully handles missing labels file
# ---------------------------------------------------------------------------

def test_runner_handles_missing_labels(tmp_path):
    """When _labels.npy is absent, both metrics must be np.nan (no exception)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from runner import BenchmarkRunner

    rng = np.random.default_rng(1)
    n, n_clusters = 60, 2

    features = rng.standard_normal((n, 2))
    df_data = pd.DataFrame(features, columns=["feature_1", "feature_2"])
    df_data.insert(0, "ID", range(1, n + 1))

    csv_path = tmp_path / "no_labels.csv"
    df_data.to_csv(csv_path, index=False)
    # Deliberately do NOT create no_labels_labels.npy

    labels = np.zeros(n, dtype=int)
    labels[n // 2 :] = 1
    df_results = pd.DataFrame({"cluster_2": labels})

    runner = BenchmarkRunner()
    metrics = runner.calculate_clustering_metrics(str(csv_path), df_results, n_clusters)

    ari = metrics.get("adjusted_rand_index")
    nmi = metrics.get("normalized_mutual_info")

    assert ari is not None, "adjusted_rand_index key missing"
    assert nmi is not None, "normalized_mutual_info key missing"
    assert np.isnan(ari), f"Expected NaN for ARI when labels absent, got {ari}"
    assert np.isnan(nmi), f"Expected NaN for NMI when labels absent, got {nmi}"


def test_runner_samples_internal_quality_metrics_for_large_n(tmp_path, monkeypatch):
    """Large-N runs should sample expensive internal metrics and keep full ARI/NMI."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import runner as runner_module
    from runner import BenchmarkRunner

    monkeypatch.setattr(runner_module, "QUALITY_EXACT_SAMPLE_LIMIT", 20)
    monkeypatch.setattr(runner_module, "QUALITY_SAMPLE_SIZE", 30)

    n = 90
    n_clusters = 3
    true_labels = np.tile(np.arange(n_clusters), n // n_clusters)
    features = true_labels[:, None] * 8.0 + np.arange(n)[:, None] * 0.001
    df_data = pd.DataFrame(features, columns=["feature_1"])
    df_data.insert(0, "ID", range(1, n + 1))

    csv_path = tmp_path / "large.csv"
    df_data.to_csv(csv_path, index=False)
    np.save(tmp_path / "large_labels.npy", true_labels.astype(np.int32))

    df_results = pd.DataFrame({"cluster_3": true_labels})
    metrics = BenchmarkRunner().calculate_clustering_metrics(
        str(csv_path),
        df_results,
        n_clusters,
    )

    assert metrics["quality_sampled"] is True
    assert metrics["quality_sample_size"] == 30
    assert metrics["silhouette_score"] > 0
    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert metrics["normalized_mutual_info"] == pytest.approx(1.0)


def test_runner_adds_resource_and_paired_speedup_metrics():
    """The saved benchmark CSV must include normalized resource and paired speedup columns."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from runner import BenchmarkRunner

    rows = []
    for impl, wall_time, cpu_time, rss in [
        ("python", 2.0, 1.8, 80.0),
        ("rust", 0.5, 0.4, 4.0),
        ("rust_parallel", 0.4, 1.2, 6.0),
    ]:
        rows.append(
            {
                "run_id": "fixture",
                "repeat_index": 1,
                "dataset_seed": 7,
                "random_state": 7,
                "implementation": impl,
                "n_samples": 1000,
                "n_features": 8,
                "n_clusters": 4,
                "k_max": 4,
                "init": "k-means++",
                "cluster_std": 0.5,
                "cluster_separation": 3.0,
                "wall_time_s": wall_time,
                "runtime": wall_time,
                "cpu_time_s": cpu_time,
                "peak_rss_mb": rss,
                "peak_memory_mb": rss,
                "requested_threads": 4 if impl == "rust_parallel" else 1,
            }
        )

    enriched = BenchmarkRunner().add_derived_metrics(pd.DataFrame(rows))

    required = {
        "nominal_work_units",
        "samples_per_second",
        "work_units_per_second",
        "rss_mb_per_1k_samples",
        "cpu_seconds_per_1k_samples",
        "speedup_vs_python",
        "speedup_vs_rust_serial",
    }
    assert required.issubset(enriched.columns)

    rust = enriched[enriched["implementation"] == "rust"].iloc[0]
    rust_parallel = enriched[enriched["implementation"] == "rust_parallel"].iloc[0]

    assert rust["nominal_work_units"] == 1000 * 8 * (1 + 2 + 3 + 4)
    assert rust["speedup_vs_python"] == pytest.approx(4.0)
    assert rust_parallel["speedup_vs_rust_serial"] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Test 4 — dashboard HTML contains all four required tab labels
# ---------------------------------------------------------------------------

def test_dashboard_builds_html_with_all_tabs(tmp_path):
    """build_dashboard.py produces HTML containing the redesigned tab labels."""
    # Minimal fixture CSV that matches the real schema
    rows = []
    for impl in ("python", "rust"):
        for n in (500, 1000):
            rows.append({
                "runtime": 1.2,
                "peak_memory_mb": 64.0,
                "silhouette_score": 0.6,
                "davies_bouldin_index": 0.4,
                "calinski_harabasz_index": 300.0,
                "inertia": 1234.5,
                "implementation": impl,
                "n_samples": n,
                "n_features": 2,
                "n_clusters": 3,
                "exit_code": 0,
                # No ARI/NMI columns — external tab should show placeholder
            })
    fixture_csv = tmp_path / "benchmark_results_fixture.csv"
    pd.DataFrame(rows).to_csv(fixture_csv, index=False)

    out_dir = tmp_path / "dashboards"
    out_dir.mkdir()
    out_html = out_dir / "index.html"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "build_dashboard.py"),
            "--input", str(fixture_csv),
            "--output", str(out_html),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"build_dashboard.py failed:\n{result.stderr}"
    assert out_html.exists(), "output HTML file not created"

    html = out_html.read_text(encoding="utf-8")
    required_tabs = [
        "Runtime vs workload",
        "Throughput",
        "Memory footprint",
        "Resource table",
        "Quality frontier",
    ]
    for tab in required_tabs:
        assert tab in html, f"Tab label '{tab}' not found in dashboard HTML"

    assert "CLI k-sweep runtime" in html
    assert "sampled RSS, CPU/resource use" in html
    assert "including CSV read/write" in html
