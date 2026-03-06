"""Tests for visualization source selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARETO_SCRIPT = PROJECT_ROOT / "src" / "visualize_quality_runtime_pareto.py"
SPEEDUP_SCRIPT = PROJECT_ROOT / "src" / "visualize_speedup_curve.py"
MEMORY_SCRIPT = PROJECT_ROOT / "src" / "visualize_memory_breakdown.py"


def _load_pareto_module():
    spec = importlib.util.spec_from_file_location("pareto_script_under_test", PARETO_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pareto_script_uses_latest_csv_with_external_quality_columns(tmp_path):
    module = _load_pareto_module()

    old_csv = tmp_path / "benchmark_results_20200101_000000.csv"
    new_csv = tmp_path / "benchmark_results_20260608_214731.csv"
    no_quality_csv = tmp_path / "benchmark_results_20270101_000000.csv"

    pd.DataFrame({"runtime": [1.0]}).to_csv(no_quality_csv, index=False)
    pd.DataFrame(
        {
            "runtime": [1.0],
            "adjusted_rand_index": [0.5],
            "normalized_mutual_info": [0.6],
        }
    ).to_csv(old_csv, index=False)
    pd.DataFrame(
        {
            "runtime": [1.0],
            "adjusted_rand_index": [0.7],
            "normalized_mutual_info": [0.8],
        }
    ).to_csv(new_csv, index=False)

    selected = module.latest_csv_with_columns(
        tmp_path,
        {"adjusted_rand_index", "normalized_mutual_info"},
    )

    assert selected == new_csv


def test_speedup_and_memory_scripts_use_latest_benchmark_csv(tmp_path):
    old_csv = tmp_path / "benchmark_results_20200101_000000.csv"
    new_csv = tmp_path / "benchmark_results_20260608_214731.csv"
    incomplete_csv = tmp_path / "benchmark_results_20270101_000000.csv"

    pd.DataFrame({"runtime": [1.0]}).to_csv(incomplete_csv, index=False)
    for path, runtime in ((old_csv, 2.0), (new_csv, 1.0)):
        pd.DataFrame(
            {
                "runtime": [runtime],
                "peak_memory_mb": [64.0],
                "implementation": ["python"],
                "n_samples": [1000],
            }
        ).to_csv(path, index=False)

    speedup = _load_module(SPEEDUP_SCRIPT, "speedup_script_under_test")
    memory = _load_module(MEMORY_SCRIPT, "memory_script_under_test")

    required = {"runtime", "implementation", "n_samples"}
    assert speedup.latest_csv_with_columns(tmp_path, required) == new_csv

    required = {"peak_memory_mb", "implementation", "n_samples"}
    assert memory.latest_csv_with_columns(tmp_path, required) == new_csv
