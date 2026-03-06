#!/usr/bin/env python3
"""Run the current published benchmark suite as one benchmark plan."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner import BenchmarkRunner


SAMPLE_SIZES = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
FEATURE_COUNTS = [2, 8, 32]
CLUSTER_COUNTS = [8, 32]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-id", default="current_20260609_kpp_single_start")
    parser.add_argument("--seed-base", type=int, default=20260609)
    parser.add_argument("--init", choices=["random", "k-means++"], default="k-means++")
    parser.add_argument("--sklearn-n-init", type=int, default=1)
    parser.add_argument("--rust-parallel-threads", type=int, default=0)
    args = parser.parse_args()

    runner = BenchmarkRunner(results_dir=args.results_dir, data_dir=args.data_dir)
    all_results: list[dict[str, Any]] = []

    total = len(SAMPLE_SIZES) * len(FEATURE_COUNTS) * len(CLUSTER_COUNTS) * 3
    experiment_num = 0

    for repeat_index in range(1, 4):
        seed = args.seed_base + repeat_index - 1
        for n_samples in SAMPLE_SIZES:
            for n_features in FEATURE_COUNTS:
                for n_clusters in CLUSTER_COUNTS:
                    experiment_num += 1
                    print(
                        f"\nExperiment {experiment_num}/{total}: "
                        f"n={n_samples}, f={n_features}, k_max={n_clusters}, "
                        f"repeat={repeat_index}, seed={seed}",
                        flush=True,
                    )
                    all_results.extend(
                        runner.run_experiment(
                            n_samples,
                            n_features,
                            n_clusters,
                            repeat_index=repeat_index,
                            dataset_seed=seed,
                            random_state=seed,
                            run_id=args.run_id,
                            init=args.init,
                            sklearn_n_init=args.sklearn_n_init,
                            rust_parallel_threads=args.rust_parallel_threads,
                        )
                    )

    results_df = runner.add_derived_metrics(pd.DataFrame(all_results))
    output = args.output or Path(args.results_dir) / (
        f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    print(f"\nResults saved to: {output}")
    runner.generate_summary(results_df)


if __name__ == "__main__":
    main()
