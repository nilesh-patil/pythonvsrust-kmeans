#!/usr/bin/env python3
"""
Benchmark the Rust K-Means parallel scaling: runs the binary with
--parallel --threads N for a range of N and records wall-clock time.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_dataset(n: int, d: int, c: int) -> Path:
    """Find or generate a dataset (n=samples, d=features, c=clusters)."""
    candidates = list((REPO_ROOT / "data").glob(f"dataset_n{n}_f{d}_c{c}_*.csv"))
    explicit = REPO_ROOT / "data" / f"input_n{n}_f{d}_c{c}.csv"
    if candidates:
        return candidates[0]
    if explicit.exists():
        return explicit
    print(f"  generating dataset n={n} d={d} c={c}...", flush=True)
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src" / "generate_data.py"),
            "--n_rows",     str(n),
            "--n_features", str(d),
            "--n_clusters", str(c),
            "--output",     str(explicit),
        ],
        check=True,
    )
    return explicit


def run_rust(binary: Path, input_csv: Path, output_csv: Path,
             k_max: int, parallel: bool, threads: int) -> float:
    cmd = [
        str(binary),
        "--input",          str(input_csv),
        "--output",         str(output_csv),
        "--k_clusters_max", str(k_max),
        "--init",           "k-means++",
    ]
    if parallel:
        cmd += ["--parallel"]
        if threads > 0:
            cmd += ["--threads", str(threads)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, check=False)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode(errors="replace"))
        raise RuntimeError(f"rust binary failed: thread count={threads}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int, default=50_000)
    parser.add_argument("--n_features", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=16)
    parser.add_argument("--k_max",      type=int, default=8)
    parser.add_argument("--runs",       type=int, default=3)
    args = parser.parse_args()

    binary = REPO_ROOT / "src" / "rust_impl" / "target" / "release" / "rust_impl"
    if not binary.exists():
        sys.exit(f"Build first: cargo build --release (looked for {binary})")

    dataset = find_dataset(args.n_samples, args.n_features, args.n_clusters)
    print(f"Dataset: {dataset.name}")

    max_cores = os.cpu_count() or 1
    thread_counts = [t for t in (1, 2, 4, 8, max_cores) if t <= max_cores]
    thread_counts = sorted(set(thread_counts))
    print(f"Thread counts to sweep: {thread_counts}")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / "_parallel_scaling_tmp.csv"

    print("\nWarmup (serial)...")
    run_rust(binary, dataset, tmp_csv, args.k_max, parallel=False, threads=0)

    print(f"\nSerial baseline ({args.runs} runs)...")
    serial_times = [run_rust(binary, dataset, tmp_csv, args.k_max,
                             parallel=False, threads=0)
                    for _ in range(args.runs)]
    serial_median = statistics.median(serial_times)
    print(f"  median: {serial_median:.3f}s  (times: {serial_times})")

    rows = [{
        "threads":      0,            # 0 marks serial
        "parallel":     False,
        "median_s":     serial_median,
        "min_s":        min(serial_times),
        "max_s":        max(serial_times),
        "speedup":      1.0,
    }]

    for t in thread_counts:
        print(f"\nParallel, threads={t} ({args.runs} runs)...")
        times = [run_rust(binary, dataset, tmp_csv, args.k_max,
                          parallel=True, threads=t)
                 for _ in range(args.runs)]
        median = statistics.median(times)
        print(f"  median: {median:.3f}s  (times: {times})")
        rows.append({
            "threads":  t,
            "parallel": True,
            "median_s": median,
            "min_s":    min(times),
            "max_s":    max(times),
            "speedup":  serial_median / median,
        })

    out_csv = out_dir / "parallel_scaling.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out_csv}")

    if tmp_csv.exists():
        tmp_csv.unlink()


if __name__ == "__main__":
    main()
