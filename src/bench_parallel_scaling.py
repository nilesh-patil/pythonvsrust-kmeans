#!/usr/bin/env python3
"""
Benchmark the Rust K-Means parallel scaling: runs the binary with
--parallel --threads N for a range of N and records wall-clock time.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
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


def parse_sample_grid(raw: str | None, fallback: int) -> list[int]:
    """Parse a comma-separated sample-size grid; preserve ascending order."""
    if not raw:
        return [fallback]
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"sample sizes must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("--n_sample_grid did not contain any sample sizes")
    return sorted(set(values))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def bench_one_sample_size(
    binary: Path,
    out_dir: Path,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    k_max: int,
    runs: int,
    thread_counts: list[int],
) -> list[dict[str, object]]:
    dataset = find_dataset(n_samples, n_features, n_clusters)
    print(f"\nDataset: {dataset.name}")

    tmp_csv = out_dir / "_parallel_scaling_tmp.csv"

    print("\nWarmup (serial)...")
    run_rust(binary, dataset, tmp_csv, k_max, parallel=False, threads=0)

    print(f"\nSerial baseline ({runs} runs)...")
    serial_times = [
        run_rust(binary, dataset, tmp_csv, k_max, parallel=False, threads=0)
        for _ in range(runs)
    ]
    serial_median = statistics.median(serial_times)
    print(f"  median: {serial_median:.3f}s  (times: {serial_times})")

    rows: list[dict[str, object]] = [{
        "threads":      0,            # 0 marks serial
        "parallel":     False,
        "median_s":     serial_median,
        "min_s":        min(serial_times),
        "max_s":        max(serial_times),
        "speedup":      1.0,
    }]

    for t in thread_counts:
        print(f"\nParallel, threads={t} ({runs} runs)...")
        times = [
            run_rust(binary, dataset, tmp_csv, k_max, parallel=True, threads=t)
            for _ in range(runs)
        ]
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

    if tmp_csv.exists():
        tmp_csv.unlink()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int, default=50_000)
    parser.add_argument(
        "--n_sample_grid",
        "--n-sample-grid",
        dest="n_sample_grid",
        default="",
        help="Comma-separated sample sizes for a scale sweep; overrides --n_samples.",
    )
    parser.add_argument("--n_features", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=16)
    parser.add_argument("--k_max",      type=int, default=8)
    parser.add_argument("--runs",       type=int, default=3)
    parser.add_argument(
        "--compat_n_samples",
        "--compat-n-samples",
        dest="compat_n_samples",
        type=int,
        default=32_000,
        help="Grid sample size copied to results/parallel_scaling.csv for compatibility.",
    )
    args = parser.parse_args()

    binary = REPO_ROOT / "src" / "rust_impl" / "target" / "release" / "rust_impl"
    if not binary.exists():
        sys.exit(f"Build first: cargo build --release (looked for {binary})")

    sample_sizes = parse_sample_grid(args.n_sample_grid, args.n_samples)
    grid_mode = bool(args.n_sample_grid)

    max_cores = os.cpu_count() or 1
    thread_counts = [t for t in (1, 2, 4, 8, max_cores) if t <= max_cores]
    thread_counts = sorted(set(thread_counts))
    print(f"Thread counts to sweep: {thread_counts}")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[int, Path] = {}
    for n_samples in sample_sizes:
        rows = bench_one_sample_size(
            binary=binary,
            out_dir=out_dir,
            n_samples=n_samples,
            n_features=args.n_features,
            n_clusters=args.n_clusters,
            k_max=args.k_max,
            runs=args.runs,
            thread_counts=thread_counts,
        )
        out_csv = (
            out_dir / f"parallel_scaling_n{n_samples}.csv"
            if grid_mode
            else out_dir / "parallel_scaling.csv"
        )
        write_rows(out_csv, rows)
        outputs[n_samples] = out_csv
        print(f"\nSaved {out_csv}")

    if grid_mode:
        compat_n = args.compat_n_samples if args.compat_n_samples in outputs else sample_sizes[-1]
        compat_csv = out_dir / "parallel_scaling.csv"
        shutil.copyfile(outputs[compat_n], compat_csv)
        print(f"Saved compatibility slice {compat_csv} from n={compat_n}")


if __name__ == "__main__":
    main()
