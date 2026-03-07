#!/usr/bin/env python3
"""
Benchmark Runner for K-Means Clustering Implementations
Orchestrates data generation, runs different implementations, and records metrics
"""

import argparse
import os
import sys
import time
import subprocess
import json
import hashlib
import resource
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import psutil
from pathlib import Path

from src.viz_style import (
    IMPL_HATCHES_MPL,
    INK_FAINT,
    apply_mpl_style,
    color,
    display_name,
    mpl_linestyle,
    mpl_marker,
    ordered_implementations,
    si_log_axis,
    style_axes,
)

pd.options.mode.copy_on_write = True

QUALITY_EXACT_SAMPLE_LIMIT = 32_000
QUALITY_SAMPLE_SIZE = 10_000


class BenchmarkRunner:
    """Main benchmark orchestrator for K-Means experiments"""
    
    def __init__(self, results_dir: str = "results", data_dir: str = "data"):
        """
        Initialize benchmark runner
        
        Args:
            results_dir: Directory to save results
            data_dir: Directory to save/load datasets
        """
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
    def get_dataset_filename(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        random_state: int = 42,
        cluster_std: float = 0.5,
        cluster_separation: float = 3.0,
    ) -> str:
        """
        Generate a unique filename for a dataset based on its parameters
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_clusters: Number of clusters
            
        Returns:
            Unique filename for the dataset
        """
        # Create a hash of the parameters for unique identification
        params_str = (
            f"n{n_samples}_f{n_features}_c{n_clusters}"
            f"_seed{random_state}_std{cluster_std:g}_sep{cluster_separation:g}"
        )
        hash_str = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"dataset_{params_str}_{hash_str}.csv"
    
    def generate_dataset(self, n_samples: int, n_features: int, n_clusters: int, 
                        random_state: int = 42, cluster_std: float = 0.5,
                        cluster_separation: float = 3.0) -> str:
        """
        Generate synthetic dataset using generate_data.py
        
        Args:
            n_samples: Number of samples
            n_features: Number of features  
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            Path to generated dataset
        """
        filename = self.get_dataset_filename(
            n_samples,
            n_features,
            n_clusters,
            random_state=random_state,
            cluster_std=cluster_std,
            cluster_separation=cluster_separation,
        )
        filepath = self.data_dir / filename
        
        # Check if dataset already exists
        if filepath.exists():
            print(f"Dataset already exists: {filepath}")
            return str(filepath)
        
        print(f"Generating dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")
        
        # Run generate_data.py
        cmd = [
            sys.executable, "src/generate_data.py",
            "--n_rows", str(n_samples),
            "--n_features", str(n_features),
            "--n_clusters", str(n_clusters),
            "--output", str(filepath),
            "--random_state", str(random_state),
            "--cluster_std", str(cluster_std),
            "--cluster_separation", str(cluster_separation),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Dataset generated: {filepath}")
            return str(filepath)
        except subprocess.CalledProcessError as e:
            print(f"Error generating dataset: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise
    
    def measure_process_metrics(self, cmd: List[str]) -> Dict:
        """
        Run a command and measure its runtime and memory usage
        
        Args:
            cmd: Command to run as list of strings
            
        Returns:
            Dictionary with metrics (runtime, peak_memory, exit_code)
        """
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start_time = time.perf_counter()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor memory usage and CPU
        peak_rss = 0
        baseline_rss = None
        peak_uss = 0
        cpu_percentages = []
        thread_counts = []
        io_read_bytes = 0
        io_write_bytes = 0
        voluntary_ctx_switches = 0
        involuntary_ctx_switches = 0
        rss_samples = 0
        try:
            ps_process = psutil.Process(process.pid)
            ps_process.cpu_percent(interval=None)
            
            # Poll memory and CPU usage while process is running
            while process.poll() is None:
                try:
                    memory_info = ps_process.memory_info()
                    if baseline_rss is None:
                        baseline_rss = memory_info.rss
                    peak_rss = max(peak_rss, memory_info.rss)
                    rss_samples += 1

                    try:
                        full_memory_info = ps_process.memory_full_info()
                        peak_uss = max(peak_uss, getattr(full_memory_info, "uss", 0))
                    except (psutil.AccessDenied, AttributeError):
                        pass
                    
                    cpu_percent = ps_process.cpu_percent(interval=None)
                    if cpu_percent > 0:  # Ignore initial 0 values
                        cpu_percentages.append(cpu_percent)

                    try:
                        thread_counts.append(ps_process.num_threads())
                    except psutil.Error:
                        pass

                    try:
                        io_counters = ps_process.io_counters()
                        io_read_bytes = getattr(io_counters, "read_bytes", io_read_bytes)
                        io_write_bytes = getattr(io_counters, "write_bytes", io_write_bytes)
                    except (psutil.AccessDenied, AttributeError):
                        pass

                    try:
                        ctx = ps_process.num_ctx_switches()
                        voluntary_ctx_switches = getattr(ctx, "voluntary", voluntary_ctx_switches)
                        involuntary_ctx_switches = getattr(ctx, "involuntary", involuntary_ctx_switches)
                    except psutil.Error:
                        pass
                    
                    time.sleep(0.01)  # Poll every 10ms
                except psutil.NoSuchProcess:
                    break
                    
        except Exception as e:
            print(f"Warning: Could not monitor memory: {e}")
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        end_time = time.perf_counter()
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

        wall_time_s = end_time - start_time
        user_cpu_s = max(0.0, usage_after.ru_utime - usage_before.ru_utime)
        system_cpu_s = max(0.0, usage_after.ru_stime - usage_before.ru_stime)
        cpu_time_s = user_cpu_s + system_cpu_s
        effective_cores = cpu_time_s / wall_time_s if wall_time_s > 0 else 0.0

        # Calculate CPU statistics
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        baseline_rss = baseline_rss or 0
        peak_memory_mb = peak_rss / (1024 * 1024)
        
        return {
            "runtime": wall_time_s,
            "wall_time_s": wall_time_s,
            "peak_memory_bytes": peak_rss,
            "peak_memory_mb": peak_memory_mb,
            "peak_rss_bytes": peak_rss,
            "peak_rss_mb": peak_memory_mb,
            "baseline_rss_bytes": baseline_rss,
            "baseline_rss_mb": baseline_rss / (1024 * 1024),
            "rss_delta_bytes": max(0, peak_rss - baseline_rss),
            "rss_delta_mb": max(0, peak_rss - baseline_rss) / (1024 * 1024),
            "peak_uss_bytes": peak_uss,
            "peak_uss_mb": peak_uss / (1024 * 1024),
            "rss_sample_count": rss_samples,
            "user_cpu_s": user_cpu_s,
            "system_cpu_s": system_cpu_s,
            "cpu_time_s": cpu_time_s,
            "effective_cores": effective_cores,
            "cpu_utilization_ratio": effective_cores,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "max_threads": max(thread_counts) if thread_counts else 0,
            "io_read_bytes": io_read_bytes,
            "io_write_bytes": io_write_bytes,
            "voluntary_ctx_switches": voluntary_ctx_switches,
            "involuntary_ctx_switches": involuntary_ctx_switches,
            "exit_code": process.returncode,
            "stdout": stdout.decode('utf-8'),
            "stderr": stderr.decode('utf-8')
        }
    
    def run_python_impl(self, dataset_path: str, n_clusters: int,
                        random_state: int = 42, init: str = "random") -> Dict:
        """
        Run pure Python K-Means implementation
        
        Args:
            dataset_path: Path to input dataset
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with metrics and results
        """
        print(f"Running Python implementation (k={n_clusters})...")
        
        output_path = self.results_dir / f"python_output_{os.path.basename(dataset_path)}"
        
        cmd = [
            sys.executable, "src/python_impl/kmeans.py",
            "--input", dataset_path,
            "--output", str(output_path),
            "--k_clusters_max", str(n_clusters),
            "--random_state", str(random_state),
            "--init", init,
        ]
        
        metrics = self.measure_process_metrics(cmd)
        
        # Calculate clustering metrics from results
        if metrics["exit_code"] == 0 and output_path.exists():
            try:
                df = pd.read_csv(output_path)
                # Calculate all clustering metrics
                clustering_metrics = self.calculate_clustering_metrics(dataset_path, df, n_clusters)
                metrics.update(clustering_metrics)
                # Clean up output file
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not calculate clustering metrics: {e}")
        
        metrics["implementation"] = "python"
        return metrics
    
    def run_sklearn_impl(self, dataset_path: str, n_clusters: int,
                         random_state: int = 42, sklearn_n_init: int = 10) -> Dict:
        """
        Run scikit-learn K-Means implementation
        
        Args:
            dataset_path: Path to input dataset
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with metrics and results
        """
        print(f"Running Scikit Learn implementation (k={n_clusters})...")
        
        output_path = self.results_dir / f"sklearn_output_{os.path.basename(dataset_path)}"
        
        cmd = [
            sys.executable, "src/sklearn_impl/kmeans.py",
            "--input", dataset_path,
            "--output", str(output_path),
            "--k_clusters_max", str(n_clusters),
            "--random_state", str(random_state),
            "--n_init", str(sklearn_n_init),
        ]
        
        metrics = self.measure_process_metrics(cmd)
        
        # Calculate clustering metrics from results
        if metrics["exit_code"] == 0 and output_path.exists():
            try:
                df = pd.read_csv(output_path)
                # Calculate all clustering metrics
                clustering_metrics = self.calculate_clustering_metrics(dataset_path, df, n_clusters)
                metrics.update(clustering_metrics)
                # Clean up output file
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not calculate clustering metrics: {e}")
        
        metrics["implementation"] = "sklearn"
        return metrics
    
    def run_rust_impl(self, dataset_path: str, n_clusters: int,
                      random_state: int = 42, init: str = "random") -> Dict:
        """
        Run Rust K-Means implementation
        
        Args:
            dataset_path: Path to input dataset
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with metrics and results
        """
        print(f"Running Rust implementation (k={n_clusters})...")
        
        output_path = self.results_dir / f"rust_output_{os.path.basename(dataset_path)}"
        
        # Path to Rust binary
        rust_binary = Path("src/rust_impl/target/release/rust_impl")
        if not rust_binary.exists():
            # Fall back to debug binary if release not built
            rust_binary = Path("src/rust_impl/target/debug/rust_impl")
            if not rust_binary.exists():
                print("Warning: Rust binary not found. Skipping Rust implementation.")
                return {
                    "runtime": None,
                    "peak_memory_bytes": None,
                    "peak_memory_mb": None,
                    "exit_code": -1,
                    "inertia": None,
                    "implementation": "rust",
                    "error": "Binary not found"
                }
        
        cmd = [
            str(rust_binary),
            "--input", dataset_path,
            "--output", str(output_path),
            "--k_clusters_max", str(n_clusters),
            "--random-state", str(random_state),
            "--init", init,
        ]
        
        metrics = self.measure_process_metrics(cmd)
        
        # Calculate clustering metrics from results
        if metrics["exit_code"] == 0 and output_path.exists():
            try:
                df = pd.read_csv(output_path)
                # Calculate all clustering metrics
                clustering_metrics = self.calculate_clustering_metrics(dataset_path, df, n_clusters)
                metrics.update(clustering_metrics)
                # Clean up output file
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not calculate clustering metrics: {e}")
        
        metrics["implementation"] = "rust"
        return metrics

    def run_rust_parallel_impl(self, dataset_path: str, n_clusters: int,
                               random_state: int = 42, init: str = "random",
                               threads: int = 0) -> Dict:
        """
        Run Rust K-Means implementation with Rayon parallelism (--parallel --threads 0).

        Args:
            dataset_path: Path to input dataset
            n_clusters: Number of clusters

        Returns:
            Dictionary with metrics and results; implementation == "rust_parallel"
        """
        print(f"Running Rust-Parallel implementation (k={n_clusters})...")

        output_path = self.results_dir / f"rust_parallel_output_{os.path.basename(dataset_path)}"

        # Path to Rust binary (same binary, different flags)
        rust_binary = Path("src/rust_impl/target/release/rust_impl")
        if not rust_binary.exists():
            rust_binary = Path("src/rust_impl/target/debug/rust_impl")
            if not rust_binary.exists():
                print("Warning: Rust binary not found. Skipping Rust-Parallel implementation.")
                return {
                    "runtime": None,
                    "peak_memory_bytes": None,
                    "peak_memory_mb": None,
                    "exit_code": -1,
                    "inertia": None,
                    "implementation": "rust_parallel",
                    "error": "Binary not found",
                }

        cmd = [
            str(rust_binary),
            "--input", dataset_path,
            "--output", str(output_path),
            "--k_clusters_max", str(n_clusters),
            "--random-state", str(random_state),
            "--init", init,
            "--parallel",
            "--threads", str(threads),  # 0 = all available cores (Rayon default)
        ]

        metrics = self.measure_process_metrics(cmd)

        if metrics["exit_code"] == 0 and output_path.exists():
            try:
                df = pd.read_csv(output_path)
                clustering_metrics = self.calculate_clustering_metrics(dataset_path, df, n_clusters)
                metrics.update(clustering_metrics)
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not calculate clustering metrics: {e}")

        metrics["implementation"] = "rust_parallel"
        return metrics

    def calculate_inertia(self, dataset_path: str, results_df: pd.DataFrame, n_clusters: int) -> float:
        """
        Calculate inertia (within-cluster sum of squares)
        
        Args:
            dataset_path: Path to original dataset
            results_df: DataFrame with cluster assignments
            n_clusters: Number of clusters
            
        Returns:
            Inertia value
        """
        # Load original data
        data_df = pd.read_csv(dataset_path)
        
        # Get feature columns
        feature_cols = [col for col in data_df.columns if col != 'ID']
        X = data_df[feature_cols].values
        
        # Get cluster assignments
        cluster_col = f'cluster_{n_clusters}' if f'cluster_{n_clusters}' in results_df.columns else 'cluster'
        if cluster_col not in results_df.columns:
            return None
            
        labels = results_df[cluster_col].values
        
        # Calculate centroids
        centroids = np.zeros((n_clusters, X.shape[1]))
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
        
        # Calculate inertia
        inertia = 0
        for i in range(len(X)):
            cluster = labels[i]
            distance = np.linalg.norm(X[i] - centroids[cluster])
            inertia += distance ** 2
            
        return inertia
    
    def calculate_clustering_metrics(self, dataset_path: str, results_df: pd.DataFrame,
                                   n_clusters: int) -> Dict:
        """
        Calculate comprehensive clustering quality metrics

        Args:
            dataset_path: Path to original dataset
            results_df: DataFrame with cluster assignments
            n_clusters: Number of clusters

        Returns:
            Dictionary with various clustering metrics
        """
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        # Load original data
        data_df = pd.read_csv(dataset_path)

        # Get feature columns
        feature_cols = [col for col in data_df.columns if col != 'ID']
        X = data_df[feature_cols].values

        # Get cluster assignments
        cluster_col = f'cluster_{n_clusters}' if f'cluster_{n_clusters}' in results_df.columns else 'cluster'
        if cluster_col not in results_df.columns:
            return {}

        labels = results_df[cluster_col].values

        metrics = {}
        metrics['quality_sampled'] = False
        metrics['quality_sample_size'] = len(X)

        quality_X = X
        quality_labels = labels
        if len(X) > QUALITY_EXACT_SAMPLE_LIMIT:
            sample_size = min(QUALITY_SAMPLE_SIZE, len(X))
            rng = np.random.default_rng(0)
            sample_idx = np.sort(rng.choice(len(X), size=sample_size, replace=False))
            quality_X = X[sample_idx]
            quality_labels = labels[sample_idx]
            metrics['quality_sampled'] = True
            metrics['quality_sample_size'] = sample_size

        try:
            # Calculate inertia
            metrics['inertia'] = self.calculate_inertia(dataset_path, results_df, n_clusters)

            # Only calculate other metrics if we have more than one cluster
            unique_labels = np.unique(quality_labels)
            if len(unique_labels) > 1:
                # Silhouette Score (-1 to 1, higher is better)
                metrics['silhouette_score'] = silhouette_score(quality_X, quality_labels)

                # Davies-Bouldin Index (lower is better)
                metrics['davies_bouldin_index'] = davies_bouldin_score(quality_X, quality_labels)

                # Calinski-Harabasz Index (higher is better)
                metrics['calinski_harabasz_index'] = calinski_harabasz_score(
                    quality_X,
                    quality_labels,
                )

        except Exception as e:
            print(f"Warning: Error calculating clustering metrics: {e}")

        # External metrics — require sibling _labels.npy written by the generator.
        # Fall back to NaN without crashing if the file is absent or malformed.
        labels_path = Path(dataset_path).with_name(Path(dataset_path).stem + "_labels.npy")
        try:
            if labels_path.exists():
                true_labels = np.load(labels_path)
                metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, labels)
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                    true_labels, labels, average_method="arithmetic"
                )
            else:
                metrics['adjusted_rand_index'] = np.nan
                metrics['normalized_mutual_info'] = np.nan
        except Exception as e:
            print(f"Warning: Error computing external quality metrics: {e}")
            metrics['adjusted_rand_index'] = np.nan
            metrics['normalized_mutual_info'] = np.nan

        return metrics

    def add_derived_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized resource, throughput, and paired speedup columns."""
        df = results_df.copy()

        if "wall_time_s" not in df.columns and "runtime" in df.columns:
            df["wall_time_s"] = df["runtime"]
        if "runtime" not in df.columns and "wall_time_s" in df.columns:
            df["runtime"] = df["wall_time_s"]
        if "peak_rss_mb" not in df.columns and "peak_memory_mb" in df.columns:
            df["peak_rss_mb"] = df["peak_memory_mb"]
        if "peak_memory_mb" not in df.columns and "peak_rss_mb" in df.columns:
            df["peak_memory_mb"] = df["peak_rss_mb"]

        k_max = df.get("k_max", df.get("n_clusters", pd.Series(1, index=df.index))).clip(lower=1)
        wall_time = df["wall_time_s"].replace(0, np.nan)
        cpu_time = df.get("cpu_time_s", pd.Series(np.nan, index=df.index)).replace(0, np.nan)

        df["input_values"] = df["n_samples"] * df["n_features"]
        df["k_sweep_fits"] = k_max
        df["k_sweep_sum_k"] = k_max * (k_max + 1) / 2
        df["nominal_work_units"] = df["n_samples"] * df["n_features"] * df["k_sweep_sum_k"]
        df["samples_per_second"] = df["n_samples"] / wall_time
        df["sample_features_per_second"] = df["input_values"] / wall_time
        df["work_units_per_second"] = df["nominal_work_units"] / wall_time
        df["wall_seconds_per_1k_samples"] = wall_time * 1000 / df["n_samples"]
        df["cpu_seconds_per_1k_samples"] = cpu_time * 1000 / df["n_samples"]
        df["rss_mb_per_1k_samples"] = df["peak_rss_mb"] * 1000 / df["n_samples"]
        df["rss_bytes_per_sample"] = df["peak_rss_mb"] * 1024 * 1024 / df["n_samples"]
        df["rss_bytes_per_sample_feature"] = (
            df["peak_rss_mb"] * 1024 * 1024 / df["input_values"].replace(0, np.nan)
        )
        df["samples_per_cpu_second"] = df["n_samples"] / cpu_time
        df["work_units_per_cpu_second"] = df["nominal_work_units"] / cpu_time
        df["cpu_efficiency_wall_over_cpu"] = wall_time / cpu_time

        paired_keys = [
            key
            for key in (
                "run_id",
                "repeat_index",
                "dataset_seed",
                "random_state",
                "n_samples",
                "n_features",
                "n_clusters",
                "k_max",
                "init",
                "cluster_std",
                "cluster_separation",
            )
            if key in df.columns
        ]
        if paired_keys:
            runtime_by_impl = df.pivot_table(
                index=paired_keys,
                columns="implementation",
                values="wall_time_s",
                aggfunc="first",
            )
            if "python" in runtime_by_impl.columns:
                python_runtime = runtime_by_impl["python"].rename("_python_wall_time_s")
                df = df.merge(python_runtime, left_on=paired_keys, right_index=True, how="left")
                df["speedup_vs_python"] = df["_python_wall_time_s"] / df["wall_time_s"]
                df.drop(columns=["_python_wall_time_s"], inplace=True)
            if "rust" in runtime_by_impl.columns:
                rust_runtime = runtime_by_impl["rust"].rename("_rust_wall_time_s")
                df = df.merge(rust_runtime, left_on=paired_keys, right_index=True, how="left")
                df["speedup_vs_rust_serial"] = df["_rust_wall_time_s"] / df["wall_time_s"]
                df.drop(columns=["_rust_wall_time_s"], inplace=True)
                if "requested_threads" in df.columns:
                    requested_threads = df["requested_threads"].replace(0, np.nan)
                    df["parallel_efficiency"] = (
                        df["speedup_vs_rust_serial"] / requested_threads
                    )

        return df
    
    def run_experiment(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        *,
        repeat_index: int = 1,
        dataset_seed: int = 42,
        random_state: int = 42,
        run_id: str = "default",
        init: str = "random",
        sklearn_n_init: int = 10,
        rust_parallel_threads: int = 0,
        cluster_std: float = 0.5,
        cluster_separation: float = 3.0,
    ) -> List[Dict]:
        """
        Run a single experiment with all implementations
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_clusters: Number of clusters for K-Means
            
        Returns:
            List of results for each implementation
        """
        results = []
        
        # Generate dataset
        dataset_path = self.generate_dataset(
            n_samples,
            n_features,
            n_clusters,
            random_state=dataset_seed,
            cluster_std=cluster_std,
            cluster_separation=cluster_separation,
        )
        dataset_id = Path(dataset_path).stem
        labels_path = Path(dataset_path).with_name(Path(dataset_path).stem + "_labels.npy")
        common_metadata: dict[str, Any] = {
            "run_id": run_id,
            "repeat_index": repeat_index,
            "dataset_seed": dataset_seed,
            "random_state": random_state,
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "labels_path": str(labels_path),
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
            "k_max": n_clusters,
            "init": init,
            "sklearn_n_init": sklearn_n_init,
            "cluster_std": cluster_std,
            "cluster_separation": cluster_separation,
        }
        
        # Run Python implementation
        python_metrics = self.run_python_impl(dataset_path, n_clusters, random_state, init)
        python_metrics.update({
            **common_metadata,
            "parallel": False,
            "requested_threads": 1,
            "actual_threads": python_metrics.get("max_threads", 0),
            "restart_policy": "single-start",
        })
        results.append(python_metrics)
        
        # Run scikit-learn implementation
        sklearn_metrics = self.run_sklearn_impl(
            dataset_path,
            n_clusters,
            random_state=random_state,
            sklearn_n_init=sklearn_n_init,
        )
        sklearn_metrics.update({
            **common_metadata,
            "parallel": False,
            "requested_threads": 1,
            "actual_threads": sklearn_metrics.get("max_threads", 0),
            "restart_policy": f"sklearn-n_init-{sklearn_n_init}",
        })
        results.append(sklearn_metrics)
        
        # Run Rust implementation (serial)
        rust_metrics = self.run_rust_impl(dataset_path, n_clusters, random_state, init)
        rust_metrics.update({
            **common_metadata,
            "parallel": False,
            "requested_threads": 1,
            "actual_threads": rust_metrics.get("max_threads", 0),
            "restart_policy": "single-start",
        })
        results.append(rust_metrics)

        # Run Rust implementation (Rayon parallel) — reuses the same dataset file
        rust_parallel_metrics = self.run_rust_parallel_impl(
            dataset_path,
            n_clusters,
            random_state=random_state,
            init=init,
            threads=rust_parallel_threads,
        )
        rust_parallel_metrics.update({
            **common_metadata,
            "parallel": True,
            "requested_threads": rust_parallel_threads,
            "actual_threads": rust_parallel_metrics.get("max_threads", 0),
            "restart_policy": "single-start",
        })
        results.append(rust_parallel_metrics)

        return results
    
    def run_benchmark_suite(self, config: Dict) -> pd.DataFrame:
        """
        Run complete benchmark suite based on configuration
        
        Args:
            config: Configuration dictionary with experiment parameters
            
        Returns:
            DataFrame with all results
        """
        all_results = []
        
        # Get experiment parameters
        sample_sizes = config.get("sample_sizes", [1000, 2000, 4000, 8000, 16000])
        feature_counts = config.get("feature_counts", [2, 4, 8, 16, 32, 64])
        cluster_counts = config.get("cluster_counts", [2, 4, 8, 16, 32, 64])
        repeats = int(config.get("repeats", 1))
        seed_base = int(config.get("seed_base", 42))
        init = config.get("init", "random")
        sklearn_n_init = int(config.get("sklearn_n_init", 10))
        rust_parallel_threads = int(config.get("rust_parallel_threads", 0))
        cluster_std = float(config.get("cluster_std", 0.5))
        cluster_separation = float(config.get("cluster_separation", 3.0))
        run_id = str(config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        total_experiments = len(sample_sizes) * len(feature_counts) * len(cluster_counts) * repeats
        experiment_num = 0
        
        print(f"Running {total_experiments} experiments...")
        print("=" * 60)

        for repeat_index in range(1, repeats + 1):
            dataset_seed = seed_base + repeat_index - 1
            random_state = seed_base + repeat_index - 1
            for n_samples in sample_sizes:
                for n_features in feature_counts:
                    for n_clusters in cluster_counts:
                        experiment_num += 1
                        print(f"\nExperiment {experiment_num}/{total_experiments}")
                        print(
                            "Parameters: "
                            f"repeat={repeat_index}/{repeats}, seed={random_state}, "
                            f"samples={n_samples}, features={n_features}, clusters={n_clusters}, "
                            f"init={init}, sklearn_n_init={sklearn_n_init}"
                        )
                        print("-" * 40)

                        try:
                            results = self.run_experiment(
                                n_samples,
                                n_features,
                                n_clusters,
                                repeat_index=repeat_index,
                                dataset_seed=dataset_seed,
                                random_state=random_state,
                                run_id=run_id,
                                init=init,
                                sklearn_n_init=sklearn_n_init,
                                rust_parallel_threads=rust_parallel_threads,
                                cluster_std=cluster_std,
                                cluster_separation=cluster_separation,
                            )
                            all_results.extend(results)
                        except Exception as e:
                            print(f"Error in experiment: {e}")
                            continue
        
        # Convert to DataFrame
        results_df = self.add_derived_metrics(pd.DataFrame(all_results))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        # Generate summary statistics
        self.generate_summary(results_df)
        
        return results_df
    
    def generate_summary(self, results_df: pd.DataFrame):
        """Generate summary statistics and save to file"""
        summary_lines = []
        summary_lines.append("K-Means Clustering Benchmark Summary")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Total experiments: {len(results_df)}")
        summary_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Group by implementation
        for impl in results_df['implementation'].unique():
            impl_df = results_df[results_df['implementation'] == impl]
            summary_lines.append(f"\n{impl.upper()} Implementation:")
            summary_lines.append("-" * 40)
            
            # Performance metrics
            summary_lines.append("Performance Metrics:")
            summary_lines.append(f"  Median wall time: {impl_df['wall_time_s'].median():.3f} seconds")
            summary_lines.append(f"  IQR wall time: {(impl_df['wall_time_s'].quantile(0.75) - impl_df['wall_time_s'].quantile(0.25)):.3f} seconds")
            summary_lines.append(f"  Median sampled RSS: {impl_df['peak_rss_mb'].median():.1f} MB")
            summary_lines.append(f"  Median RSS / 1k samples: {impl_df['rss_mb_per_1k_samples'].median():.2f} MB")
            if 'cpu_time_s' in impl_df.columns:
                summary_lines.append(f"  Median CPU time: {impl_df['cpu_time_s'].median():.3f} seconds")
                summary_lines.append(f"  Median effective cores: {impl_df['effective_cores'].median():.2f}")
            if 'avg_cpu_percent' in impl_df.columns:
                summary_lines.append(f"  Average CPU usage: {impl_df['avg_cpu_percent'].mean():.1f}%")
                summary_lines.append(f"  Max CPU usage: {impl_df['max_cpu_percent'].mean():.1f}%")
            if 'samples_per_second' in impl_df.columns:
                summary_lines.append(f"  Median throughput: {impl_df['samples_per_second'].median():.0f} samples/sec")
            if 'work_units_per_second' in impl_df.columns:
                summary_lines.append(f"  Median nominal work throughput: {impl_df['work_units_per_second'].median():.0f} units/sec")
            
            # Clustering quality metrics
            summary_lines.append("\nClustering Quality Metrics:")
            if impl_df['inertia'].notna().any():
                summary_lines.append(f"  Average inertia: {impl_df['inertia'].mean():.2f}")
            if 'silhouette_score' in impl_df.columns and impl_df['silhouette_score'].notna().any():
                summary_lines.append(f"  Average silhouette score: {impl_df['silhouette_score'].mean():.3f}")
            if 'davies_bouldin_index' in impl_df.columns and impl_df['davies_bouldin_index'].notna().any():
                summary_lines.append(f"  Average Davies-Bouldin index: {impl_df['davies_bouldin_index'].mean():.3f}")
            if 'calinski_harabasz_index' in impl_df.columns and impl_df['calinski_harabasz_index'].notna().any():
                summary_lines.append(f"  Average Calinski-Harabasz index: {impl_df['calinski_harabasz_index'].mean():.2f}")
        
        # Add comparison section
        summary_lines.append("\n\nCOMPARISON SUMMARY")
        summary_lines.append("=" * 60)
        
        # Calculate speedup ratios (using Python as baseline)
        if 'python' in results_df['implementation'].values:
            python_df = results_df[results_df['implementation'] == 'python']
            python_runtime = python_df['wall_time_s'].median()
            
            summary_lines.append("\nSpeedup relative to Python implementation:")
            for impl in results_df['implementation'].unique():
                if impl != 'python':
                    impl_df = results_df[results_df['implementation'] == impl]
                    impl_runtime = impl_df['wall_time_s'].median()
                    speedup = python_runtime / impl_runtime
                    summary_lines.append(f"  {impl.upper()}: {speedup:.2f}x faster by median wall time")
        
        # Memory efficiency comparison
        summary_lines.append("\nMemory efficiency (MB per 1000 samples):")
        for impl in results_df['implementation'].unique():
            impl_df = results_df[results_df['implementation'] == impl]
            avg_memory_efficiency = impl_df['rss_mb_per_1k_samples'].median()
            summary_lines.append(f"  {impl.upper()}: {avg_memory_efficiency:.2f} MB/1k samples (median sampled RSS)")
        
        # Save summary
        summary_file = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nSummary saved to: {summary_file}")
        
        # Generate visualization plots
        self.generate_plots(results_df)
        
    def generate_plots(self, results_df: pd.DataFrame, plot_path: Optional[Path] = None):
        """Generate comparison plots for benchmark results.

        Pass ``plot_path`` to write to a fixed filename (used when regenerating
        the canonical figure for the site); otherwise a timestamped name is used.
        """
        try:
            import matplotlib.pyplot as plt

            apply_mpl_style()
            results_df = self.add_derived_metrics(results_df)
            impl_order = ordered_implementations(results_df['implementation'].unique())

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('K-means implementation comparison', fontsize=13)
            subtitle = (
                f"{len(results_df)} rows · end-to-end CLI k-sweep · "
                "nominal work = n_samples x n_features x sum(k)"
            )
            fig.text(0.5, 0.96, subtitle, ha='center', va='top', fontsize=9, color=INK_FAINT)

            def plot_metric(ax, y_col: str, ylabel: str, title: str) -> None:
                default_offsets = {
                    'python': (6, -2),
                    'rust': (6, -10),
                    'rust_parallel': (6, 6),
                    'sklearn': (6, 14),
                }
                offset_by_metric = {
                    'wall_time_s': {
                        'python': (8, 12),
                        'rust': (8, 0),
                        'rust_parallel': (8, -12),
                        'sklearn': (8, -24),
                    },
                    'work_units_per_second': {
                        'python': (6, -8),
                        'rust': (6, -18),
                        'rust_parallel': (6, 12),
                        'sklearn': (6, 8),
                    },
                    'peak_rss_mb': {
                        'python': (8, 14),
                        'rust': (8, -14),
                        'rust_parallel': (8, 0),
                        'sklearn': (8, -10),
                    },
                }
                label_offsets = offset_by_metric.get(y_col, default_offsets)
                for impl in impl_order:
                    impl_df = results_df[results_df['implementation'] == impl]
                    grp = (
                        impl_df.groupby('nominal_work_units')[y_col]
                        .median()
                        .dropna()
                        .sort_index()
                    )
                    if grp.empty:
                        continue
                    ax.plot(
                        grp.index,
                        grp.values,
                        marker=mpl_marker(impl),
                        markersize=5,
                        lw=1.8,
                        ls=mpl_linestyle(impl),
                        color=color(impl),
                        alpha=0.9,
                        label=display_name(impl),
                    )
                    last_x, last_y = grp.index[-1], grp.values[-1]
                    label_offset = label_offsets.get(impl, (6, 0))
                    # Direct line-end label instead of a legend box.
                    ax.annotate(
                        display_name(impl),
                        xy=(last_x, last_y),
                        xytext=label_offset,
                        textcoords='offset points',
                        va='center',
                        fontsize=8,
                        color=color(impl),
                        clip_on=False,
                    )
                ax.set_xscale('log', base=2)
                ax.set_yscale('log', base=2)
                si_log_axis(ax, 'both')
                ax.set_xlabel('Nominal k-sweep work')
                ax.set_ylabel(ylabel)
                ax.set_title(title, fontsize=11)
                style_axes(ax)
                ax.margins(x=0.18)

            ax = axes[0, 0]
            plot_metric(
                ax,
                'wall_time_s',
                'Runtime (s)',
                'Runtime vs matched workload',
            )

            ax = axes[0, 1]
            plot_metric(
                ax,
                'work_units_per_second',
                'Nominal work units / s',
                'Throughput vs matched workload',
            )

            ax = axes[1, 0]
            plot_metric(
                ax,
                'peak_rss_mb',
                'Sampled RSS (MB)',
                'Memory vs matched workload',
            )

            ax = axes[1, 1]
            quality_col = None
            quality_label = None
            if 'adjusted_rand_index' in results_df.columns and results_df['adjusted_rand_index'].notna().any():
                quality_col = 'adjusted_rand_index'
                quality_label = 'Adjusted Rand Index'
            elif 'silhouette_score' in results_df.columns and results_df['silhouette_score'].notna().any():
                quality_col = 'silhouette_score'
                quality_label = 'Silhouette score'

            if quality_col:
                for impl in impl_order:
                    impl_df = results_df[results_df['implementation'] == impl]
                    grp = (
                        impl_df.groupby('nominal_work_units')
                        .agg(wall_time_s=('wall_time_s', 'median'), quality=(quality_col, 'median'))
                        .dropna()
                        .sort_values('wall_time_s')
                    )
                    if grp.empty:
                        continue
                    ax.scatter(
                        grp['wall_time_s'],
                        grp['quality'],
                        marker=mpl_marker(impl),
                        s=34,
                        color=color(impl),
                        alpha=0.85,
                        label=display_name(impl),
                    )
                ax.set_xscale('log', base=2)
                si_log_axis(ax, 'x')
                ax.set_ylim(0, 1)
                ax.set_title('Quality vs runtime frontier', fontsize=11)
                ax.set_xlabel('Runtime (s)')
                ax.set_ylabel(quality_label)
                style_axes(ax)
                ax.margins(x=0.08)
                ax.legend(fontsize=8, loc='lower right', frameon=False)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No quality metrics available', ha='center', va='center')

            plt.tight_layout(rect=(0, 0, 1, 0.94))

            # Save plot (fixed name when regenerating the canonical figure).
            if plot_path is None:
                plot_file = self.results_dir / f"benchmark_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
            else:
                plot_file = Path(plot_path)
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()

            print(f"Plots saved to: {plot_file}")

        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Benchmark runner for K-Means implementations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with small datasets')

    parser.add_argument('--sample-sizes', type=int, nargs='+', default=None, help='List of sample sizes to test')
    parser.add_argument('--feature-counts', type=int, nargs='+', default=None, help='List of feature counts to test')
    parser.add_argument('--cluster-counts', type=int, nargs='+', default=None, help='List of cluster counts to test')
    parser.add_argument('--repeats', type=int, default=1, help='Paired repeats per workload cell')
    parser.add_argument('--seed-base', type=int, default=42, help='Base seed for dataset and algorithm repeats')
    parser.add_argument('--init', choices=["random", "k-means++"], default="random", help='Init policy for Python/Rust implementations')
    parser.add_argument('--sklearn-n-init', type=int, default=10, help='Number of scikit-learn restarts per k')
    parser.add_argument('--rust-parallel-threads', type=int, default=0, help='Rayon threads for rust_parallel; 0 means all cores')
    parser.add_argument('--cluster-std', type=float, default=0.5, help='Synthetic cluster standard deviation')
    parser.add_argument('--cluster-separation', type=float, default=3.0, help='Synthetic cluster separation')
    parser.add_argument('--run-id', type=str, default=None, help='Run identifier stored in the results CSV')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save/load datasets' )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create benchmark runner
    runner = BenchmarkRunner(results_dir=args.results_dir, data_dir=args.data_dir)
    
    # Configure experiments
    if args.quick:
        # Quick test configuration
        config = {
            "sample_sizes":   [1000, 2000, 4000, 8000],
            "feature_counts": [2, 4, 8],
            "cluster_counts": [8]
        }
    else:
        # Full benchmark configuration
        config = {
            "sample_sizes": args.sample_sizes or [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000],
            "feature_counts": args.feature_counts or [2, 4, 8, 16, 32, 64, 128],
            "cluster_counts": args.cluster_counts or [64]
        }

    config.update({
        "repeats": args.repeats,
        "seed_base": args.seed_base,
        "init": args.init,
        "sklearn_n_init": args.sklearn_n_init,
        "rust_parallel_threads": args.rust_parallel_threads,
        "cluster_std": args.cluster_std,
        "cluster_separation": args.cluster_separation,
        "run_id": args.run_id,
    })
    
    print("K-Means Clustering Benchmark Runner")
    print("=" * 60)
    print("Configuration:")
    print(f"  Sample sizes: {config['sample_sizes']}")
    print(f"  Feature counts: {config['feature_counts']}")
    print(f"  Cluster counts: {config['cluster_counts']}")
    print(f"  Repeats: {config['repeats']}")
    print(f"  Seed base: {config['seed_base']}")
    print(f"  Init policy: {config['init']}")
    print(f"  sklearn n_init: {config['sklearn_n_init']}")
    print(f"  Rust parallel threads: {config['rust_parallel_threads'] or 'all cores'}")
    print(f"  Cluster std/separation: {config['cluster_std']} / {config['cluster_separation']}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Data directory: {args.data_dir}")
    print("=" * 60)
    
    # Run benchmarks
    results_df = runner.run_benchmark_suite(config)
    
    print("\nBenchmark complete!")
    print(f"Total experiments run: {len(results_df)}")


if __name__ == "__main__":
    main()
