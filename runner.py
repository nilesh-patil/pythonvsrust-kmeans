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
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import psutil
from pathlib import Path


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
        
    def get_dataset_filename(self, n_samples: int, n_features: int, n_clusters: int) -> str:
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
        params_str = f"n{n_samples}_f{n_features}_c{n_clusters}"
        hash_str = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"dataset_{params_str}_{hash_str}.csv"
    
    def generate_dataset(self, n_samples: int, n_features: int, n_clusters: int, 
                        random_state: int = 42) -> str:
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
        filename = self.get_dataset_filename(n_samples, n_features, n_clusters)
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
            "--random_state", str(random_state)
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
        # Start process
        start_time = time.perf_counter()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor memory usage
        try:
            ps_process = psutil.Process(process.pid)
            peak_memory = 0
            
            # Poll memory usage while process is running
            while process.poll() is None:
                try:
                    memory_info = ps_process.memory_info()
                    peak_memory = max(peak_memory, memory_info.rss)
                    time.sleep(0.01)  # Poll every 10ms
                except psutil.NoSuchProcess:
                    break
                    
        except Exception as e:
            print(f"Warning: Could not monitor memory: {e}")
            peak_memory = 0
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        end_time = time.perf_counter()
        
        return {
            "runtime": end_time - start_time,
            "peak_memory_bytes": peak_memory,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "exit_code": process.returncode,
            "stdout": stdout.decode('utf-8'),
            "stderr": stderr.decode('utf-8')
        }
    
    def run_python_impl(self, dataset_path: str, n_clusters: int) -> Dict:
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
            "--random_state", "42"
        ]
        
        metrics = self.measure_process_metrics(cmd)
        
        # Calculate inertia from results
        inertia = None
        if metrics["exit_code"] == 0 and output_path.exists():
            try:
                df = pd.read_csv(output_path)
                # Calculate inertia (within-cluster sum of squares)
                inertia = self.calculate_inertia(dataset_path, df, n_clusters)
                # Clean up output file
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not calculate inertia: {e}")
        
        metrics["inertia"] = inertia
        metrics["implementation"] = "python"
        return metrics
    
    def run_sklearn_impl(self, dataset_path: str, n_clusters: int) -> Dict:
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
            "--random_state", "42"
        ]
        
        metrics = self.measure_process_metrics(cmd)
        
        # Calculate inertia from results
        inertia = None
        if metrics["exit_code"] == 0 and output_path.exists():
            try:
                df = pd.read_csv(output_path)
                # Calculate inertia (within-cluster sum of squares)
                inertia = self.calculate_inertia(dataset_path, df, n_clusters)
                # Clean up output file
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not calculate inertia: {e}")
        
        metrics["inertia"] = inertia
        metrics["implementation"] = "sklearn"
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
    
    def run_experiment(self, n_samples: int, n_features: int, n_clusters: int) -> List[Dict]:
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
        dataset_path = self.generate_dataset(n_samples, n_features, n_clusters)
        
        # Run Python implementation
        python_metrics = self.run_python_impl(dataset_path, n_clusters)
        python_metrics.update({
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters
        })
        results.append(python_metrics)
        
        # Run scikit-learn implementation
        sklearn_metrics = self.run_sklearn_impl(dataset_path, n_clusters)
        sklearn_metrics.update({
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters
        })
        results.append(sklearn_metrics)
        
        # TODO: Run Rust implementation when available
        
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
        sample_sizes = config.get("sample_sizes", [1000, 10000, 100000])
        feature_counts = config.get("feature_counts", [2, 10, 50, 100])
        cluster_counts = config.get("cluster_counts", [2, 5, 10, 20, 30])
        
        total_experiments = len(sample_sizes) * len(feature_counts) * len(cluster_counts)
        experiment_num = 0
        
        print(f"Running {total_experiments} experiments...")
        print("=" * 60)
        
        for n_samples in sample_sizes:
            for n_features in feature_counts:
                for n_clusters in cluster_counts:
                    experiment_num += 1
                    print(f"\nExperiment {experiment_num}/{total_experiments}")
                    print(f"Parameters: samples={n_samples}, features={n_features}, clusters={n_clusters}")
                    print("-" * 40)
                    
                    try:
                        results = self.run_experiment(n_samples, n_features, n_clusters)
                        all_results.extend(results)
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
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
            summary_lines.append(f"Average runtime: {impl_df['runtime'].mean():.3f} seconds")
            summary_lines.append(f"Average memory: {impl_df['peak_memory_mb'].mean():.1f} MB")
            if impl_df['inertia'].notna().any():
                summary_lines.append(f"Average inertia: {impl_df['inertia'].mean():.2f}")
        
        # Save summary
        summary_file = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nSummary saved to: {summary_file}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Benchmark runner for K-Means implementations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with small datasets'
    )
    
    parser.add_argument(
        '--sample-sizes',
        type=int,
        nargs='+',
        default=None,
        help='List of sample sizes to test'
    )
    
    parser.add_argument(
        '--feature-counts',
        type=int,
        nargs='+',
        default=None,
        help='List of feature counts to test'
    )
    
    parser.add_argument(
        '--cluster-counts',
        type=int,
        nargs='+',
        default=None,
        help='List of cluster counts to test'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to save/load datasets'
    )
    
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
            "sample_sizes": [1000, 5000],
            "feature_counts": [2, 10],
            "cluster_counts": [3, 5]
        }
    else:
        # Full benchmark configuration
        config = {
            "sample_sizes": args.sample_sizes or [1000, 10000, 100000],
            "feature_counts": args.feature_counts or [2, 10, 50, 100],
            "cluster_counts": args.cluster_counts or [2, 5, 10, 20, 30]
        }
    
    print("K-Means Clustering Benchmark Runner")
    print("=" * 60)
    print("Configuration:")
    print(f"  Sample sizes: {config['sample_sizes']}")
    print(f"  Feature counts: {config['feature_counts']}")
    print(f"  Cluster counts: {config['cluster_counts']}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Data directory: {args.data_dir}")
    print("=" * 60)
    
    # Run benchmarks
    results_df = runner.run_benchmark_suite(config)
    
    print("\nBenchmark complete!")
    print(f"Total experiments run: {len(results_df)}")


if __name__ == "__main__":
    main()
