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

pd.options.mode.copy_on_write = True

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
        
        # Monitor memory usage and CPU
        try:
            ps_process = psutil.Process(process.pid)
            peak_memory = 0
            cpu_percentages = []
            
            # Poll memory and CPU usage while process is running
            while process.poll() is None:
                try:
                    memory_info = ps_process.memory_info()
                    peak_memory = max(peak_memory, memory_info.rss)
                    
                    # Get CPU usage (returns percentage since last call)
                    cpu_percent = ps_process.cpu_percent(interval=0.01)
                    if cpu_percent > 0:  # Ignore initial 0 values
                        cpu_percentages.append(cpu_percent)
                    
                    time.sleep(0.01)  # Poll every 10ms
                except psutil.NoSuchProcess:
                    break
                    
        except Exception as e:
            print(f"Warning: Could not monitor memory: {e}")
            peak_memory = 0
            cpu_percentages = []
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        end_time = time.perf_counter()
        
        # Calculate CPU statistics
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        
        return {
            "runtime": end_time - start_time,
            "peak_memory_bytes": peak_memory,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
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
    
    def run_rust_impl(self, dataset_path: str, n_clusters: int) -> Dict:
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
            "--random-state", "42"
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
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
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
        
        try:
            # Calculate inertia
            metrics['inertia'] = self.calculate_inertia(dataset_path, results_df, n_clusters)
            
            # Only calculate other metrics if we have more than one cluster
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                # Silhouette Score (-1 to 1, higher is better)
                metrics['silhouette_score'] = silhouette_score(X, labels)
                
                # Davies-Bouldin Index (lower is better)
                metrics['davies_bouldin_index'] = davies_bouldin_score(X, labels)
                
                # Calinski-Harabasz Index (higher is better)
                metrics['calinski_harabasz_index'] = calinski_harabasz_score(X, labels)
                
            # Calculate samples per second throughput
            n_samples = len(X)
            if 'runtime' in metrics:
                metrics['samples_per_second'] = n_samples / metrics['runtime']
                
        except Exception as e:
            print(f"Warning: Error calculating clustering metrics: {e}")
            
        return metrics
    
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
        
        # Run Rust implementation
        rust_metrics = self.run_rust_impl(dataset_path, n_clusters)
        rust_metrics.update({
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters
        })
        results.append(rust_metrics)
        
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
            
            # Performance metrics
            summary_lines.append("Performance Metrics:")
            summary_lines.append(f"  Average runtime: {impl_df['runtime'].mean():.3f} seconds")
            summary_lines.append(f"  Average memory: {impl_df['peak_memory_mb'].mean():.1f} MB")
            if 'avg_cpu_percent' in impl_df.columns:
                summary_lines.append(f"  Average CPU usage: {impl_df['avg_cpu_percent'].mean():.1f}%")
                summary_lines.append(f"  Max CPU usage: {impl_df['max_cpu_percent'].mean():.1f}%")
            if 'samples_per_second' in impl_df.columns:
                summary_lines.append(f"  Throughput: {impl_df['samples_per_second'].mean():.0f} samples/sec")
            
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
            python_runtime = python_df['runtime'].mean()
            
            summary_lines.append("\nSpeedup relative to Python implementation:")
            for impl in results_df['implementation'].unique():
                if impl != 'python':
                    impl_df = results_df[results_df['implementation'] == impl]
                    impl_runtime = impl_df['runtime'].mean()
                    speedup = python_runtime / impl_runtime
                    summary_lines.append(f"  {impl.upper()}: {speedup:.2f}x faster")
        
        # Memory efficiency comparison
        summary_lines.append("\nMemory efficiency (MB per 1000 samples):")
        for impl in results_df['implementation'].unique():
            impl_df = results_df[results_df['implementation'] == impl]
            # Calculate memory per 1000 samples
            impl_df['memory_per_1k_samples'] = (impl_df['peak_memory_mb'] / impl_df['n_samples']) * 1000
            avg_memory_efficiency = impl_df['memory_per_1k_samples'].mean()
            summary_lines.append(f"  {impl.upper()}: {avg_memory_efficiency:.2f} MB/1k samples")
        
        # Save summary
        summary_file = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nSummary saved to: {summary_file}")
        
        # Generate visualization plots
        self.generate_plots(results_df)
        
    def generate_plots(self, results_df: pd.DataFrame):
        """Generate comparison plots for benchmark results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('K-Means Implementation Comparison', fontsize=16)
            
            # 1. Runtime comparison
            ax = axes[0, 0]
            sns.boxplot(data=results_df, x='implementation', y='runtime', ax=ax)
            ax.set_title('Runtime Distribution')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_xlabel('Implementation')
            
            # 2. Memory usage comparison
            ax = axes[0, 1]
            sns.boxplot(data=results_df, x='implementation', y='peak_memory_mb', ax=ax)
            ax.set_title('Memory Usage Distribution')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_xlabel('Implementation')
            
            # 3. Scalability plot (runtime vs data size)
            ax = axes[1, 0]
            for impl in results_df['implementation'].unique():
                impl_df = results_df[results_df['implementation'] == impl]
                grouped = impl_df.groupby('n_samples')['runtime'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=impl.upper())
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Scalability: Runtime vs Data Size')
            ax.set_xlabel('Number of Samples')
            ax.set_ylabel('Runtime (seconds)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Quality comparison (if available)
            ax = axes[1, 1]
            if 'silhouette_score' in results_df.columns:
                quality_metrics = results_df.groupby('implementation')['silhouette_score'].mean()
                bars = ax.bar(quality_metrics.index, quality_metrics.values)
                ax.set_title('Average Clustering Quality (Silhouette Score)')
                ax.set_ylabel('Silhouette Score')
                ax.set_xlabel('Implementation')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.results_dir / f"benchmark_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
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
