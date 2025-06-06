#!/usr/bin/env python3
"""
K-Means Clustering Implementation
Pure Python implementation using only numpy and pandas (no scikit-learn)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class KMeansClustering:
    """K-Means clustering implementation from scratch"""
    
    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: Optional[int] = None):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _calculate_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance between points and centroids
        
        Args:
            X: Data points (n_samples, n_features)
            centroids: Cluster centers (n_clusters, n_features)
            
        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        return distances
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using random selection from data points
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Initial centroids (n_clusters, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the nearest centroid
        
        Args:
            X: Data points (n_samples, n_features)
            centroids: Cluster centers (n_clusters, n_features)
            
        Returns:
            Cluster assignments (n_samples,)
        """
        distances = self._calculate_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids based on mean of assigned points
        
        Args:
            X: Data points (n_samples, n_features)
            labels: Cluster assignments (n_samples,)
            
        Returns:
            Updated centroids (n_clusters, n_features)
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If no points assigned, keep the centroid as is or reinitialize
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids
    
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """
        Fit K-Means clustering to data
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Self
        """
        # Handle edge case where n_clusters >= n_samples
        if self.n_clusters >= X.shape[0]:
            self.labels = np.arange(X.shape[0])
            self.centroids = X.copy()
            return self
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iterative optimization
        for iteration in range(self.max_iter):
            # Assign clusters
            new_labels = self._assign_clusters(X, self.centroids)
            
            # Check for convergence
            if self.labels is not None and np.array_equal(self.labels, new_labels):
                break
                
            self.labels = new_labels
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Cluster assignments (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
        return self._assign_clusters(X, self.centroids)


def parse_arguments():
    
    parser = argparse.ArgumentParser(description='K-Means Clustering Implementation')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--k_clusters_max', type=int, required=True, help='Number of max clusters')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--id_column', type=str, default='ID', help='ID column name')
    
    return parser.parse_args()


def load_data(filepath: str, id_column: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        id_column: Name of ID column
        
    Returns:
        Tuple of (full dataframe, feature array, id dataframe)
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Check if ID column exists
    if id_column not in df.columns:
        print(f"Error: ID column '{id_column}' not found in input file")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Separate ID column and features
    id_df = df[[id_column]]
    feature_columns = [col for col in df.columns if col != id_column]
    
    if not feature_columns:
        print("Error: No feature columns found (only ID column present)")
        sys.exit(1)
    
    # Convert features to numpy array
    X = df[feature_columns].values.astype(float)
    
    # Handle missing values
    if np.any(np.isnan(X)):
        print("Warning: Missing values found in features. Filling with column means.")
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]
    
    return df, X, id_df


def run_kmeans_multiple_k(X: np.ndarray, k_max: int, random_state: int) -> dict:
    """
    Run k-means for multiple values of k
    
    Args:
        X: Feature array
        k_max: Maximum number of clusters
        random_state: Random seed
        
    Returns:
        Dictionary mapping k to cluster labels
    """
    results = {}
    
    for k in range(1, min(k_max + 1, X.shape[0] + 1)):
        print(f"Running k-means with k={k}...")
        kmeans = KMeansClustering(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        results[k] = kmeans.labels
    
    return results


def save_results(df: pd.DataFrame, results: dict, output_path: str):
    """
    Save clustering results to CSV
    
    Args:
        df: Original dataframe
        results: Dictionary of k -> cluster labels
        output_path: Path to save output
    """
    # Start with original dataframe
    output_df = df.copy()
    
    # Add cluster columns for each k
    for k, labels in results.items():
        output_df[f'cluster_{k}'] = labels
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    try:
        output_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    print("K-Means Clustering Implementation")
    print("=" * 40)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Max clusters: {args.k_clusters_max}")
    print(f"ID column: {args.id_column}")
    print(f"Random state: {args.random_state}")
    print("=" * 40)
    
    # Load data
    print("\nLoading data...")
    df, X, id_df = load_data(args.input, args.id_column)
    print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run k-means for multiple k values
    print(f"\nRunning k-means for k=1 to k={args.k_clusters_max}...")
    results = run_kmeans_multiple_k(X, args.k_clusters_max, args.random_state)
    
    # Save results
    print("\nSaving results...")
    save_results(df, results, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
