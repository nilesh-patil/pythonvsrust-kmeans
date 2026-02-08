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

    _VALID_INIT = frozenset({"random", "k-means++"})

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        random_state: Optional[int] = None,
        init: str = "random",
    ):
        """
        Initialize K-Means clustering

        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            init: Centroid initialisation strategy — ``"random"`` (default) or
                ``"k-means++"`` (Arthur & Vassilvitskii 2007).
        """
        if init not in self._VALID_INIT:
            raise ValueError(f"init must be one of {sorted(self._VALID_INIT)!r}, got {init!r}")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
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
        Initialize centroids, dispatching to the method selected by ``self.init``.

        Args:
            X: Data points (n_samples, n_features)

        Returns:
            Initial centroids (n_clusters, n_features)
        """
        if self.init == "k-means++":
            return self._initialize_centroids_kpp(X)
        return self._initialize_centroids_random(X)

    def _initialize_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids by sampling k points uniformly at random (without replacement).

        Uses ``np.random.RandomState`` rather than the global RNG so that
        concurrent runs with different seeds don't interfere with each other.

        Args:
            X: Data points (n_samples, n_features)

        Returns:
            Initial centroids (n_clusters, n_features)
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        random_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        return X[random_indices]

    def _initialize_centroids_kpp(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the Arthur & Vassilvitskii (2007) k-means++ algorithm.

        Picks the first centroid uniformly at random, then iteratively selects
        each subsequent centroid with probability proportional to the squared
        Euclidean distance to the nearest already-chosen centroid (D² sampling).
        Squared distances are used throughout — no sqrt is needed for the
        probability weights, and it avoids numerical drift.

        Args:
            X: Data points (n_samples, n_features)

        Returns:
            Initial centroids (n_clusters, n_features) — each row is a point from X.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        # Step 1 — pick the first centroid uniformly
        first_idx = rng.randint(0, n_samples)
        chosen_indices = [first_idx]

        # Maintain a running vector of min-squared-distances to chosen centroids.
        # Initialised to squared distance from every point to the first centroid.
        diff = X - X[first_idx]
        min_sq_dist = np.einsum("ij,ij->i", diff, diff)  # (n_samples,)

        for _ in range(1, self.n_clusters):
            # Step 2 — sample proportional to D²
            total = min_sq_dist.sum()
            if total == 0.0:
                # All remaining candidate points coincide with existing centroids;
                # fall back to uniform sampling among unchosen indices.
                unchosen = np.setdiff1d(np.arange(n_samples), chosen_indices)
                next_idx = int(rng.choice(unchosen))
            else:
                probabilities = min_sq_dist / total
                next_idx = int(rng.choice(n_samples, p=probabilities))

            chosen_indices.append(next_idx)

            # Update min-squared-distance with the newly added centroid
            diff = X - X[next_idx]
            sq_dist_to_new = np.einsum("ij,ij->i", diff, diff)
            np.minimum(min_sq_dist, sq_dist_to_new, out=min_sq_dist)

        return X[chosen_indices]
    
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
                # If no points assigned, reinitialize from a random data point.
                # Use a fresh RandomState seeded off k so this edge-case path is
                # still deterministic without touching the global RNG.
                rng = np.random.RandomState(
                    None if self.random_state is None else self.random_state + k
                )
                centroids[k] = X[rng.choice(X.shape[0])]
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
    parser.add_argument(
        '--init',
        type=str,
        choices=["random", "k-means++"],
        default="random",
        help='Centroid initialisation method: "random" (default) or "k-means++"',
    )

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


def run_kmeans_multiple_k(X: np.ndarray, k_max: int, random_state: int, init: str = "random") -> dict:
    """
    Run k-means for multiple values of k.

    Args:
        X: Feature array
        k_max: Maximum number of clusters
        random_state: Random seed
        init: Centroid initialisation method — ``"random"`` (default) or ``"k-means++"``.

    Returns:
        Dictionary mapping k to cluster labels
    """
    results = {}

    for k in range(1, min(k_max + 1, X.shape[0] + 1)):
        print(f"Running k-means with k={k}...")
        kmeans = KMeansClustering(n_clusters=k, random_state=random_state, init=init)
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
    print(f"Init method:  {args.init}")
    print("=" * 40)

    # Load data
    print("\nLoading data...")
    df, X, id_df = load_data(args.input, args.id_column)
    print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")

    # Run k-means for multiple k values
    print(f"\nRunning k-means for k=1 to k={args.k_clusters_max}...")
    results = run_kmeans_multiple_k(X, args.k_clusters_max, args.random_state, init=args.init)
    
    # Save results
    print("\nSaving results...")
    save_results(df, results, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
