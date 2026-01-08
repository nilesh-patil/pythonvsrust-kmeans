#!/usr/bin/env python3
"""
Generate synthetic clustered data for K-Means clustering experiments
"""

import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple
import sys


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic clustered data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n_rows',
        type=int,
        default=1000,
        help='Number of rows (data points) to generate'
    )
    
    parser.add_argument(
        '--n_features',
        type=int,
        default=5,
        help='Number of feature columns to generate'
    )
    
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=3,
        help='Number of distinct clusters to generate'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/input.csv',
        help='Output file path'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--cluster_separation',
        type=float,
        default=3.0,
        help='Minimum separation between cluster centers (in units of std dev)'
    )
    
    parser.add_argument(
        '--cluster_std',
        type=float,
        default=0.5,
        help='Standard deviation for each cluster (smaller = tighter clusters)'
    )
    
    return parser.parse_args()


def generate_cluster_centers(n_clusters: int, n_features: int, 
                           min_separation: float, random_state: int) -> np.ndarray:
    """
    Generate cluster centers that are well-separated
    
    Args:
        n_clusters: Number of clusters
        n_features: Number of features
        min_separation: Minimum distance between cluster centers
        random_state: Random seed
        
    Returns:
        Array of cluster centers (n_clusters, n_features)
    """
    np.random.seed(random_state)
    
    centers = np.zeros((n_clusters, n_features))
    
    # First center at origin
    centers[0] = np.random.randn(n_features) * 2
    
    # Place subsequent centers ensuring minimum separation
    for i in range(1, n_clusters):
        valid = False
        attempts = 0
        max_attempts = 1000
        
        while not valid and attempts < max_attempts:
            # Generate a candidate center
            candidate = np.random.randn(n_features) * 2
            # Scale and shift to ensure separation
            candidate += np.random.choice([-1, 1], n_features) * min_separation * (i + 1)
            
            # Check distance to all existing centers
            valid = True
            for j in range(i):
                dist = np.linalg.norm(candidate - centers[j])
                if dist < min_separation:
                    valid = False
                    break
            
            attempts += 1
        
        if attempts >= max_attempts:
            # If we can't find a valid position, place it far away
            candidate = centers[i-1] + np.random.choice([-1, 1], n_features) * min_separation * 2
        
        centers[i] = candidate
    
    return centers


def generate_clustered_data(n_rows: int, n_features: int, n_clusters: int,
                           cluster_std: float, cluster_separation: float,
                           random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered data
    
    Args:
        n_rows: Number of data points
        n_features: Number of features
        n_clusters: Number of clusters
        cluster_std: Standard deviation for each cluster
        cluster_separation: Minimum separation between clusters
        random_state: Random seed
        
    Returns:
        Tuple of (data array, cluster labels)
    """
    np.random.seed(random_state)
    
    # Generate cluster centers
    centers = generate_cluster_centers(n_clusters, n_features, 
                                     cluster_separation, random_state)
    
    # Allocate data array
    data = np.zeros((n_rows, n_features))
    labels = np.zeros(n_rows, dtype=int)
    
    # Determine number of points per cluster
    points_per_cluster = n_rows // n_clusters
    remainder = n_rows % n_clusters
    
    current_idx = 0
    for cluster_idx in range(n_clusters):
        # Number of points for this cluster
        n_points = points_per_cluster + (1 if cluster_idx < remainder else 0)
        
        # Generate points for this cluster
        cluster_data = np.random.randn(n_points, n_features) * cluster_std
        cluster_data += centers[cluster_idx]
        
        # Store in data array
        end_idx = current_idx + n_points
        data[current_idx:end_idx] = cluster_data
        labels[current_idx:end_idx] = cluster_idx
        
        current_idx = end_idx
    
    # Shuffle the data to mix clusters
    shuffle_idx = np.random.permutation(n_rows)
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return data, labels


def save_data(data: np.ndarray, output_path: str):
    """
    Save generated data to CSV file
    
    Args:
        data: Generated data array
        output_path: Path to save the CSV file
    """
    n_rows, n_features = data.shape
    
    # Create DataFrame with ID column
    df_dict = {'ID': range(1, n_rows + 1)}
    
    # Add feature columns
    for i in range(n_features):
        df_dict[f'feature_{i+1}'] = data[:, i]
    
    df = pd.DataFrame(df_dict)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)


def visualize_clusters_2d(data: np.ndarray, labels: np.ndarray):
    """
    Simple text-based visualization of clusters (first 2 dimensions)
    """
    if data.shape[1] < 2:
        return
    
    print("\nCluster visualization (first 2 features):")
    print("-" * 40)
    
    # Get data range for scaling
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    
    # Create a simple grid
    grid_width = 40
    grid_height = 20
    
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Plot points
    symbols = ['o', '*', '+', 'x', '#', '@', '&', '%', '!', '?']
    for i in range(len(data)):
        x = int((data[i, 0] - x_min) / (x_max - x_min) * (grid_width - 1))
        y = int((data[i, 1] - y_min) / (y_max - y_min) * (grid_height - 1))
        
        x = max(0, min(grid_width - 1, x))
        y = max(0, min(grid_height - 1, y))
        
        cluster = labels[i]
        symbol = symbols[cluster % len(symbols)]
        
        # Flip y-axis for display
        grid[grid_height - 1 - y][x] = symbol
    
    # Print grid
    for row in grid:
        print(''.join(row))
    
    print("-" * 40)
    print("Clusters: ", end="")
    for i in range(len(set(labels))):
        print(f"{symbols[i % len(symbols)]}={i} ", end="")
    print()


def main():
    """Main function"""
    args = parse_arguments()
    
    print("Synthetic Data Generator for K-Means Clustering")
    print("=" * 50)
    print(f"Number of rows: {args.n_rows}")
    print(f"Number of features: {args.n_features}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Cluster separation: {args.cluster_separation}")
    print(f"Cluster std dev: {args.cluster_std}")
    print(f"Output file: {args.output}")
    print(f"Random state: {args.random_state}")
    print("=" * 50)
    
    # Validate inputs
    if args.n_rows <= 0:
        print("Error: Number of rows must be positive")
        sys.exit(1)
    
    if args.n_features <= 0:
        print("Error: Number of features must be positive")
        sys.exit(1)
    
    if args.n_clusters <= 0:
        print("Error: Number of clusters must be positive")
        sys.exit(1)
    
    if args.n_clusters > args.n_rows:
        print("Error: Number of clusters cannot exceed number of rows")
        sys.exit(1)
    
    # Generate data
    print("\nGenerating clustered data...")
    data, labels = generate_clustered_data(
        args.n_rows, 
        args.n_features, 
        args.n_clusters,
        args.cluster_std,
        args.cluster_separation,
        args.random_state
    )
    
    # Show statistics
    print(f"\nGenerated {args.n_rows} data points in {args.n_clusters} clusters")
    print(f"Data shape: {data.shape}")
    print(f"Feature ranges:")
    for i in range(args.n_features):
        print(f"  Feature {i+1}: [{data[:, i].min():.2f}, {data[:, i].max():.2f}]")
    
    # Show cluster distribution
    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} points ({count/args.n_rows*100:.1f}%)")
    
    # Simple visualization if 2D or more
    if args.n_features >= 2 and args.n_rows <= 500:
        visualize_clusters_2d(data, labels)
    
    # Save data
    print(f"\nSaving data...")
    save_data(data, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
