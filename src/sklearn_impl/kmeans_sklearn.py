#!/usr/bin/env python3
"""
K-Means Clustering using scikit-learn
Wrapper for benchmarking purposes
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='K-Means using scikit-learn')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--id_column', type=str, default='ID', help='ID column name')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Separate features
    feature_cols = [col for col in df.columns if col != args.id_column]
    X = df[feature_cols].values
    
    # Run K-Means
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Save results
    output_df = df.copy()
    output_df['cluster'] = labels
    output_df['inertia'] = kmeans.inertia_
    output_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
