# K-Means Clustering - Pure Python Implementation

This directory contains a native Python implementation of the k-means clustering algorithm without using scikit-learn or other machine learning libraries.

## Overview

The `kmeans.py` file implements k-means clustering from scratch using only standard Python libraries and NumPy / Pandas for numerical computations. This implementation provides a clear understanding of how the k-means algorithm works under the hood.

## File Description

### `kmeans.py`
- **Purpose**: Implements k-means clustering algorithm in pure Python
- **Dependencies**: Only standard Python libraries, numpy and pandas (no scikit-learn)
- **Input**: CSV file with features and unique ID column
- **Output**: CSV file with cluster assignments

## Configuration

The script uses configurable parameters at the start of the file:

- **Input CSV Path**: Path to the input CSV file containing the data to be clustered
- **Output CSV Path**: Path where the results will be saved
- **ID Column**: Column name that contains unique identifiers for each row
- **Feature Columns**: Columns to be used for clustering (all columns except ID)

## Input Data Format

The input CSV file should have the following structure:
- **ID Column**: Contains unique identifiers for each data point
- **Feature Columns**: Each column represents a feature/dimension for clustering
- **Headers**: First row should contain column names

Example:
```csv
ID,feature1,feature2,feature3
1,2.5,3.1,1.8
2,1.2,2.4,3.5
3,4.1,1.9,2.7
...
```

## Output Data Format

The output CSV file will contain:
- **Original ID Column**: Preserved from input
- **All Original Features**: All feature columns from input
- **Cluster Assignments**: Multiple columns (cluster_1, cluster_2, ..., cluster_k) showing cluster assignments for each k value

Example:
```csv
ID,feature1,feature2,feature3,cluster_1,cluster_2,cluster_3
1,2.5,3.1,1.8,0,0,1
2,1.2,2.4,3.5,0,1,0
3,4.1,1.9,2.7,0,0,2
...
```

## Algorithm Implementation

The k-means implementation includes:

1. **Initialization**: Random placement of initial centroids
2. **Assignment Step**: Assign each point to the nearest centroid
3. **Update Step**: Recalculate centroids based on assigned points
4. **Convergence Check**: Repeat until centroids stabilize or max iterations reached
5. **Output Generation**: Save results with cluster assignments

## Usage

1. Configure the input/output file paths while running `kmeans.py`, with default paths under "data" folder of the home directory
2. Set the desired maximum number of clusters (k)
3. The kmeans script runs from 1 cluster to k clusters and creates columns for each k value in the output csv
4. Run the script:

   ```bash
   python kmeans.py --input="input path" --output="output path" --k_clusters_max="10"
   ```

   Optional parameters:
   ```bash
   python kmeans.py --input="data.csv" --output="results.csv" --k_clusters_max="10" --id_column="ID" --random_state="42"
   ```

## Features

- **Pure Python**: No external ML libraries required
- **Configurable**: Easy to modify input/output paths and parameters
- **ID Preservation**: Maintains original row identifiers
- **CSV Compatible**: Works with standard CSV format
- **Extensible**: Easy to modify for different distance metrics or initialization methods

## Notes

- The algorithm uses Euclidean distance for similarity measurement
- Random initialization may lead to different results on different runs
- Consider running multiple times and selecting the best result based on inertia
- Large datasets may require performance optimizations 