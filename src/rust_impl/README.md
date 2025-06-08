# K-Means Clustering - Pure Rust Implementation

This directory contains a native Rust implementation of the k-means clustering algorithm without using external machine learning libraries.

## Overview

The `main.rs` file implements k-means clustering from scratch using only standard Rust libraries and minimal external crates for CSV handling and numerical computations. This implementation provides a clear understanding of how the k-means algorithm works under the hood, with the performance benefits and memory safety guarantees of Rust.

## File Description

### `main.rs`
- **Purpose**: Implements k-means clustering algorithm in pure Rust
- **Dependencies**: Only standard Rust library and minimal crates (csv for file I/O, no ML frameworks)
- **Input**: CSV file with features and unique ID column
- **Output**: CSV file with cluster assignments

## Configuration

The program accepts command-line arguments for configuration:

- **Input CSV Path**: Path to the input CSV file containing the data to be clustered
- **Output CSV Path**: Path where the results will be saved
- **ID Column**: Column name that contains unique identifiers for each row
- **Feature Columns**: Columns to be used for clustering (all columns except ID)
- **Max Clusters**: Maximum number of clusters (k) to compute

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

1. **Initialization**: Random placement of initial centroids using Rust's thread-safe RNG
2. **Assignment Step**: Assign each point to the nearest centroid using efficient distance calculations
3. **Update Step**: Recalculate centroids based on assigned points
4. **Convergence Check**: Repeat until centroids stabilize or max iterations reached
5. **Output Generation**: Save results with cluster assignments

## Usage

1. Build the project:
   ```bash
   cargo build --release
   ```

2. Run the clustering algorithm:
   ```bash
   cargo run --release -- --input="input_path.csv" --output="output_path.csv" --k-clusters-max=10
   ```

   With optional parameters:
   ```bash
   cargo run --release -- --input="data.csv" --output="results.csv" --k-clusters-max=10 --id-column="ID" --random-seed=42
   ```

## Command Line Arguments

- `--input`: Path to input CSV file (required)
- `--output`: Path to output CSV file (required)
- `--k-clusters-max`: Maximum number of clusters to compute (required)
- `--id-column`: Name of the ID column (optional, default: "ID")
- `--random-seed`: Seed for random number generation (optional)
- `--max-iterations`: Maximum iterations per k-means run (optional, default: 300)

## Features

- **Pure Rust**: No external ML libraries required
- **High Performance**: Leverages Rust's zero-cost abstractions and efficient memory management
- **Memory Safe**: Guaranteed memory safety without garbage collection overhead
- **Parallel Processing**: Can leverage Rust's fearless concurrency for parallel computations
- **Type Safe**: Strong typing prevents runtime errors common in dynamic languages
- **CSV Compatible**: Works with standard CSV format using the `csv` crate
- **Configurable**: Command-line interface for easy parameter adjustment
- **ID Preservation**: Maintains original row identifiers

## Dependencies

Add these to your `Cargo.toml`:

```toml
[dependencies]
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
clap = { version = "4.4", features = ["derive"] }
rand = "0.8"
```

## Notes

- The algorithm uses Euclidean distance for similarity measurement
- Random initialization can be controlled with the `--random-seed` parameter for reproducible results
- The Rust implementation offers significant performance improvements over interpreted languages, especially for large datasets
- Memory usage is optimized through Rust's ownership system
- Consider using the `--release` flag for optimal performance 