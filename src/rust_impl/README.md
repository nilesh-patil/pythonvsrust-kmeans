# K-Means Clustering - Pure Rust Implementation

This directory contains a native Rust implementation of the k-means clustering algorithm without using external machine learning libraries.

## Overview

The library (`src/lib.rs`) implements k-means clustering from scratch using only standard Rust libraries and minimal external crates for CSV handling and numerical computations. The binary (`src/main.rs`) provides a CLI wrapper. This implementation provides a clear understanding of how the k-means algorithm works under the hood, with the performance benefits and memory safety guarantees of Rust.

## File Description

### `src/lib.rs`
- **Purpose**: Core KMeans struct, DataPoint, and InitMethod types — importable as a library and tested in-process.

### `src/main.rs`
- **Purpose**: CLI binary; parses arguments, loads CSV, runs clustering, writes output.

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

1. **Initialization**: Random or k-means++ centroid initialization (see `--init` below)
2. **Assignment Step**: Assign each point to the nearest centroid using Euclidean distance
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
   cargo run --release -- --input="data.csv" --output="results.csv" --k-clusters-max=10 --id-column="ID" --random-state=42
   ```

   Using k-means++ initialization:
   ```bash
   cargo run --release -- --input="data.csv" --output="results.csv" --k-clusters-max=10 --random-state=42 --init="k-means++"
   ```

## Command Line Arguments

- `--input`: Path to input CSV file (required)
- `--output`: Path to output CSV file (required)
- `--k-clusters-max`: Maximum number of clusters to compute (required)
- `--id-column`: Name of the ID column (optional, default: `"ID"`)
- `--random-state`: Seed for random number generation (optional, default: `42`)
- `--max-iterations`: Maximum iterations per k-means run (optional, default: `300`)
- `--init`: Centroid initialization method — `random` (default) or `k-means++` (optional)

### `--init` details

| Value | Behaviour |
|-------|-----------|
| `random` | Uniform random sample of k points — original behaviour, fully reproducible with `--random-state`. |
| `k-means++` | Arthur-Vassilvitskii 2007 D² weighted sampling. First centroid is chosen uniformly; each successive centroid is sampled with probability proportional to its squared distance to the nearest already-chosen centroid. Typically converges to lower inertia than random init, especially on well-separated clusters. |

Using `--init random` (the default) produces **identical results** to the pre-feature-1 binary for any given `--random-state` value.

## Features

- **Pure Rust**: No external ML libraries required
- **High Performance**: Leverages Rust's zero-cost abstractions and efficient memory management
- **Memory Safe**: Guaranteed memory safety without garbage collection overhead
- **Type Safe**: Strong typing prevents runtime errors common in dynamic languages
- **CSV Compatible**: Works with standard CSV format using the `csv` crate
- **Configurable**: Command-line interface for easy parameter adjustment
- **ID Preservation**: Maintains original row identifiers
- **Library + Binary**: Core logic lives in `lib.rs`; tests run against the library directly

## Dependencies

```toml
[dependencies]
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
clap = { version = "4.4", features = ["derive"] }
rand = "0.8"
```

## Notes

- The algorithm uses Euclidean distance for cluster assignment; D² sampling in k-means++ uses squared Euclidean (no sqrt).
- Use `--random-state` to control the seed for reproducible results.
- The `--release` flag is recommended for optimal performance on large datasets.
- Consider k-means++ (`--init k-means++`) when cluster quality matters more than raw throughput — it typically requires fewer EM iterations to converge.
