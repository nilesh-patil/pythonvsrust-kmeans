# Data Generation Script

## Overview

`generate_data.py` creates synthetic datasets with well-separated clusters using Gaussian distributions. Each cluster is generated with:
- Distinct cluster centers that maintain minimum separation
- Configurable standard deviation for cluster tightness
- No overlap between clusters

## Features

- Generates N rows with M features and K distinct clusters
- Each row has a unique ID column
- Clusters are created using Gaussian distributions with random means
- Automatic validation to ensure clusters don't overlap
- Simple text-based visualization for small 2D datasets
- Reproducible results with random seed support

## Usage

### Basic Usage
```bash
python src/generate_data.py
```
This will generate a dataset with default parameters (1000 rows, 5 features, 3 clusters) and save it to `data/input.csv`.

### Custom Parameters
```bash
python src/generate_data.py --n_rows 5000 --n_features 10 --n_clusters 7 --output data/custom_data.csv
```

### All Parameters
- `--n_rows`: Number of data points to generate (default: 1000)
- `--n_features`: Number of feature columns (default: 5)
- `--n_clusters`: Number of distinct clusters (default: 3)
- `--output`: Output file path (default: data/input.csv)
- `--cluster_separation`: Minimum distance between cluster centers (default: 3.0)
- `--cluster_std`: Standard deviation for each cluster (default: 0.5)
- `--random_state`: Random seed for reproducibility (default: 42)

### Examples

1. **Small test dataset with visualization**:
   ```bash
   python src/generate_data.py --n_rows 200 --n_features 2 --n_clusters 3
   ```

2. **Large dataset for performance testing**:
   ```bash
   python src/generate_data.py --n_rows 100000 --n_features 20 --n_clusters 10
   ```

3. **Tighter clusters with more separation**:
   ```bash
   python src/generate_data.py --cluster_std 0.3 --cluster_separation 5.0
   ```

## Output Format

The generated CSV file contains:
- `ID`: Unique identifier for each row (1 to N)
- `feature_1`, `feature_2`, ..., `feature_M`: Feature columns with floating-point values

Example output structure:
```csv
ID,feature_1,feature_2,feature_3,feature_4,feature_5
1,2.345,-1.234,0.567,3.890,-0.123
2,2.456,-1.345,0.678,3.901,-0.234
...
```

## Implementation Details

- Uses NumPy for efficient numerical operations
- Pandas for CSV file handling
- Smart cluster center placement algorithm to ensure minimum separation
- Data points are shuffled after generation to mix clusters randomly 