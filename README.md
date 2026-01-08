# pythonvsrust-kmeans

A comparative study of K-Means clustering implementations written in pure Python, Rust and scikit-learn.  The repository is structured so that you can generate synthetic data, run each implementation under identical scenarios, capture execution metrics and visualise the results.

---

## 📁 Directory structure

```
pythonvsrust-kmeans/
│
├── data/                # Auto-generated datasets (CSV/NPY) used in the benchmarks
├── notebooks/           # Jupyter notebooks used to explore data and analyse results
├── results/             # Raw benchmark results and summary statistics
│
├── src/                 # Source code for the experiment
│   ├── generate_data.py # Helper script to create synthetic datasets of varying size & dimensionality
│   ├── python_impl/     # Pure-Python reference implementation of K-Means
│   ├── rust_impl/       # Rust implementation (compiled to a CLI binary)
│   └── sklearn_impl/    # Thin wrapper around sklearn.cluster.KMeans for parity testing
│
├── runner.py            # Top-level orchestrator that drives data gen, runs each impl & records metrics
│
├── .gitignore           # Standard Git exclusions
└── README.md            # You are here ✅
```

> **Tip:** The folder names are intentionally short so that file paths stay readable inside result tables and plots.

---

## 🎯 Project goal

K-Means is one of the most widely-used clustering algorithms but its performance profile can vary dramatically depending on implementation language, data layout and compilation strategy.  The aim of this project is therefore **to quantify how a hand-rolled Python implementation, a high-performance Rust implementation and the industrial-strength scikit-learn implementation behave under a matrix of realistic workloads**.

The comparison focuses on three axes:

1. **Runtime** – wall-clock time spent in the `fit` phase.
2. **Memory usage** – peak resident set size (RSS) while clustering.
3. **Result quality** – inertia / within-cluster sum-of-squares to ensure all variants reach comparable minima.

By systematically sweeping through different dataset sizes (\(10^3\) – \(10^6\) samples), dimensionalities (2 – 100 features) and cluster counts (2 – 30), we hope to surface the trade-offs that practitioners can expect in production.

---

## 🛠️ Implementation strategy

1. **Synthetic data generation**  
   A single source of truth (`src/generate_data.py`) produces repeatable Gaussian blobs so that each implementation receives identical input.  Datasets are cached under `data/` using a hashed filename that encodes `n_samples`, `n_features` and `n_clusters`.

2. **Algorithm implementations**
   * **Pure Python (`src/python_impl`)** – A straightforward NumPy-based reference that mirrors the textbook algorithm with no low-level optimisations.  Serves as the baseline.
   * **Rust (`src/rust_impl`)** – Written with the `ndarray` crate and compiled to a CLI tool that accepts the dataset path and outputs centroids.  The binary is called from Python via `subprocess` so that timing/memory are captured from the shell.
   * **scikit-learn (`src/sklearn_impl`)** – A thin wrapper that delegates to `sklearn.cluster.KMeans`, providing a mature C-accelerated yardstick.

3. **Benchmark harness (`runner.py`)**
   * Parses experiment parameters (either CLI flags or `.env` defaults).
   * Ensures the necessary datasets exist, generating them on the fly if required.
   * Executes each implementation in isolation while recording runtime with Python's `time.perf_counter` and memory with `psutil.Process.memory_info`.
   * Emits a `.csv` per scenario into `results/` containing metrics for each language variant.

4. **Analysis notebooks**  
   The `notebooks/` directory contains exploratory notebooks that pivot/plot the raw results – e.g. runtime vs. sample count log-log plots, memory heatmaps, silhouette scores, etc.

5. **Continuous integration hooks** (optional)  
   A lightweight GitHub Actions workflow can be added later to run a representative subset of benchmarks on every push, preventing performance regressions.

---
