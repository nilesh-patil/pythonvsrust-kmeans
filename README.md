# pythonvsrust-kmeans

A comparative study of K-Means clustering implementations written in pure Python, serial Rust, Rayon-parallel Rust, and scikit-learn.
The repository is structured so that you can generate synthetic data, run each implementation under identical scenarios, capture execution metrics and visualise the results.

> **🌐 Live site:** A companion [GitHub Pages site](https://nilesh-patil.github.io/pythonvsrust-kmeans/) renders all results as interactive Plotly dashboards and includes a [live, in-browser WebAssembly demo](https://nilesh-patil.github.io/pythonvsrust-kmeans/demo/) of the Rust K-Means with step-by-step Lloyd's-iteration animation, six point distributions (blobs, rings, moons, anisotropic, uniform, spiral), and a WASM-vs-pure-JS speed race.

![Fresh comparative benchmark: runtime, throughput, sampled memory, and quality](./results/benchmark_plots_20260609_112255.png)

### Lloyd's iterations, animated

| random init | k-means++ init | pathological random seed | two moons (k-means fails) | concentric rings (k-means fails) |
|---|---|---|---|---|
| ![random](./results/animations/convergence_random.gif) | ![k-means++](./results/animations/convergence_kpp.gif) | ![pathological](./results/animations/convergence_pathological.gif) | ![moons](./results/animations/convergence_moons.gif) | ![circles](./results/animations/convergence_circles.gif) |

Random init converges slowly when two centroids start in the same blob (third panel). k-means++ usually picks one centroid per blob on the first try (second panel), converging in 2–3 iterations. The right-hand panels show k-means' two classic failure modes — moons get bisected and rings get pie-sliced because k-means imposes Voronoi (convex-polytope) cluster boundaries.

### k-means++ vs random initialization

![init comparison](./results/init_comparison.png)

### Parallel Rust K-Means (Rayon)

![parallel scaling](./results/parallel_scaling.png)

The Rust-only parallel sweep follows the same log2 sample progression as the main suite, from 1k through 256k rows at 32 features and `k_max=32`. The current data peaks at 1.32x over serial Rust on the 256k-row slice; the full sweep is saved as `results/parallel_scaling_n*.csv`, with `results/parallel_scaling.csv` kept as the 32k compatibility slice.

---

## 📁 Directory structure

```
pythonvsrust-kmeans/
│
├── data/                # Auto-generated datasets (CSV/NPY) used in the benchmarks
├── notebooks/           # Jupyter notebooks used to explore data and analyse results
├── results/             # Benchmark CSVs, PNGs, GIFs, dashboard HTML
│   ├── animations/      # Lloyd's-iteration GIFs
│   └── dashboards/      # Interactive Plotly HTML
│
├── docs/                # Jekyll GitHub Pages site (served from here)
│   ├── _config.yml      # Site config (theme=minima, baseurl)
│   ├── _layouts/        # Page layouts
│   ├── assets/          # Mirrored PNGs, GIFs, CSS
│   ├── wasm/            # Compiled Rust → WebAssembly module + JS glue
│   └── *.md             # Site pages (index, algorithms, parallel, …)
│
├── specs/               # Spec-Driven Development docs, one per feature
├── tests/               # pytest suite (init, animations, metrics, site)
│
├── src/                 # Source code for the experiment
│   ├── generate_data.py            # Synthetic datasets + ground-truth labels (.npy)
│   ├── animate_convergence.py      # Lloyd's-iteration GIF generator
│   ├── visualize_init_comparison.py
│   ├── visualize_parallel_scaling.py
│   ├── bench_parallel_scaling.py   # Rust thread-count sweep
│   ├── build_dashboard.py          # Interactive Plotly dashboard
│   ├── sync_assets.py              # Mirror results/ into docs/assets/
│   ├── python_impl/     # Pure-Python K-Means (random + k-means++)
│   ├── rust_impl/       # Rust CLI binary + lib (serial + Rayon-parallel)
│   ├── sklearn_impl/    # Thin sklearn wrapper
│   └── wasm_impl/       # WebAssembly build of Rust K-Means
│
├── runner.py            # Orchestrator: data gen + run each impl + capture metrics + ARI/NMI
│
├── pixi.toml           # Pixi dependency management configuration
├── pixi.lock           # Lock file for reproducible environments
├── .gitignore          # Standard Git exclusions
└── README.md           # You are here ✅
```

---

## 🚀 Getting Started

### Prerequisites

This project uses [Pixi](https://pixi.sh/) for dependency management, which provides a reproducible environment across different machines.

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/nilesh-patil/pythonvsrust-kmeans.git
   cd pythonvsrust-kmeans
   ```

3. **Install dependencies**:
   ```bash
   pixi install
   ```

### Running the benchmarks

1. **Generate synthetic datasets**:
   ```bash
   pixi run python src/generate_data.py
   ```

2. **Run the full benchmark suite**:
   ```bash
   pixi run python runner.py
   ```

3. **View results**:
   Results are saved in the `results/` directory as CSV files and visualization plots.

> **Four rows per matched workload.** Each benchmark cell produces result rows for
> `python`, `sklearn`, `rust` (serial), and `rust_parallel` (Rayon). The runner can
> repeat each cell with paired dataset/algorithm seeds, records wall time, CPU time,
> effective cores, sampled RSS, normalized throughput, ARI/NMI, and paired speedups,
> then feeds those metrics into the dashboard and static figures. Implementation
> colors and symbols are centralized in `src/viz_style.py`.

### Using Pixi Tasks (Recommended)

The project includes pre-defined pixi tasks that streamline common workflows:

#### Quick Start
```bash
# Run everything with one command (build Rust, generate data, run benchmarks)
pixi run run-all

# Or for a quick test with smaller datasets
pixi run run-quick
```

#### Data Generation
```bash
pixi run generate-data      # Generate all default datasets
pixi run generate-small     # Small dataset (1K samples)
pixi run generate-medium    # Medium dataset (10K samples)
pixi run generate-large     # Large dataset (100K samples)
```

#### Building
```bash
pixi run build-rust         # Build Rust implementation (release mode)
pixi run build-rust-debug   # Build Rust implementation (debug mode)
```

#### Testing Individual Implementations
```bash
pixi run test-python        # Test Python implementation
pixi run test-rust          # Test Rust implementation
pixi run test-sklearn       # Test scikit-learn implementation
```

#### Benchmarking
```bash
pixi run benchmark          # Run default benchmark suite
pixi run benchmark-quick    # Quick benchmark with fewer parameters
pixi run benchmark-full     # Comprehensive benchmark (takes longer)
```

#### Utilities
```bash
pixi run clean-data         # Remove generated datasets
pixi run clean-results      # Remove benchmark results
pixi run clean-all          # Clean everything
```

#### Help
```bash
pixi run help               # List available pixi tasks
```

---

## 🎯 Project goal

K-Means is one of the most widely-used clustering algorithms but its performance profile can vary dramatically depending on implementation language, data layout and compilation strategy.  The aim of this project is therefore **to quantify how hand-rolled Python, serial Rust, Rayon-parallel Rust, and industrial-strength scikit-learn implementations behave under a matrix of realistic workloads**.

The comparison focuses on three axes:

1. **Runtime** - end-to-end CLI subprocess wall time, including CSV loading, fitting every `k` from 1 to `k_max`, output CSV writing and process overhead.
2. **Memory usage** - sampled process resident set size (RSS) while the CLI runs, polled every 10 ms. This is not a platform max-RSS measurement.
3. **Result quality** - inertia, internal clustering metrics, and ground-truth ARI/NMI when generated labels are available.

The current website refresh uses a paired log2 sample sequence from 1k through 256k rows. Every sample size covers feature counts 2, 8, and 32 with k_max values 8 and 32 over three paired repeats.

---

## 🛠️ Implementation strategy

1. **Synthetic data generation**  
   A single source of truth (`src/generate_data.py`) produces repeatable Gaussian blobs so that each implementation receives identical input. Datasets are cached under `data/` using a hashed filename that encodes sample count, feature count, cluster count, seed, cluster standard deviation, cluster separation, and a short hash.

2. **Algorithm implementations**
   * **Pure Python (`src/python_impl/kmeans.py`)** – A straightforward NumPy-based reference that mirrors the textbook algorithm with no low-level optimisations. Serves as the baseline. Can be run as a CLI tool.
   * **Rust (`src/rust_impl`)** – Serial and Rayon-parallel CLI paths that accept the dataset path, output path, and max cluster count. The binary is called from Python via `subprocess` so that timing/memory are captured from the shell. Build with `cargo build --release` in the `src/rust_impl` directory.
   * **scikit-learn (`src/sklearn_impl/kmeans.py`)** – A thin wrapper that delegates to `sklearn.cluster.KMeans`, providing a mature C-accelerated yardstick. Also executable as a CLI tool.

3. **Benchmark harness (`runner.py`)**
   * Parses experiment parameters (either CLI flags or pre-defined defaults).
   * Ensures the necessary datasets exist, generating them on the fly if required.
   * Executes each implementation in isolation while recording subprocess runtime with Python's `time.perf_counter` and sampled RSS with `psutil.Process.memory_info`.
   * Emits one suite-level `benchmark_results_<timestamp>.csv` into `results/`, with one row per implementation/workload/repeat plus paired resource and quality metrics.
   * Writes per-implementation output CSVs while each subprocess runs, then uses the suite-level metrics file for the dashboard and static visualizations.

4. **Analysis notebooks** [to-do] 
   The `notebooks/` directory can be used for exploratory pivots of the raw results, including nominal-workload runtime curves, log2 resource scaling, throughput comparisons, sampled-RSS views, and quality/runtime frontiers.

---

## 📦 Dependencies

The project uses Pixi for dependency management, which handles both Python and Rust toolchains. Key dependencies include:

- **Python 3.11**: Core language runtime
- **NumPy & Pandas**: Data manipulation and numerical computing
- **scikit-learn**: Reference K-Means implementation
- **Matplotlib & Seaborn**: Visualization
- **Rust toolchain**: For compiling the Rust implementation
- **psutil**: Process monitoring for memory usage tracking

All dependencies are specified in `pixi.toml` and locked in `pixi.lock` for reproducibility.

---
