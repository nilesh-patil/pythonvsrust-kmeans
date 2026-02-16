# Feature 4 — Ground-truth quality metrics + Plotly dashboard

## Why
The existing benchmark measures **internal** quality (silhouette, Davies-Bouldin, Calinski-Harabasz) — none of which need ground truth. But `generate_data.py` already *knows* the true cluster assignments and currently throws them away. By saving them, we can compute **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** — gold-standard external metrics that directly measure "did clustering recover the truth?". Pairing this with an interactive Plotly dashboard makes the project's results explorable on the web.

## Changes

### Generator (`src/generate_data.py`)
- After writing `dataset_*.csv`, also write `dataset_*_labels.npy` (raw `np.int32` array of true cluster ids, ordered the same as the CSV rows).
- The labels file lives next to the CSV in `data/`.

### Runner (`runner.py`)
- After running each implementation, if `<dataset>_labels.npy` exists alongside the CSV, also load it and compute:
  - `adjusted_rand_index` (sklearn.metrics.adjusted_rand_score)
  - `normalized_mutual_info` (sklearn.metrics.normalized_mutual_info_score, arithmetic average method)
- Append these two columns to the benchmark CSV. Default values: `np.nan` when ground truth is unavailable (preserves backward compatibility for existing data).

### Dashboard (`src/build_dashboard.py`)
- Read the latest `results/benchmark_results_*.csv`.
- Produce `results/dashboards/index.html` — a single self-contained Plotly HTML page with:
  - **Tab 1 — Runtime vs scale**: log-log scatter, hovertext = (implementation, n_samples, n_features, n_clusters, runtime).
  - **Tab 2 — Memory footprint**: bar chart of MB/1k-samples by implementation.
  - **Tab 3 — Quality (internal)**: silhouette + Davies-Bouldin grouped bars.
  - **Tab 4 — Quality (external)**: ARI + NMI grouped bars (only present when columns are non-NaN).
- Use the `plotly.io.to_html(fig, include_plotlyjs="cdn")` API so the file is small and uses Plotly's CDN.
- Page also has a title, description paragraph, and a footer with the timestamp of the source CSV.

## Tests (write first)
`tests/test_quality_metrics.py`:
1. `test_generator_writes_labels_file`: invoke `generate_data.py` via subprocess on a tiny dataset; assert the `_labels.npy` file exists and has shape `(n_rows,)` and dtype `int`.
2. `test_runner_attaches_ari_nmi_when_labels_exist`: Construct a tiny benchmark scenario with a known labels file, run the metric helper directly, assert ARI is in [-1, 1] and NMI in [0, 1].
3. `test_runner_handles_missing_labels`: when `_labels.npy` is absent, the metric helper returns `(np.nan, np.nan)` without raising.
4. `test_dashboard_builds_html_with_all_tabs`: invoke `build_dashboard.py` against a fixture CSV; assert the output HTML contains the four tab labels.

## Acceptance criteria
- All 4 new tests green.
- `pixi run python src/generate_data.py --n_rows 100 --output data/_tmp.csv` produces both `_tmp.csv` and `_tmp_labels.npy`.
- A benchmark run with ground-truth labels populates `adjusted_rand_index` and `normalized_mutual_info` columns.
- `results/dashboards/index.html` opens in a browser and shows interactive Plotly charts.

## Out of scope
- HTML5 widget interactivity beyond what Plotly provides natively.
- Adjusted Mutual Information (AMI) — NMI suffices; AMI's reference distribution adjustment is overkill for our cluster counts.
- Hosting — that's Feature 5.
