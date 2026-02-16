#!/usr/bin/env python3
"""
Build an interactive Plotly HTML dashboard from the latest benchmark CSV.

Usage:
    python src/build_dashboard.py                        # uses latest results/*.csv
    python src/build_dashboard.py --input <csv> --output <html>
"""

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── colour palette ────────────────────────────────────────────────────────────
IMPL_COLORS: dict[str, str] = {
    "python": "#3776AB",
    "rust": "#CE422B",
    "sklearn": "#F7931E",
}


def _color(impl: str) -> str:
    return IMPL_COLORS.get(impl.lower(), "#888888")


# ── chart builders ────────────────────────────────────────────────────────────

def _runtime_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 1 — log-log scatter: runtime vs n_samples by implementation."""
    fig = go.Figure()
    for impl, grp in df.groupby("implementation"):
        grp = grp.sort_values("n_samples")
        fig.add_trace(
            go.Scatter(
                x=grp["n_samples"],
                y=grp["runtime"],
                mode="markers+lines",
                name=impl,
                marker=dict(color=_color(impl), size=8),
                line=dict(color=_color(impl)),
                customdata=grp[["n_features", "n_clusters", "runtime"]].values,
                hovertemplate=(
                    f"<b>{impl}</b><br>"
                    "n_samples=%{x}<br>"
                    "n_features=%{customdata[0]}<br>"
                    "n_clusters=%{customdata[1]}<br>"
                    "runtime=%{customdata[2]:.4f}s<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Runtime vs Scale",
        xaxis=dict(title="n_samples", type="log"),
        yaxis=dict(title="Runtime (s)", type="log"),
        legend_title="Implementation",
    )
    return fig


def _memory_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 2 — bar chart: MB per 1k samples by implementation."""
    df = df.copy()
    df["mb_per_1k"] = df["peak_memory_mb"] / (df["n_samples"] / 1_000)
    agg = df.groupby("implementation")["mb_per_1k"].mean().reset_index()

    fig = go.Figure(
        go.Bar(
            x=agg["implementation"],
            y=agg["mb_per_1k"],
            marker_color=[_color(i) for i in agg["implementation"]],
            hovertemplate="%{x}: %{y:.2f} MB/1k samples<extra></extra>",
        )
    )
    fig.update_layout(
        title="Memory Footprint (mean MB per 1k samples)",
        xaxis_title="Implementation",
        yaxis_title="MB / 1k samples",
    )
    return fig


def _internal_quality_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 3 — grouped bars: silhouette + Davies-Bouldin by implementation."""
    fig = go.Figure()
    impls = df["implementation"].unique().tolist()
    agg = df.groupby("implementation")[["silhouette_score", "davies_bouldin_index"]].mean()

    for impl in impls:
        row = agg.loc[impl] if impl in agg.index else pd.Series({"silhouette_score": 0, "davies_bouldin_index": 0})
        fig.add_trace(
            go.Bar(
                name=f"{impl} — silhouette",
                x=["Silhouette (↑)"],
                y=[row["silhouette_score"]],
                marker_color=_color(impl),
                legendgroup=impl,
                hovertemplate=f"{impl} silhouette=%{{y:.4f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{impl} — Davies-Bouldin",
                x=["Davies-Bouldin (↓)"],
                y=[row["davies_bouldin_index"]],
                marker_color=_color(impl),
                legendgroup=impl,
                showlegend=False,
                hovertemplate=f"{impl} Davies-Bouldin=%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Internal Quality Metrics (mean over all runs)",
        barmode="group",
        yaxis_title="Score",
        legend_title="Implementation",
    )
    return fig


def _external_quality_chart(df: pd.DataFrame) -> go.Figure | None:
    """Tab 4 — grouped bars: ARI + NMI by implementation. Returns None when all NaN."""
    if "adjusted_rand_index" not in df.columns or "normalized_mutual_info" not in df.columns:
        return None

    valid = df.dropna(subset=["adjusted_rand_index", "normalized_mutual_info"])
    if valid.empty:
        return None

    fig = go.Figure()
    agg = valid.groupby("implementation")[["adjusted_rand_index", "normalized_mutual_info"]].mean()

    for impl, row in agg.iterrows():
        fig.add_trace(
            go.Bar(
                name=f"{impl} — ARI",
                x=["ARI (↑)"],
                y=[row["adjusted_rand_index"]],
                marker_color=_color(str(impl)),
                legendgroup=str(impl),
                hovertemplate=f"{impl} ARI=%{{y:.4f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{impl} — NMI",
                x=["NMI (↑)"],
                y=[row["normalized_mutual_info"]],
                marker_color=_color(str(impl)),
                legendgroup=str(impl),
                showlegend=False,
                hovertemplate=f"{impl} NMI=%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="External Quality Metrics — ARI & NMI (mean over all runs)",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05]),
        legend_title="Implementation",
    )
    return fig


# ── HTML assembly ─────────────────────────────────────────────────────────────

def _fig_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def _build_html(
    df: pd.DataFrame,
    source_csv_name: str,
) -> str:
    runtime_div = _fig_div(_runtime_chart(df))
    memory_div = _fig_div(_memory_chart(df))
    internal_div = _fig_div(_internal_quality_chart(df))
    external_fig = _external_quality_chart(df)

    if external_fig is not None:
        external_content = _fig_div(external_fig)
    else:
        external_content = dedent("""
            <p style="padding:1rem;color:#555;font-style:italic;">
              Run with ground-truth labels (Feature 4) to populate external metrics.
              Re-generate datasets via <code>src/generate_data.py</code> and
              re-run the benchmark to produce <em>adjusted_rand_index</em> and
              <em>normalized_mutual_info</em> columns.
            </p>
        """).strip()

    # Radio-button CSS tabs — no JS framework required
    html = dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width,initial-scale=1"/>
          <title>K-Means Benchmark Dashboard</title>
          <style>
            body {{font-family:system-ui,sans-serif;margin:0;padding:0 1rem 2rem;background:#fafafa;color:#222;}}
            h1 {{font-size:1.6rem;margin-bottom:.25rem;}}
            .subtitle {{color:#555;margin-bottom:1.5rem;}}
            /* Tab radio inputs are hidden — only labels are visible */
            .tabs input[type=radio] {{display:none;}}
            .tabs {{margin-bottom:1rem;}}
            .tab-labels {{display:flex;gap:.5rem;flex-wrap:wrap;border-bottom:2px solid #ddd;padding-bottom:.5rem;}}
            .tab-labels label {{
              padding:.45rem 1rem;cursor:pointer;border-radius:4px 4px 0 0;
              background:#e8e8e8;border:1px solid #ccc;border-bottom:none;
              font-size:.9rem;user-select:none;
            }}
            .tab-labels label:hover {{background:#d0d8e8;}}
            .tab-content {{display:none;padding:.5rem 0;}}
            #tab1:checked ~ .tab-labels label[for=tab1],
            #tab2:checked ~ .tab-labels label[for=tab2],
            #tab3:checked ~ .tab-labels label[for=tab3],
            #tab4:checked ~ .tab-labels label[for=tab4]
            {{background:#fff;border-bottom:2px solid #fff;margin-bottom:-2px;font-weight:600;}}
            #tab1:checked ~ .tab-contents #content1,
            #tab2:checked ~ .tab-contents #content2,
            #tab3:checked ~ .tab-contents #content3,
            #tab4:checked ~ .tab-contents #content4 {{display:block;}}
            footer {{margin-top:3rem;font-size:.8rem;color:#888;border-top:1px solid #ddd;padding-top:.75rem;}}
          </style>
        </head>
        <body>
          <h1>K-Means Benchmark Dashboard</h1>
          <p class="subtitle">
            Interactive comparison of Python, Rust, and sklearn K-Means implementations
            across runtime, memory, and clustering quality dimensions.
          </p>

          <div class="tabs">
            <input type="radio" id="tab1" name="tabs" checked/>
            <input type="radio" id="tab2" name="tabs"/>
            <input type="radio" id="tab3" name="tabs"/>
            <input type="radio" id="tab4" name="tabs"/>
            <div class="tab-labels">
              <label for="tab1">Runtime vs scale</label>
              <label for="tab2">Memory footprint</label>
              <label for="tab3">Quality (internal)</label>
              <label for="tab4">Quality (external)</label>
            </div>
            <div class="tab-contents">
              <div class="tab-content" id="content1">{runtime_div}</div>
              <div class="tab-content" id="content2">{memory_div}</div>
              <div class="tab-content" id="content3">{internal_div}</div>
              <div class="tab-content" id="content4">{external_content}</div>
            </div>
          </div>

          <footer>Source CSV: {source_csv_name}</footer>
        </body>
        </html>
    """)
    return html


# ── CLI ───────────────────────────────────────────────────────────────────────

def _latest_csv(results_dir: Path) -> Path:
    """Return the most-recently modified benchmark_results_*.csv in results_dir."""
    candidates = sorted(
        results_dir.glob("benchmark_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No benchmark_results_*.csv found in {results_dir}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plotly dashboard from benchmark CSV.")
    parser.add_argument("--input", type=Path, default=None, help="Path to benchmark CSV (default: latest in results/)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: results/dashboards/index.html)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    csv_path: Path = args.input if args.input else _latest_csv(project_root / "results")
    out_path: Path = args.output if args.output else project_root / "results" / "dashboards" / "index.html"

    df = pd.read_csv(csv_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html(df, source_csv_name=csv_path.name)
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"Dashboard written to: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
