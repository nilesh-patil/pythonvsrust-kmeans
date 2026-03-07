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
from plotly.offline import get_plotlyjs

try:
    from viz_style import (
        DISPLAY_NAMES,
        IMPL_COLORS,
        INK,
        INK_FAINT,
        PAPER,
        PLOTLY_FONT_FAMILY,
        RULE,
        color,
        display_name,
        implementation_from_display,
        log2_tick_values,
        ordered_implementations,
        plotly_dash,
        plotly_pattern,
        plotly_symbol,
        si_labels,
    )
except ImportError:
    from src.viz_style import (
        DISPLAY_NAMES,
        IMPL_COLORS,
        INK,
        INK_FAINT,
        PAPER,
        PLOTLY_FONT_FAMILY,
        RULE,
        color,
        display_name,
        implementation_from_display,
        log2_tick_values,
        ordered_implementations,
        plotly_dash,
        plotly_pattern,
        plotly_symbol,
        si_labels,
    )


def _color(impl: str) -> str:
    return color(impl)


def _display_name(impl: str) -> str:
    return display_name(impl)


def _implementation_from_display(display: str) -> str:
    return implementation_from_display(display)


def _with_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a benchmark frame with derived workload/resource columns."""
    out = df.copy()
    if "wall_time_s" not in out.columns and "runtime" in out.columns:
        out["wall_time_s"] = out["runtime"]
    if "peak_rss_mb" not in out.columns and "peak_memory_mb" in out.columns:
        out["peak_rss_mb"] = out["peak_memory_mb"]
    if "k_max" not in out.columns:
        out["k_max"] = out["n_clusters"]
    if "cpu_time_s" not in out.columns:
        out["cpu_time_s"] = np.nan
    if "effective_cores" not in out.columns:
        out["effective_cores"] = np.nan

    k_max = out["k_max"].clip(lower=1)
    wall = out["wall_time_s"].replace(0, np.nan)
    cpu = out["cpu_time_s"].replace(0, np.nan)
    out["input_values"] = out["n_samples"] * out["n_features"]
    out["k_sweep_sum_k"] = k_max * (k_max + 1) / 2
    out["nominal_work_units"] = out["input_values"] * out["k_sweep_sum_k"]
    out["samples_per_second"] = out["n_samples"] / wall
    out["work_units_per_second"] = out["nominal_work_units"] / wall
    out["rss_mb_per_1k_samples"] = out["peak_rss_mb"] * 1000 / out["n_samples"]
    out["cpu_seconds_per_1k_samples"] = cpu * 1000 / out["n_samples"]
    return out


def _median_by_workload(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["implementation", "nominal_work_units", "n_samples", "n_features", "n_clusters"]
    optional = [key for key in ("k_max", "init", "sklearn_n_init") if key in df.columns]
    return (
        df.groupby(keys + optional, dropna=False)[metric]
        .median()
        .reset_index()
        .sort_values(["implementation", "nominal_work_units"])
    )


def _trend_by_workload(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median trend by nominal workload, collapsing duplicate workload shapes."""
    return (
        df.groupby(["implementation", "nominal_work_units"], dropna=False)
        .agg(
            value=(metric, "median"),
            workload_shapes=("nominal_work_units", "size"),
        )
        .reset_index()
        .sort_values(["implementation", "nominal_work_units"])
    )


def _apply_tufte_plotly_style(fig: go.Figure) -> go.Figure:
    """Quiet Plotly defaults so data marks carry the visual weight.

    Paper off-white background, serif (Newsreader/Georgia) type, hairline
    left+bottom axes only, and an ultra-light y-grid — matching the matplotlib
    figure family.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=PLOTLY_FONT_FAMILY, color=INK, size=14),
        title=dict(font=dict(size=16, color=INK)),
        hoverlabel=dict(
            bgcolor=PAPER,
            bordercolor=RULE,
            font=dict(family=PLOTLY_FONT_FAMILY, size=12, color=INK),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(family=PLOTLY_FONT_FAMILY, size=12),
        ),
        margin=dict(l=72, r=120, t=72, b=64),
    )
    # x-axis: hairline baseline, no vertical grid.
    fig.update_xaxes(
        showline=True,
        linecolor="#999999",
        linewidth=1,
        mirror=False,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickcolor="#bbbbbb",
        ticklen=4,
        tickfont=dict(color=INK_FAINT),
    )
    # y-axis: hairline spine, ultra-light dashed grid behind the data.
    fig.update_yaxes(
        showline=True,
        linecolor="#999999",
        linewidth=1,
        mirror=False,
        showgrid=True,
        gridcolor="#eeeeee",
        gridwidth=1,
        griddash="dot",
        zeroline=False,
        ticks="outside",
        tickcolor="#bbbbbb",
        ticklen=4,
        tickfont=dict(color=INK_FAINT),
    )
    return fig


# ── chart builders ────────────────────────────────────────────────────────────

def _runtime_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 1 — CLI runtime vs nominal k-sweep workload."""
    df = _with_analysis_columns(df)
    agg = _median_by_workload(df, "wall_time_s")
    trend = _trend_by_workload(agg, "wall_time_s")
    fig = go.Figure()
    for impl in ordered_implementations(trend["implementation"].unique()):
        grp = trend[trend["implementation"] == impl]
        display = _display_name(impl)
        fig.add_trace(
            go.Scatter(
                x=grp["nominal_work_units"],
                y=grp["value"],
                mode="lines+markers",
                name=display,
                line=dict(color=_color(impl), width=2, dash=plotly_dash(impl)),
                marker=dict(color=_color(impl), symbol=plotly_symbol(impl), size=9),
                customdata=grp[["workload_shapes"]].values,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "work=%{x:.3g}<br>"
                    "median runtime=%{y:.4f}s<br>"
                    "matched workload shapes=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    runtime_ticks = log2_tick_values(trend["value"])
    work_ticks = log2_tick_values(trend["nominal_work_units"])
    fig.update_layout(
        title="CLI k-sweep runtime vs nominal workload",
        xaxis=dict(
            title="n_samples x n_features x sum(k=1..k_max)",
            type="log",
            tickvals=work_ticks,
            ticktext=si_labels(work_ticks),
        ),
        yaxis=dict(
            title="Runtime (s, including CSV read/write)",
            type="log",
            tickvals=runtime_ticks,
            ticktext=si_labels(runtime_ticks),
        ),
        legend_title="Implementation",
    )
    _apply_tufte_plotly_style(fig)
    return fig


def _throughput_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 2 — nominal work throughput vs workload."""
    df = _with_analysis_columns(df)
    agg = _median_by_workload(df, "work_units_per_second")
    trend = _trend_by_workload(agg, "work_units_per_second")
    fig = go.Figure()
    for impl in ordered_implementations(trend["implementation"].unique()):
        grp = trend[trend["implementation"] == impl]
        display = _display_name(impl)
        fig.add_trace(
            go.Scatter(
                x=grp["nominal_work_units"],
                y=grp["value"],
                mode="lines+markers",
                name=display,
                line=dict(color=_color(impl), width=2, dash=plotly_dash(impl)),
                marker=dict(color=_color(impl), symbol=plotly_symbol(impl), size=9),
                customdata=grp[["workload_shapes"]].values,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "work=%{x:.3g}<br>"
                    "throughput=%{y:.3g} work units/s<br>"
                    "matched workload shapes=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    throughput_ticks = log2_tick_values(trend["value"])
    work_ticks = log2_tick_values(trend["nominal_work_units"])
    fig.update_layout(
        title="Nominal k-sweep throughput",
        xaxis=dict(
            title="n_samples x n_features x sum(k=1..k_max)",
            type="log",
            tickvals=work_ticks,
            ticktext=si_labels(work_ticks),
        ),
        yaxis=dict(
            title="Nominal work units / s",
            type="log",
            tickvals=throughput_ticks,
            ticktext=si_labels(throughput_ticks),
        ),
        legend_title="Implementation",
    )
    _apply_tufte_plotly_style(fig)
    return fig


def _memory_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 3 — sampled RSS vs nominal k-sweep workload."""
    df = _with_analysis_columns(df)
    agg = _median_by_workload(df, "peak_rss_mb")
    trend = _trend_by_workload(agg, "peak_rss_mb")
    fig = go.Figure()
    for impl in ordered_implementations(trend["implementation"].unique()):
        grp = trend[trend["implementation"] == impl]
        display = _display_name(impl)
        fig.add_trace(
            go.Scatter(
                x=grp["nominal_work_units"],
                y=grp["value"],
                mode="lines+markers",
                name=display,
                line=dict(color=_color(impl), width=2, dash=plotly_dash(impl)),
                marker=dict(color=_color(impl), symbol=plotly_symbol(impl), size=9),
                customdata=grp[["workload_shapes"]].values,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "work=%{x:.3g}<br>"
                    "median sampled RSS=%{y:.2f} MB<br>"
                    "matched workload shapes=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    memory_ticks = log2_tick_values(trend["value"])
    work_ticks = log2_tick_values(trend["nominal_work_units"])
    fig.update_layout(
        title="Memory footprint vs nominal workload",
        xaxis=dict(
            title="n_samples x n_features x sum(k=1..k_max)",
            type="log",
            tickvals=work_ticks,
            ticktext=si_labels(work_ticks),
        ),
        yaxis=dict(
            title="Sampled RSS (MB)",
            type="log",
            tickvals=memory_ticks,
            ticktext=si_labels(memory_ticks),
        ),
        legend_title="Implementation",
    )
    _apply_tufte_plotly_style(fig)
    return fig


def _internal_quality_chart(df: pd.DataFrame) -> go.Figure:
    """Tab 3 — grouped bars: silhouette + Davies-Bouldin by implementation."""
    fig = go.Figure()
    impls = df["implementation"].unique().tolist()
    agg = df.groupby("implementation")[["silhouette_score", "davies_bouldin_index"]].mean()

    for impl in impls:
        row = agg.loc[impl] if impl in agg.index else pd.Series({"silhouette_score": 0, "davies_bouldin_index": 0})
        display = _display_name(impl)
        fig.add_trace(
            go.Bar(
                name=f"{display} — silhouette",
                x=["Silhouette (↑)"],
                y=[row["silhouette_score"]],
                marker=dict(color=_color(impl), pattern=dict(shape=plotly_pattern(impl))),
                legendgroup=display,
                hovertemplate=f"{display} silhouette=%{{y:.4f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{display} — Davies-Bouldin",
                x=["Davies-Bouldin (↓)"],
                y=[row["davies_bouldin_index"]],
                marker=dict(color=_color(impl), pattern=dict(shape=plotly_pattern(impl))),
                legendgroup=display,
                showlegend=False,
                hovertemplate=f"{display} Davies-Bouldin=%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Internal quality metrics (mean over all runs)",
        barmode="group",
        yaxis_title="Score",
        legend_title="Implementation",
    )
    _apply_tufte_plotly_style(fig)
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
        display = _display_name(str(impl))
        fig.add_trace(
            go.Bar(
                name=f"{display} — ARI",
                x=["ARI (↑)"],
                y=[row["adjusted_rand_index"]],
                marker=dict(color=_color(str(impl)), pattern=dict(shape=plotly_pattern(str(impl)))),
                legendgroup=display,
                hovertemplate=f"{display} ARI=%{{y:.4f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{display} — NMI",
                x=["NMI (↑)"],
                y=[row["normalized_mutual_info"]],
                marker=dict(color=_color(str(impl)), pattern=dict(shape=plotly_pattern(str(impl)))),
                legendgroup=display,
                showlegend=False,
                hovertemplate=f"{display} NMI=%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="External quality metrics — ARI and NMI (mean over all runs)",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05]),
        legend_title="Implementation",
    )
    _apply_tufte_plotly_style(fig)
    return fig


def _quality_frontier_chart(df: pd.DataFrame) -> go.Figure | None:
    """Quality/runtime frontier: external ARI if available, otherwise silhouette."""
    df = _with_analysis_columns(df)
    metric = None
    metric_label = None
    y_range = None
    if "adjusted_rand_index" in df.columns and df["adjusted_rand_index"].notna().any():
        metric = "adjusted_rand_index"
        metric_label = "Adjusted Rand Index (higher is better)"
        y_range = [0, 1.05]
    elif "silhouette_score" in df.columns and df["silhouette_score"].notna().any():
        metric = "silhouette_score"
        metric_label = "Silhouette score (higher is better)"
        y_range = [0, 1.05]
    else:
        return None

    keys = ["implementation", "n_samples", "n_features", "n_clusters", "nominal_work_units"]
    agg = (
        df.groupby(keys, dropna=False)[["wall_time_s", metric]]
        .median()
        .reset_index()
        .sort_values(["implementation", "nominal_work_units"])
    )

    fig = go.Figure()
    for impl in ordered_implementations(agg["implementation"].unique()):
        grp = agg[agg["implementation"] == impl]
        display = _display_name(impl)
        fig.add_trace(
            go.Scatter(
                x=grp["wall_time_s"],
                y=grp[metric],
                mode="markers",
                name=display,
                marker=dict(color=_color(impl), symbol=plotly_symbol(impl), size=10),
                customdata=grp[["n_samples", "n_features", "n_clusters"]].values,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "runtime=%{x:.4f}s<br>"
                    f"{metric_label}=%{{y:.4f}}<br>"
                    "n_samples=%{customdata[0]}<br>"
                    "n_features=%{customdata[1]}<br>"
                    "k_max=%{customdata[2]}<extra></extra>"
                ),
            )
        )
    runtime_ticks = log2_tick_values(agg["wall_time_s"])
    fig.update_layout(
        title="Quality vs runtime frontier",
        xaxis=dict(
            title="Median runtime (s)",
            type="log",
            tickvals=runtime_ticks,
            ticktext=si_labels(runtime_ticks),
        ),
        yaxis=dict(title=metric_label, range=y_range),
        legend_title="Implementation",
    )
    _apply_tufte_plotly_style(fig)
    return fig


def _resource_table(df: pd.DataFrame) -> str:
    """Compact Tufte-style resource table with medians and IQRs."""
    df = _with_analysis_columns(df)
    agg = (
        df.groupby("implementation", dropna=False)
        .agg(
            median_runtime_s=("wall_time_s", "median"),
            p25_runtime_s=("wall_time_s", lambda s: s.quantile(0.25)),
            p75_runtime_s=("wall_time_s", lambda s: s.quantile(0.75)),
            median_rss_mb=("peak_rss_mb", "median"),
            median_rss_per_1k=("rss_mb_per_1k_samples", "median"),
            median_cpu_s_per_1k=("cpu_seconds_per_1k_samples", "median"),
            median_effective_cores=("effective_cores", "median"),
            median_throughput=("work_units_per_second", "median"),
        )
        .reset_index()
        .sort_values("median_runtime_s")
    )
    if "adjusted_rand_index" in df.columns:
        quality = df.groupby("implementation")["adjusted_rand_index"].median()
        agg["median_ari"] = agg["implementation"].map(quality)

    def fmt(value: float, precision: int = 3) -> str:
        if pd.isna(value):
            return "n/a"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{precision}f}"

    rows = []
    for _, row in agg.iterrows():
        impl = str(row["implementation"])
        iqr = row["p75_runtime_s"] - row["p25_runtime_s"]
        rows.append(
            "<tr>"
            f"<th><span style=\"color:{_color(impl)}\">{_display_name(impl)}</span></th>"
            f"<td>{fmt(row['median_runtime_s'])}</td>"
            f"<td>{fmt(iqr)}</td>"
            f"<td>{fmt(row['median_rss_mb'], 2)}</td>"
            f"<td>{fmt(row['median_rss_per_1k'], 2)}</td>"
            f"<td>{fmt(row['median_cpu_s_per_1k'])}</td>"
            f"<td>{fmt(row['median_effective_cores'], 2)}</td>"
            f"<td>{fmt(row['median_throughput'])}</td>"
            f"<td>{fmt(row.get('median_ari', np.nan), 3)}</td>"
            "</tr>"
        )

    return dedent(f"""
        <div class="resource-table-wrap">
          <table class="resource-table">
            <thead>
              <tr>
                <th>Implementation</th>
                <th>Runtime median s</th>
                <th>Runtime IQR s</th>
                <th>RSS median MB</th>
                <th>RSS MB / 1k</th>
                <th>CPU s / 1k</th>
                <th>Effective cores</th>
                <th>Work units / s</th>
                <th>ARI median</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
    """).strip()


# ── HTML assembly ─────────────────────────────────────────────────────────────

def _fig_div(fig: go.Figure) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True},
        default_height="520px",
    )


def _build_html(
    df: pd.DataFrame,
    source_csv_name: str,
) -> str:
    runtime_div = _fig_div(_runtime_chart(df))
    throughput_div = _fig_div(_throughput_chart(df))
    memory_div = _fig_div(_memory_chart(df))
    resource_table = _resource_table(df)
    quality_fig = _quality_frontier_chart(df)

    if quality_fig is not None:
        quality_content = _fig_div(quality_fig)
    else:
        quality_content = dedent("""
            <p style="padding:1rem;color:#555;font-style:italic;">
              Run with ground-truth labels to populate quality metrics.
              Re-generate datasets via <code>src/generate_data.py</code> and
              re-run the benchmark to produce ARI/NMI columns.
            </p>
        """).strip()

    plotly_js = get_plotlyjs()

    # Radio-button CSS tabs — no JS framework required
    html = dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width,initial-scale=1"/>
          <title>K-Means Benchmark Dashboard</title>
          <link rel="icon" href="../assets/favicon.svg" type="image/svg+xml"/>
          <link rel="preconnect" href="https://fonts.googleapis.com"/>
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
          <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet"/>
          <style>
            body {{
              font-family:Newsreader,Georgia,'Times New Roman',serif;
              margin:0 auto;padding:1.5rem 1rem 2rem;max-width:960px;
              background:#fffff8;color:#151515;
              line-height:1.5;
            }}
            h1 {{font-size:1.6rem;font-weight:600;margin-bottom:.25rem;}}
            .subtitle {{color:#595959;margin-bottom:1.5rem;max-width:64ch;}}
            /* Embedded in the benchmarks page: the page supplies heading and
               context, so drop the duplicate chrome and let the tabs blend in. */
            .embedded body {{padding:0;max-width:none;}}
            .embedded h1, .embedded .subtitle, .embedded footer {{display:none;}}
            /* Tab radio inputs are hidden — only labels are visible */
            .tabs input[type=radio] {{display:none;}}
            .tabs {{margin-bottom:1rem;}}
            .tab-labels {{display:flex;gap:.5rem;flex-wrap:wrap;border-bottom:1px solid #dcdcd4;padding-bottom:.5rem;}}
            .tab-labels label {{
              padding:.45rem 1rem;cursor:pointer;border-radius:4px 4px 0 0;
              background:transparent;border:1px solid transparent;border-bottom:none;
              font-size:.95rem;font-style:italic;color:#595959;user-select:none;
            }}
            .tab-labels label:hover {{color:#151515;}}
            .tab-content {{display:none;padding:.5rem 0;}}
            #tab1:checked ~ .tab-labels label[for=tab1],
            #tab2:checked ~ .tab-labels label[for=tab2],
            #tab3:checked ~ .tab-labels label[for=tab3],
            #tab4:checked ~ .tab-labels label[for=tab4],
            #tab5:checked ~ .tab-labels label[for=tab5]
            {{color:#151515;font-style:normal;border-bottom:2px solid #a32015;margin-bottom:-1px;}}
            #tab1:checked ~ .tab-contents #content1,
            #tab2:checked ~ .tab-contents #content2,
            #tab3:checked ~ .tab-contents #content3,
            #tab4:checked ~ .tab-contents #content4,
            #tab5:checked ~ .tab-contents #content5 {{display:block;}}
            .resource-table-wrap {{overflow-x:auto;margin:1rem 0;}}
            .resource-table {{border-collapse:collapse;min-width:860px;background:transparent;}}
            .resource-table th,.resource-table td {{
              border:none;padding:.45rem .6rem;text-align:right;
              font-variant-numeric:tabular-nums;
            }}
            .resource-table th:first-child,.resource-table td:first-child {{text-align:left;}}
            .resource-table thead th {{
              color:#595959;font-weight:600;font-style:italic;
              border-bottom:1px solid #151515;
            }}
            .resource-table tbody tr:last-child td,
            .resource-table tbody tr:last-child th {{border-bottom:1px solid #151515;}}
            footer {{margin-top:3rem;font-size:.85rem;font-style:italic;color:#595959;border-top:1px solid #dcdcd4;padding-top:.75rem;}}
          </style>
          <script>{plotly_js}</script>
        </head>
        <body>
          <h1>K-Means Benchmark Dashboard</h1>
          <p class="subtitle">
            Interactive comparison of Python, Rust, Rust-Parallel, and scikit-learn K-Means execution paths
            across CLI k-sweep runtime, throughput, sampled RSS, CPU/resource use, and clustering quality.
            Runtime is measured end-to-end, including CSV read/write and fitting k = 1..k_max.
          </p>

          <div class="tabs">
            <input type="radio" id="tab1" name="tabs" checked/>
            <input type="radio" id="tab2" name="tabs"/>
            <input type="radio" id="tab3" name="tabs"/>
            <input type="radio" id="tab4" name="tabs"/>
            <input type="radio" id="tab5" name="tabs"/>
            <div class="tab-labels">
              <label for="tab1">Runtime vs workload</label>
              <label for="tab2">Throughput</label>
              <label for="tab3">Memory footprint</label>
              <label for="tab4">Resource table</label>
              <label for="tab5">Quality frontier</label>
            </div>
            <div class="tab-contents">
              <div class="tab-content" id="content1">{runtime_div}</div>
              <div class="tab-content" id="content2">{throughput_div}</div>
              <div class="tab-content" id="content3">{memory_div}</div>
              <div class="tab-content" id="content4">{resource_table}</div>
              <div class="tab-content" id="content5">{quality_content}</div>
            </div>
          </div>

          <!-- source CSV: {source_csv_name} -->
          <script>
            if (window.self !== window.top) {{
              document.documentElement.classList.add("embedded");
            }}
            // Plotly divs are built with responsive:true, but a chart laid out
            // while its tab is display:none picks up a stale/default width and
            // only re-measures on a window resize. When any tab is selected,
            // wait one frame so the container is visible, then nudge Plotly to
            // re-measure by dispatching a resize event.
            document.querySelectorAll('.tabs input[type=radio]').forEach(function (radio) {{
              radio.addEventListener('change', function () {{
                requestAnimationFrame(function () {{
                  setTimeout(function () {{
                    window.dispatchEvent(new Event('resize'));
                  }}, 50);
                }});
              }});
            }});
          </script>
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
