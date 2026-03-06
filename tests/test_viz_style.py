"""Tests for implementation-level visual encodings."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_each_implementation_has_distinct_color_and_symbol():
    from viz_style import IMPL_COLORS, IMPL_SYMBOLS_PLOTLY, IMPLEMENTATION_ORDER, log2_tick_text

    colors = [IMPL_COLORS[impl] for impl in IMPLEMENTATION_ORDER]
    symbols = [IMPL_SYMBOLS_PLOTLY[impl] for impl in IMPLEMENTATION_ORDER]

    assert len(colors) == len(set(colors))
    assert len(symbols) == len(set(symbols))
    assert log2_tick_text([0.25, 1.0, 1024.0]) == ["2^-2", "2^0", "2^10"]


def test_dashboard_runtime_and_memory_use_symbols_and_log2_axes():
    import build_dashboard
    from viz_style import IMPL_SYMBOLS_PLOTLY

    df = pd.DataFrame(
        [
            {
                "implementation": impl,
                "n_samples": n,
                "n_features": 2,
                "n_clusters": 8,
                "runtime": runtime,
                "peak_memory_mb": memory,
                "silhouette_score": 0.5,
                "davies_bouldin_index": 1.0,
            }
            for impl, runtime, memory in [
                ("python", 0.50, 80.0),
                ("rust", 0.03125, 1.0),
                ("rust_parallel", 0.0625, 4.0),
                ("sklearn", 2.0, 256.0),
            ]
            for n in (1000, 2000)
        ]
    )

    runtime_fig = build_dashboard._runtime_chart(df)
    throughput_fig = build_dashboard._throughput_chart(df)
    memory_fig = build_dashboard._memory_chart(df)

    assert runtime_fig.layout.yaxis.type == "log"
    assert "log2" in runtime_fig.layout.yaxis.title.text
    assert all(str(text).startswith("2^") for text in runtime_fig.layout.yaxis.ticktext)
    assert runtime_fig.layout.plot_bgcolor == "white"
    assert memory_fig.layout.yaxis.type == "log"
    assert "log2" in memory_fig.layout.yaxis.title.text
    assert memory_fig.layout.plot_bgcolor == "white"

    for trace in runtime_fig.data:
        impl = build_dashboard._implementation_from_display(trace.name)
        assert trace.mode == "lines+markers"
        assert trace.marker.symbol == IMPL_SYMBOLS_PLOTLY[impl]

    for trace in throughput_fig.data:
        assert trace.mode == "lines+markers"

    for trace in memory_fig.data:
        impl = build_dashboard._implementation_from_display(trace.name)
        assert trace.mode == "lines+markers"
        assert trace.marker.symbol == IMPL_SYMBOLS_PLOTLY[impl]


def test_parallel_scaling_plot_uses_shared_rust_styles_and_log2_runtime(tmp_path, monkeypatch):
    import visualize_parallel_scaling
    from viz_style import color, mpl_marker

    csv_path = tmp_path / "parallel_scaling.csv"
    out_path = tmp_path / "parallel_scaling.png"
    pd.DataFrame(
        [
            {"threads": 0, "median_s": 8.0, "min_s": 7.8, "max_s": 8.2},
            {"threads": 1, "median_s": 8.5, "min_s": 8.2, "max_s": 8.8},
            {"threads": 2, "median_s": 4.5, "min_s": 4.3, "max_s": 4.7},
            {"threads": 4, "median_s": 2.5, "min_s": 2.4, "max_s": 2.7},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(visualize_parallel_scaling, "CSV_PATH", csv_path)
    monkeypatch.setattr(visualize_parallel_scaling, "OUT_PATH", out_path)

    visualize_parallel_scaling.main()
    fig = plt.gcf()
    try:
        runtime_ax = fig.axes[0]
        speedup_ax = fig.axes[1]

        assert runtime_ax.get_yscale() == "log"
        assert "log2" in runtime_ax.get_ylabel()

        serial_line, parallel_line = runtime_ax.get_lines()[:2]
        assert serial_line.get_color() == color("rust")
        assert serial_line.get_marker() == mpl_marker("rust")
        assert parallel_line.get_color() == color("rust_parallel")
        assert parallel_line.get_marker() == mpl_marker("rust_parallel")

        speedup_lines = speedup_ax.get_lines()
        vs_serial_line = next(line for line in speedup_lines if line.get_label() == "vs Rust")
        vs_parallel_line = next(
            line
            for line in speedup_lines
            if line.get_label() == "vs Rust - Parallel @ 1 thread"
        )
        assert vs_serial_line.get_color() == color("rust")
        assert vs_serial_line.get_marker() == mpl_marker("rust")
        assert vs_parallel_line.get_color() == color("rust_parallel")
        assert vs_parallel_line.get_marker() == mpl_marker("rust_parallel")
    finally:
        plt.close(fig)
