#!/usr/bin/env python3
"""Visualize the parallel scaling CSV produced by bench_parallel_scaling.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH  = REPO_ROOT / "results" / "parallel_scaling.csv"
OUT_PATH  = REPO_ROOT / "results" / "parallel_scaling.png"


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"missing {CSV_PATH} — run src/bench_parallel_scaling.py first")
    df = pd.read_csv(CSV_PATH)

    serial = df[df["threads"] == 0].iloc[0]
    par    = df[df["threads"] > 0].sort_values("threads").reset_index(drop=True)

    # Speedup is computed two ways:
    #   * vs serial-Rust baseline (the honest end-to-end comparison)
    #   * vs parallel-1-thread (isolates pure parallel scaling, excluding rayon overhead)
    par["speedup_vs_serial"]    = serial["median_s"] / par["median_s"]
    par_1t                       = par[par["threads"] == 1]["median_s"].iloc[0]
    par["speedup_vs_parallel1"] = par_1t / par["median_s"]

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Runtime panel
    ax_t.axhline(serial["median_s"], color="#dc2626", ls="--", lw=1.5,
                 label=f"serial baseline ({serial['median_s']:.2f}s)")
    ax_t.plot(par["threads"], par["median_s"], marker="o", lw=2,
              color="#0ea5e9", label="parallel")
    ax_t.fill_between(par["threads"], par["min_s"], par["max_s"],
                      color="#0ea5e9", alpha=0.30, label="min/max")
    ax_t.set_xlabel("Threads")
    ax_t.set_ylabel("Wall-clock runtime (s)")
    ax_t.set_title("Runtime vs thread count")
    ax_t.set_xscale("log", base=2)
    ax_t.set_xticks(par["threads"])
    ax_t.set_xticklabels([str(t) for t in par["threads"]])
    ax_t.grid(True, ls=":", alpha=0.6)
    ax_t.legend()

    # --- Speedup panel
    max_t = int(par["threads"].max())
    ax_s.plot([1, max_t], [1, max_t], ls="--", color="#9ca3af",
              label="ideal (linear)")
    ax_s.plot(par["threads"], par["speedup_vs_serial"], marker="s", lw=2,
              color="#dc2626", label="vs serial Rust")
    ax_s.plot(par["threads"], par["speedup_vs_parallel1"], marker="o", lw=2,
              color="#0ea5e9", label="vs parallel @ 1 thread")

    # Vertical dashed line at the physical core count.
    # os.cpu_count() returns logical CPUs; on most laptops that equals physical
    # cores × 2 (HT).  We use it as a practical "wall" marker regardless.
    n_phys = os.cpu_count() or 1
    ax_s.axvline(n_phys, ls=":", color="#6b7280", lw=1.4)
    ax_s.text(n_phys * 1.05, ax_s.get_ylim()[0] if ax_s.get_ylim()[0] > 0 else 1,
              "physical cores", fontsize=8, color="#6b7280", va="bottom")

    # Efficiency curve (speedup / threads) on a secondary right-y axis.
    # Efficiency lives in [0, ~1.2]; speedup can reach N so they need separate scales.
    ax_eff = ax_s.twinx()
    par["efficiency"] = par["speedup_vs_parallel1"] / par["threads"]
    ax_eff.plot(par["threads"], par["efficiency"], marker="^", lw=1.5,
                color="#a855f7", ls="--", label="efficiency")
    ax_eff.set_ylabel("Efficiency (speedup / threads)", color="#a855f7")
    ax_eff.tick_params(axis="y", labelcolor="#a855f7")
    ax_eff.set_ylim(0, 1.4)
    ax_eff.axhline(1.0, ls=":", color="#a855f7", lw=0.8, alpha=0.5)
    # Merge legends from both axes.
    handles_s, labels_s = ax_s.get_legend_handles_labels()
    handles_e, labels_e = ax_eff.get_legend_handles_labels()
    ax_s.legend(handles_s + handles_e, labels_s + labels_e, fontsize=8)

    ax_s.set_xlabel("Threads")
    ax_s.set_ylabel("Speedup")
    ax_s.set_title("Speedup vs thread count")
    ax_s.set_xscale("log", base=2)
    ax_s.set_yscale("log", base=2)
    ax_s.set_xticks(par["threads"])
    ax_s.set_xticklabels([str(t) for t in par["threads"]])
    ax_s.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Rust K-Means parallel scaling (Rayon)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
