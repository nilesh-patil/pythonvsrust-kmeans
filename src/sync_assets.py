#!/usr/bin/env python3
"""Mirror results/*.{svg,png} and results/animations/*.gif into docs/assets/. Idempotent."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results"
DOCS      = REPO_ROOT / "docs"


def sync(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob(pattern)):
        dst = dst_dir / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            count += 1
    return count


def main() -> None:
    images = DOCS / "assets" / "images"
    n_svg  = sync(RESULTS,                images,                         "*.svg")
    n_png  = sync(RESULTS,                images,                         "*.png")
    n_gif  = sync(RESULTS / "animations", DOCS / "assets" / "animations", "*.gif")
    n_dash = sync(RESULTS / "dashboards", DOCS / "dashboard",             "*.html")
    print(f"synced  {n_svg} SVG, {n_png} PNG, {n_gif} GIF, {n_dash} dashboard HTML")


if __name__ == "__main__":
    sys.exit(main())
