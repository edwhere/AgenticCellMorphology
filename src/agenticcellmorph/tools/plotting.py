"""Thread-safe figure helpers.

Agent tools can run in worker threads (and two modeling runs run concurrently),
so figures are built with matplotlib's object API, never pyplot, and saved under
a lock.
"""

from __future__ import annotations

import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

PLOT_LOCK = threading.Lock()


def new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained")


def save(fig: Figure, path: Path) -> str:
    with PLOT_LOCK:
        fig.savefig(path, dpi=120)
    return str(path)


def grid_shape(n: int, max_cols: int = 4) -> tuple[int, int]:
    cols = min(max_cols, max(n, 1))
    rows = (n + cols - 1) // cols
    return rows, cols
