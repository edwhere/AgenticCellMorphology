"""Dependencies shared with agent tools.

Tools write their exact results into ``results``; the orchestrator assembles the
reports from there, so numbers never pass through the LLM.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import AnalysisRequest
from .tools.modeling import DataSplit


@dataclass
class AnalysisDeps:
    df: pd.DataFrame
    request: AnalysisRequest
    output_dir: Path
    features: list[str]  # features in play for this agent run
    run_name: str = "eda"
    split: DataSplit | None = None
    results: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def figures_dir(self) -> Path:
        path = self.output_dir / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path
