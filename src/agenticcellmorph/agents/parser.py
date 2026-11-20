"""Agent 0: turns the user's free-text prompt into an AnalysisRequest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic_ai import Agent, ModelRetry

from ..schemas import AnalysisRequest

parser_agent = Agent(
    None,
    output_type=AnalysisRequest,
    name="request_parser",
    instructions=(
        "Extract the analysis request from the user's message: the path to the CSV file, "
        "the list of feature variable names, and the single target variable name. "
        "Copy the path and column names exactly as written; do not invent columns. "
        "If the user says 'all other columns' (or similar) are features, list every column "
        "in the file except the target."
    ),
    retries=3,
    defer_model_check=True,
)


def validate_request(req: AnalysisRequest) -> list[str]:
    """Return a list of problems with the request (empty if it is usable)."""
    path = Path(req.csv_path).expanduser()
    if not path.is_file():
        return [f"CSV file not found: {req.csv_path}"]
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # noqa: BLE001 - report any parse failure back to the model/user
        return [f"Could not read {req.csv_path} as CSV: {exc}"]
    problems = []
    unknown = [c for c in [*req.features, req.target] if c not in columns]
    if unknown:
        problems.append(f"Columns not in the file: {unknown}. Available columns: {columns}")
    if req.target in req.features:
        problems.append(f"The target '{req.target}' is also listed as a feature.")
    if not req.features:
        problems.append("No feature variables were given.")
    if len(set(req.features)) != len(req.features):
        problems.append("Some features are listed more than once.")
    return problems


@parser_agent.output_validator
def _check(req: AnalysisRequest) -> AnalysisRequest:
    problems = validate_request(req)
    if problems:
        raise ModelRetry("; ".join(problems))
    return req
