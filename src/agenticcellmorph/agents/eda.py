"""Agent 1: descriptive statistics, correlations and figures for the features."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .. import config
from ..deps import AnalysisDeps
from ..tools import eda as T


class EDANotes(BaseModel):
    notes: list[str] = Field(
        description="3-8 short observations about the data (skew, outliers, missing values, "
        "class balance, correlated features, features that separate the classes)."
    )


REQUIRED = ("feature_stats", "target_distribution", "correlation_matrix", "correlated_pairs")

eda_agent = Agent(
    None,
    deps_type=AnalysisDeps,
    output_type=EDANotes,
    name="eda_analyst",
    instructions=(
        "You are a data analyst doing exploratory analysis of the feature variables in a CSV file. "
        "Use your tools: describe the features, check the target distribution, find strongly "
        "correlated feature pairs, and make the distribution, correlation-heatmap and "
        "feature-vs-target figures. Every tool works on the features you were given; you do not "
        "need to pass column names. Then return short, specific observations grounded in the "
        "tool results. Quote numbers exactly as the tools reported them."
    ),
    retries=2,
    defer_model_check=True,
)


@eda_agent.tool
def describe_features(ctx: RunContext[AnalysisDeps]) -> list[dict]:
    """Summary statistics for every feature: count, missing, unique values, mean, std,
    quartiles, skew (numeric) or most common categories (categorical)."""
    stats = T.describe_features(ctx.deps.df, ctx.deps.features)
    ctx.deps.results["feature_stats"] = stats
    return [s.model_dump(exclude_none=True) for s in stats]


@eda_agent.tool
def get_target_distribution(ctx: RunContext[AnalysisDeps]) -> dict[str, int]:
    """Number of rows in each class of the target variable."""
    dist = T.target_distribution(ctx.deps.df, ctx.deps.request.target)
    ctx.deps.results["target_distribution"] = dist
    return dist


@eda_agent.tool
def find_correlated_features(ctx: RunContext[AnalysisDeps]) -> dict:
    """Correlation matrix of the numeric features and the pairs whose |r| is at or above
    the configured threshold."""
    corr = T.correlation_matrix(ctx.deps.df, ctx.deps.features, config.CORR_METHOD)
    pairs = T.correlated_pairs(corr, config.CORR_THRESHOLD)
    ctx.deps.results["correlation_matrix"] = T.corr_to_dict(corr)
    ctx.deps.results["correlated_pairs"] = pairs
    return {
        "method": config.CORR_METHOD,
        "threshold": config.CORR_THRESHOLD,
        "correlated_pairs": [p.model_dump() for p in pairs],
        "matrix": T.corr_to_dict(corr),
    }


def _add_figure(ctx: RunContext[AnalysisDeps], path: str) -> str:
    ctx.deps.results.setdefault("figures", [])
    if path not in ctx.deps.results["figures"]:
        ctx.deps.results["figures"].append(path)
    return f"Saved figure to {path}"


@eda_agent.tool
def plot_distributions(ctx: RunContext[AnalysisDeps]) -> str:
    """Save a histogram (numeric) or bar chart (categorical) for each feature."""
    path = T.plot_distributions(ctx.deps.df, ctx.deps.features,
                                ctx.deps.figures_dir / "feature_distributions.png")
    return _add_figure(ctx, path)


@eda_agent.tool
def plot_correlation_heatmap(ctx: RunContext[AnalysisDeps]) -> str:
    """Save a heatmap of the numeric feature correlation matrix."""
    corr = T.correlation_matrix(ctx.deps.df, ctx.deps.features, config.CORR_METHOD)
    if corr.empty:
        return "No numeric features, so no heatmap."
    path = T.plot_correlation_heatmap(corr, config.CORR_THRESHOLD,
                                      ctx.deps.figures_dir / "correlation_heatmap.png")
    return _add_figure(ctx, path)


@eda_agent.tool
def plot_features_by_target(ctx: RunContext[AnalysisDeps]) -> str:
    """Save box plots of each numeric feature split by target class."""
    try:
        path = T.plot_features_by_target(ctx.deps.df, ctx.deps.features, ctx.deps.request.target,
                                         ctx.deps.figures_dir / "features_by_target.png")
    except ValueError as exc:
        return str(exc)
    return _add_figure(ctx, path)


@eda_agent.output_validator
def _require_tools(ctx: RunContext[AnalysisDeps], out: EDANotes) -> EDANotes:
    missing = [k for k in REQUIRED if k not in ctx.deps.results]
    if missing:
        raise ModelRetry(f"Call the tools that produce {missing} before answering.")
    return out
