"""Agent 2: logistic regression with cross-validation, validation metrics and SHAP.

The orchestrator runs this agent once per feature set. Each run gets its own deps
(features, run_name) but the same train/validation split and CV folds.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .. import config
from ..deps import AnalysisDeps
from ..tools import explain, modeling as M


class ModelNotes(BaseModel):
    notes: list[str] = Field(
        description="3-6 short observations: CV vs validation accuracy and stability, "
        "comparison with the baseline, the most influential features and their direction."
    )


REQUIRED = ("cv", "validation", "coefficients", "shap")

modeling_agent = Agent(
    None,
    deps_type=AnalysisDeps,
    output_type=ModelNotes,
    name="modeler",
    instructions=(
        "You build and assess a logistic regression classifier. The data split, feature set "
        "and preprocessing (median imputation, standardization, one-hot encoding) are already "
        "fixed for you. Use your tools to: describe the run, cross-validate on the training "
        "set, fit and evaluate on the held-out validation set, get the coefficients, and "
        "compute SHAP explanations. Then return short observations grounded in the tool "
        "results, quoting numbers exactly. Compare accuracy with the majority-class baseline."
    ),
    retries=2,
    defer_model_check=True,
)


def _ensure_fitted(deps: AnalysisDeps) -> M.FittedModel:
    with deps.lock:  # tools may run concurrently; fit exactly once
        if "model" not in deps.results:
            deps.results["model"] = M.fit_model(deps.df, deps.features, deps.request.target,
                                                deps.split, config.MAX_ITER)
            deps.results["validation"] = M.evaluate(deps.results["model"], deps.df,
                                                    deps.request.target, deps.split)
        return deps.results["model"]


@modeling_agent.tool
def describe_run(ctx: RunContext[AnalysisDeps]) -> dict:
    """The run name, features used, class labels, split sizes and baseline accuracy."""
    d = ctx.deps
    y = d.df[d.request.target]
    baseline = M.baseline_accuracy(d.df, d.request.target, d.split)
    d.results["baseline_accuracy"] = round(baseline, 4)
    return {
        "run_name": d.run_name,
        "features": d.features,
        "classes": sorted(map(str, y.unique())),
        "n_train": len(d.split.train_idx),
        "n_validation": len(d.split.val_idx),
        "stratified": d.split.stratified,
        "majority_class_baseline_accuracy": round(baseline, 4),
    }


@modeling_agent.tool
def run_cross_validation(ctx: RunContext[AnalysisDeps]) -> dict:
    """Stratified k-fold cross-validation on the training rows: accuracy and ROC-AUC per fold,
    with mean and standard deviation."""
    d = ctx.deps
    cv = M.cross_validate_model(d.df, d.features, d.request.target, d.split, config.MAX_ITER)
    d.results["cv"] = cv
    return cv.model_dump()


@modeling_agent.tool
def fit_and_evaluate(ctx: RunContext[AnalysisDeps]) -> dict:
    """Fit on the full training set and score on the held-out validation set: accuracy,
    balanced accuracy, macro precision/recall/F1, ROC-AUC and the confusion matrix."""
    _ensure_fitted(ctx.deps)
    return ctx.deps.results["validation"].model_dump()


@modeling_agent.tool
def get_coefficients(ctx: RunContext[AnalysisDeps]) -> list[dict]:
    """Model coefficients on standardized features (one-hot columns for categoricals),
    largest magnitude first. Positive values raise the log-odds of `class_label`."""
    model = _ensure_fitted(ctx.deps)
    coefs = M.coefficients(model)
    ctx.deps.results["coefficients"] = coefs
    return [c.model_dump() for c in coefs[:20]]


@modeling_agent.tool
def explain_with_shap(ctx: RunContext[AnalysisDeps]) -> dict:
    """SHAP values on the validation set: global importance per feature with direction,
    local explanations for a few representative rows, and summary figures."""
    d = ctx.deps
    model = _ensure_fitted(d)
    summary = explain.compute_shap(
        model, d.df, d.request.target, d.split, d.figures_dir, d.run_name,
        config.SHAP_BACKGROUND_SAMPLES, config.N_LOCAL_EXPLANATIONS, config.TOP_CONTRIBUTIONS,
    )
    d.results["shap"] = summary
    return summary.model_dump()


@modeling_agent.output_validator
def _require_tools(ctx: RunContext[AnalysisDeps], out: ModelNotes) -> ModelNotes:
    missing = [k for k in REQUIRED if k not in ctx.deps.results]
    if missing:
        raise ModelRetry(f"Call the tools that produce {missing} before answering.")
    return out
