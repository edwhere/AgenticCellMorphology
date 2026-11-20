"""Typed data passed between the agents and the orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """What the user asked for, extracted from their prompt."""

    csv_path: str = Field(description="Path to the CSV file, exactly as the user gave it.")
    features: list[str] = Field(description="Column names to use as feature variables.")
    target: str = Field(description="Column name of the target variable to predict.")


# ---------------------------------------------------------------- EDA


class FeatureStats(BaseModel):
    name: str
    kind: str  # "numeric" or "categorical"
    count: int
    missing: int
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max: float | None = None
    skew: float | None = None
    top_categories: dict[str, int] | None = None  # categorical only


class CorrelatedPair(BaseModel):
    feature_a: str
    feature_b: str
    r: float


class EDAReport(BaseModel):
    n_rows: int
    n_features: int
    target: str
    target_distribution: dict[str, int]
    feature_stats: list[FeatureStats]
    correlation_method: str
    correlation_threshold: float
    correlation_matrix: dict[str, dict[str, float]]
    correlated_pairs: list[CorrelatedPair]
    figures: list[str]
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- feature selection


class DroppedFeature(BaseModel):
    feature: str
    correlated_with: str
    r: float
    reason: str


class FeatureSelection(BaseModel):
    threshold: float
    method: str
    kept: list[str]
    dropped: list[DroppedFeature]


# ---------------------------------------------------------------- modeling


class MetricSummary(BaseModel):
    mean: float
    std: float
    folds: list[float]


class CVResult(BaseModel):
    n_folds: int
    accuracy: MetricSummary
    roc_auc: MetricSummary | None = None


class ValidationMetrics(BaseModel):
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: float | None = None
    labels: list[str]
    confusion_matrix: list[list[int]]  # rows = true label, columns = predicted label


class Coefficient(BaseModel):
    term: str  # transformed column, e.g. "area" or "shape=round"
    feature: str  # original feature it came from
    class_label: str  # class whose log-odds this coefficient raises
    coef: float


class ShapImportance(BaseModel):
    feature: str
    mean_abs_shap: float
    share: float  # fraction of total mean |SHAP| across features
    # Correlation between the feature's value and its SHAP value, per explained class.
    # Positive: higher values push predictions toward that class. None for categoricals.
    direction_by_class: dict[str, float | None]


class Contribution(BaseModel):
    feature: str
    value: str
    # Where the value sits among training rows (0-100); None for categoricals/missing.
    percentile: float | None = None
    shap: float


class LocalExplanation(BaseModel):
    row_index: int
    kind: str  # e.g. "confident correct", "borderline", "misclassified"
    true_label: str
    predicted_label: str
    predicted_probability: float
    explained_class: str  # SHAP values below are log-odds contributions toward this class
    base_value: float
    top_contributions: list[Contribution]


class ShapSummary(BaseModel):
    method: str
    output_space: str
    explained_classes: list[str]
    global_importance: list[ShapImportance]
    local_explanations: list[LocalExplanation]
    figures: list[str]


class ModelReport(BaseModel):
    run_name: str
    features_used: list[str]
    classes: list[str]
    n_train: int
    n_validation: int
    baseline_accuracy: float  # always predicting the training majority class
    cv: CVResult
    validation: ValidationMetrics
    coefficients: list[Coefficient]
    shap: ShapSummary
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- comparison + summary


class ImportanceShift(BaseModel):
    feature: str
    share_full: float | None
    share_reduced: float | None


class RunComparison(BaseModel):
    full_run: str
    reduced_run: str
    dropped_features: list[str]
    cv_accuracy_full: float
    cv_accuracy_reduced: float
    cv_accuracy_diff: float  # reduced - full
    cv_accuracy_fold_diffs: list[float]  # paired, same folds
    cv_auc_diff: float | None
    val_accuracy_diff: float
    val_auc_diff: float | None
    importance_shift: list[ImportanceShift]


class FinalReport(BaseModel):
    markdown: str = Field(description="The full report in Markdown, with headings.")
    key_findings: list[str] = Field(description="3-7 one-sentence findings.")
    recommended_run: str = Field(
        description="run_name of the model you recommend, or the only run if there is one."
    )
    prediction_explanation: str = Field(
        description="Plain-language explanation of what drives the model's predictions."
    )


class AnalysisResult(BaseModel):
    request: AnalysisRequest
    eda: EDAReport
    selection: FeatureSelection | None
    runs: list[ModelReport]
    comparison: RunComparison | None
    final: FinalReport
