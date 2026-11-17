"""Train/validation split, cross-validation and logistic regression fitting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..schemas import Coefficient, CVResult, MetricSummary, ValidationMetrics
from .eda import split_feature_types


@dataclass
class DataSplit:
    """One train/validation split and CV scheme, shared by every model run."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    cv: StratifiedKFold | KFold
    stratified: bool


@dataclass
class FittedModel:
    pipeline: Pipeline
    features: list[str]
    classes: list
    terms: list[str]  # transformed column names
    term_to_feature: list[str]  # original feature for each transformed column


def make_split(df: pd.DataFrame, target: str, test_size: float, n_folds: int,
               random_state: int) -> DataSplit:
    idx = np.arange(len(df))
    y = df[target].to_numpy()
    counts = pd.Series(y).value_counts()
    # Stratify only when every class has enough rows for both the split and the folds.
    stratified = bool(counts.min() >= max(2, n_folds + 1))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state,
        stratify=y if stratified else None,
    )
    cv_cls = StratifiedKFold if stratified else KFold
    cv = cv_cls(n_splits=n_folds, shuffle=True, random_state=random_state)
    return DataSplit(np.sort(train_idx), np.sort(val_idx), cv, stratified)


def build_pipeline(df: pd.DataFrame, features: list[str], max_iter: int) -> Pipeline:
    numeric, categorical = split_feature_types(df, features)
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=max_iter))])


def _xy(df: pd.DataFrame, features: list[str], target: str, idx: np.ndarray):
    sub = df.iloc[idx]
    X = sub[features].copy()
    _, categorical = split_feature_types(df, features)
    for c in categorical:  # uniform object dtype so the imputer/encoder behave
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X, sub[target].to_numpy()


def baseline_accuracy(df: pd.DataFrame, target: str, split: DataSplit) -> float:
    y = df[target].to_numpy()
    majority = pd.Series(y[split.train_idx]).mode().iloc[0]
    return float(np.mean(y[split.val_idx] == majority))


def _summary(values: np.ndarray) -> MetricSummary | None:
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        return None
    return MetricSummary(mean=round(values.mean(), 4), std=round(values.std(ddof=1), 4),
                         folds=[round(v, 4) for v in values])


def cross_validate_model(df: pd.DataFrame, features: list[str], target: str, split: DataSplit,
                         max_iter: int) -> CVResult:
    """K-fold CV on the training rows only; the validation rows are never touched."""
    X, y = _xy(df, features, target, split.train_idx)
    n_classes = len(np.unique(y))
    auc_name = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            build_pipeline(df, features, max_iter), X, y, cv=split.cv,
            scoring={"accuracy": "accuracy", "auc": auc_name}, error_score=np.nan,
        )
    return CVResult(
        n_folds=split.cv.get_n_splits(),
        accuracy=_summary(scores["test_accuracy"]),
        roc_auc=_summary(scores["test_auc"]),
    )


def fit_model(df: pd.DataFrame, features: list[str], target: str, split: DataSplit,
              max_iter: int) -> FittedModel:
    X, y = _xy(df, features, target, split.train_idx)
    pipe = build_pipeline(df, features, max_iter).fit(X, y)
    terms, term_to_feature = _term_map(pipe, df, features)
    return FittedModel(pipe, features, list(pipe.classes_), terms, term_to_feature)


def _term_map(pipe: Pipeline, df: pd.DataFrame, features: list[str]):
    numeric, categorical = split_feature_types(df, features)
    terms, owners = list(numeric), list(numeric)
    if categorical:
        enc = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(categorical, enc.categories_):
            for cat in cats:
                terms.append(f"{col}={cat}")
                owners.append(col)
    return terms, owners


def evaluate(model: FittedModel, df: pd.DataFrame, target: str, split: DataSplit) -> ValidationMetrics:
    X, y = _xy(df, model.features, target, split.val_idx)
    pred = model.pipeline.predict(X)
    proba = model.pipeline.predict_proba(X)
    classes = model.classes
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y == classes[1], proba[:, 1])
        else:
            auc = roc_auc_score(y, proba, multi_class="ovr", labels=classes)
        auc = round(float(auc), 4)
    except ValueError:  # e.g. a class missing from the validation rows
        auc = None
    return ValidationMetrics(
        accuracy=round(float(accuracy_score(y, pred)), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(y, pred)), 4),
        precision_macro=round(float(p), 4),
        recall_macro=round(float(r), 4),
        f1_macro=round(float(f1), 4),
        roc_auc=auc,
        labels=[str(c) for c in classes],
        confusion_matrix=confusion_matrix(y, pred, labels=classes).tolist(),
    )


def coefficients(model: FittedModel) -> list[Coefficient]:
    """Coefficients on standardized numeric features / one-hot columns, largest first."""
    clf: LogisticRegression = model.pipeline.named_steps["clf"]
    coef = np.atleast_2d(clf.coef_)
    class_labels = [model.classes[1]] if coef.shape[0] == 1 else model.classes
    out = [
        Coefficient(term=term, feature=owner, class_label=str(cls), coef=round(float(coef[k, j]), 4))
        for k, cls in enumerate(class_labels)
        for j, (term, owner) in enumerate(zip(model.terms, model.term_to_feature))
    ]
    return sorted(out, key=lambda c: -abs(c.coef))
