"""Choosing an uncorrelated feature subset, and comparing two model runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import DroppedFeature, FeatureSelection, ImportanceShift, ModelReport, RunComparison
from .eda import split_feature_types
from .modeling import DataSplit


def _target_association(df: pd.DataFrame, col: str, target: str) -> float:
    """Strongest |correlation| between the feature and any one-vs-rest class indicator."""
    x = df[col].astype(float)
    best = 0.0
    for cls in df[target].dropna().unique():
        r = x.corr((df[target] == cls).astype(float))
        if not np.isnan(r):
            best = max(best, abs(float(r)))
    return best


def select_uncorrelated_features(df: pd.DataFrame, features: list[str], target: str,
                                 split: DataSplit, threshold: float, method: str) -> FeatureSelection:
    """Greedily drop features until no remaining numeric pair has |r| >= threshold.

    Uses training rows only. Each step drops the feature with the highest mean |r|
    to the other remaining numeric features; ties go to keeping the feature more
    strongly associated with the target. Categorical features are always kept.
    """
    train = df.iloc[split.train_idx]
    numeric, _ = split_feature_types(df, features)
    corr = train[numeric].astype(float).corr(method=method).abs()
    assoc = {c: _target_association(train, c, target) for c in numeric}
    remaining = list(numeric)
    dropped: list[DroppedFeature] = []

    while True:
        arr = corr.loc[remaining, remaining].to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        sub = pd.DataFrame(arr, index=remaining, columns=remaining)
        involved = [c for c in remaining if (sub[c] >= threshold).any()]
        if not involved:
            break
        mean_r = sub.mean(axis=1) if len(remaining) > 1 else sub.max(axis=1)
        worst = max(involved, key=lambda c: (round(float(mean_r[c]), 9), -assoc[c]))
        partner = sub[worst].idxmax()
        r = float(train[[worst, partner]].astype(float).corr(method=method).iloc[0, 1])
        dropped.append(DroppedFeature(
            feature=worst, correlated_with=partner, r=round(r, 4),
            reason=(f"|r|={abs(r):.2f} with {partner}; highest mean |r| to the other features "
                    f"({mean_r[worst]:.2f}); target association {assoc[worst]:.2f} "
                    f"vs {assoc[partner]:.2f} for {partner}"),
        ))
        remaining.remove(worst)

    dropped_names = {d.feature for d in dropped}
    return FeatureSelection(
        threshold=threshold, method=method,
        kept=[f for f in features if f not in dropped_names], dropped=dropped,
    )


def compare_runs(full: ModelReport, reduced: ModelReport, selection: FeatureSelection) -> RunComparison:
    fold_diffs = [round(b - a, 4) for a, b in zip(full.cv.accuracy.folds, reduced.cv.accuracy.folds)]
    cv_auc = (
        round(reduced.cv.roc_auc.mean - full.cv.roc_auc.mean, 4)
        if full.cv.roc_auc and reduced.cv.roc_auc else None
    )
    val_auc = (
        round(reduced.validation.roc_auc - full.validation.roc_auc, 4)
        if full.validation.roc_auc is not None and reduced.validation.roc_auc is not None else None
    )
    share_full = {i.feature: i.share for i in full.shap.global_importance}
    share_red = {i.feature: i.share for i in reduced.shap.global_importance}
    shifts = [
        ImportanceShift(feature=f, share_full=share_full.get(f), share_reduced=share_red.get(f))
        for f in full.features_used
    ]
    return RunComparison(
        full_run=full.run_name,
        reduced_run=reduced.run_name,
        dropped_features=[d.feature for d in selection.dropped],
        cv_accuracy_full=full.cv.accuracy.mean,
        cv_accuracy_reduced=reduced.cv.accuracy.mean,
        cv_accuracy_diff=round(reduced.cv.accuracy.mean - full.cv.accuracy.mean, 4),
        cv_accuracy_fold_diffs=fold_diffs,
        cv_auc_diff=cv_auc,
        val_accuracy_diff=round(reduced.validation.accuracy - full.validation.accuracy, 4),
        val_auc_diff=val_auc,
        importance_shift=sorted(shifts, key=lambda s: -(s.share_full or 0)),
    )
