"""Descriptive statistics, correlations and EDA figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schemas import CorrelatedPair, FeatureStats
from .plotting import grid_shape, new_figure, save


def _f(x) -> float | None:
    """Round to a JSON-friendly float, mapping NaN to None."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 6)


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def describe_features(df: pd.DataFrame, features: list[str]) -> list[FeatureStats]:
    numeric, _ = split_feature_types(df, features)
    stats = []
    for col in features:
        s = df[col]
        base = dict(
            name=col,
            count=int(s.notna().sum()),
            missing=int(s.isna().sum()),
            n_unique=int(s.nunique(dropna=True)),
        )
        if col in numeric:
            d = s.astype(float).describe()
            stats.append(
                FeatureStats(
                    kind="numeric",
                    mean=_f(d["mean"]),
                    std=_f(d["std"]),
                    min=_f(d["min"]),
                    q25=_f(d["25%"]),
                    median=_f(d["50%"]),
                    q75=_f(d["75%"]),
                    max=_f(d["max"]),
                    skew=_f(s.astype(float).skew()),
                    **base,
                )
            )
        else:
            top = s.astype(str).where(s.notna()).value_counts().head(10)
            stats.append(
                FeatureStats(
                    kind="categorical",
                    top_categories={str(k): int(v) for k, v in top.items()},
                    **base,
                )
            )
    return stats


def target_distribution(df: pd.DataFrame, target: str) -> dict[str, int]:
    counts = df[target].value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def correlation_matrix(df: pd.DataFrame, features: list[str], method: str) -> pd.DataFrame:
    numeric, _ = split_feature_types(df, features)
    return df[numeric].astype(float).corr(method=method)


def correlated_pairs(corr: pd.DataFrame, threshold: float) -> list[CorrelatedPair]:
    cols = list(corr.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if not np.isnan(r) and abs(r) >= threshold:
                pairs.append(CorrelatedPair(feature_a=a, feature_b=b, r=round(float(r), 4)))
    return sorted(pairs, key=lambda p: -abs(p.r))


def corr_to_dict(corr: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {a: {b: round(float(corr.loc[a, b]), 4) for b in corr.columns} for a in corr.index}


# ---------------------------------------------------------------- figures


def plot_distributions(df: pd.DataFrame, features: list[str], path: Path) -> str:
    numeric, _ = split_feature_types(df, features)
    rows, cols = grid_shape(len(features))
    fig = new_figure(3.2 * cols, 2.6 * rows)
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax, col in zip(axes, features):
        s = df[col].dropna()
        if col in numeric:
            ax.hist(s.astype(float), bins=30, color="#4C72B0", edgecolor="white")
        else:
            counts = s.astype(str).value_counts().head(15)
            ax.bar(counts.index, counts.values, color="#4C72B0")
            ax.tick_params(axis="x", labelrotation=45)
        ax.set_title(col, fontsize=10)
    for ax in axes[len(features) :]:
        ax.set_visible(False)
    fig.suptitle("Feature distributions")
    return save(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, threshold: float, path: Path) -> str:
    n = len(corr)
    size = max(4.0, 0.6 * n + 2)
    fig = new_figure(size + 1, size)
    ax = fig.subplots()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            weight = "bold" if i != j and abs(v) >= threshold else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, weight=weight,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Correlation matrix (bold: |r| ≥ {threshold})")
    return save(fig, path)


def plot_features_by_target(df: pd.DataFrame, features: list[str], target: str, path: Path) -> str:
    numeric, _ = split_feature_types(df, features)
    if not numeric:
        raise ValueError("No numeric features to plot against the target.")
    classes = sorted(df[target].dropna().unique(), key=str)
    rows, cols = grid_shape(len(numeric))
    fig = new_figure(3.2 * cols, 2.8 * rows)
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax, col in zip(axes, numeric):
        data = [df.loc[df[target] == c, col].dropna().astype(float) for c in classes]
        ax.boxplot(data, tick_labels=[str(c) for c in classes])
        ax.set_title(col, fontsize=10)
    for ax in axes[len(numeric) :]:
        ax.set_visible(False)
    fig.suptitle(f"Numeric features by {target}")
    return save(fig, path)
