"""SHAP explanations for the fitted logistic regression."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from ..schemas import Contribution, LocalExplanation, ShapImportance, ShapSummary
from .eda import split_feature_types
from .modeling import DataSplit, FittedModel, _xy
from .plotting import new_figure, save


def _value_str(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else f"{v:.4g}"
    return str(v)


def _percentile(X_train: pd.DataFrame, feature: str, value, numeric: list[str]) -> float | None:
    if feature not in numeric or pd.isna(value):
        return None
    ref = X_train[feature].astype(float).dropna().to_numpy()
    return round(float((ref < float(value)).mean() * 100), 1) if ref.size else None


def compute_shap(model: FittedModel, df: pd.DataFrame, target: str, split: DataSplit,
                 figures_dir: Path, run_name: str, background_samples: int,
                 n_local: int, top_k: int) -> ShapSummary:
    pre = model.pipeline.named_steps["pre"]
    clf = model.pipeline.named_steps["clf"]
    X_train, _ = _xy(df, model.features, target, split.train_idx)
    X_val, y_val = _xy(df, model.features, target, split.val_idx)
    Xt_train, Xt_val = pre.transform(X_train), pre.transform(X_val)

    masker = shap.maskers.Independent(Xt_train, max_samples=min(background_samples, len(Xt_train)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = shap.LinearExplainer(clf, masker)(Xt_val)

    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float)
    if values.ndim == 2:  # binary: one output, the log-odds of classes[1]
        values = values[:, :, None]
        base = base.reshape(-1, 1) if base.ndim else np.full((len(values), 1), float(base))
        explained = [model.classes[1]]
    else:
        base = np.broadcast_to(base, (values.shape[0], values.shape[2]))
        explained = list(model.classes)

    # Sum one-hot columns back onto their original categorical feature.
    feats = model.features
    owner_ix = np.array([feats.index(o) for o in model.term_to_feature])
    agg = np.zeros((values.shape[0], len(feats), values.shape[2]))
    for j, f_ix in enumerate(owner_ix):
        agg[:, f_ix, :] += values[:, j, :]

    numeric, _ = split_feature_types(df, feats)
    importance = _global_importance(agg, X_val, feats, numeric, explained)

    proba = model.pipeline.predict_proba(X_val)
    pred_ix = proba.argmax(axis=1)
    rows = _pick_rows(proba, pred_ix, y_val, model.classes, n_local)
    local = []
    for row, kind in rows:
        out_k = 0 if len(explained) == 1 else int(pred_ix[row])
        contrib = agg[row, :, out_k]
        order = np.argsort(-np.abs(contrib))[:top_k]
        local.append(LocalExplanation(
            row_index=int(split.val_idx[row]),
            kind=kind,
            true_label=str(y_val[row]),
            predicted_label=str(model.classes[pred_ix[row]]),
            predicted_probability=round(float(proba[row, pred_ix[row]]), 4),
            explained_class=str(explained[out_k]),
            base_value=round(float(base[row, out_k]), 4),
            top_contributions=[
                Contribution(feature=feats[i], value=_value_str(X_val.iloc[row, i]),
                             percentile=_percentile(X_train, feats[i], X_val.iloc[row, i], numeric),
                             shap=round(float(contrib[i]), 4))
                for i in order
            ],
        ))

    figures = [
        _plot_bar(importance, figures_dir / f"{run_name}_shap_importance.png", run_name),
        _plot_beeswarm(agg, pred_ix if len(explained) > 1 else None, X_val, feats, numeric,
                       importance, figures_dir / f"{run_name}_shap_beeswarm.png", run_name,
                       explained),
    ]
    return ShapSummary(
        method="shap.LinearExplainer (interventional, training-set background)",
        output_space="log-odds",
        explained_classes=[str(c) for c in explained],
        global_importance=importance,
        local_explanations=local,
        figures=figures,
    )


def _global_importance(agg, X_val, feats, numeric, explained) -> list[ShapImportance]:
    mean_abs = np.abs(agg).mean(axis=0).mean(axis=1)  # average over rows, then classes
    total = mean_abs.sum() or 1.0
    out = []
    for i, f in enumerate(feats):
        direction: dict[str, float | None] = {}
        for k, cls in enumerate(explained):
            if f in numeric:
                x = X_val[f].astype(float).to_numpy()
                s = agg[:, i, k]
                ok = ~np.isnan(x)
                if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(s[ok]) > 0:
                    direction[str(cls)] = round(float(np.corrcoef(x[ok], s[ok])[0, 1]), 3)
                else:
                    direction[str(cls)] = None
            else:
                direction[str(cls)] = None
        out.append(ShapImportance(feature=f, mean_abs_shap=round(float(mean_abs[i]), 4),
                                  share=round(float(mean_abs[i] / total), 4),
                                  direction_by_class=direction))
    return sorted(out, key=lambda s: -s.mean_abs_shap)


def _pick_rows(proba, pred_ix, y_val, classes, n_local) -> list[tuple[int, str]]:
    """Choose a few representative validation rows to explain."""
    correct = np.array([classes[p] == y for p, y in zip(pred_ix, y_val)])
    top = proba.max(axis=1)
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    picks: dict[int, str] = {}

    def add(candidates: np.ndarray, key: np.ndarray, kind: str, limit: int = 1):
        for i in candidates[np.argsort(key[candidates], kind="stable")]:
            if len(picks) >= n_local or limit == 0:
                return
            if int(i) not in picks:
                picks[int(i)] = kind
                limit -= 1

    wrong, right = np.where(~correct)[0], np.where(correct)[0]
    add(right, -top, "confident correct")
    add(np.arange(len(proba)), margin, "borderline")
    add(wrong, -top, "confident misclassification")
    # Fill any remaining slots: other misclassifications first, then near-boundary correct rows.
    add(wrong, -top, "misclassified", limit=n_local)
    add(right, margin, "correct, near the decision boundary", limit=n_local)
    return list(picks.items())


def _plot_bar(importance: list[ShapImportance], path: Path, run_name: str) -> str:
    items = importance[:15][::-1]
    fig = new_figure(6, 0.35 * len(items) + 1.5)
    ax = fig.subplots()
    ax.barh([i.feature for i in items], [i.mean_abs_shap for i in items], color="#4C72B0")
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title(f"Global feature importance — {run_name}")
    return save(fig, path)


def _plot_beeswarm(agg, pred_ix, X_val, feats, numeric, importance, path: Path, run_name: str,
                   explained) -> str:
    """Strip plot of per-row SHAP values, coloured by feature value."""
    order = [feats.index(i.feature) for i in importance[:15]][::-1]
    rng = np.random.default_rng(0)
    fig = new_figure(7, 0.45 * len(order) + 1.5)
    ax = fig.subplots()
    sc = None
    for y_pos, f_ix in enumerate(order):
        if pred_ix is None:
            s = agg[:, f_ix, 0]
        else:  # multiclass: contribution toward each row's predicted class
            s = agg[np.arange(len(agg)), f_ix, pred_ix]
        jitter = rng.uniform(-0.3, 0.3, len(s))
        f = feats[f_ix]
        if f in numeric:
            x = X_val[f].astype(float).to_numpy()
            lo, hi = np.nanpercentile(x, [5, 95]) if np.isfinite(x).any() else (0, 1)
            c = np.clip((x - lo) / ((hi - lo) or 1), 0, 1)
            sc = ax.scatter(s, y_pos + jitter, c=c, cmap="coolwarm", s=8, vmin=0, vmax=1)
        else:
            ax.scatter(s, y_pos + jitter, color="gray", s=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(order)), [feats[i] for i in order])
    target = f"toward {explained[0]}" if pred_ix is None else "toward predicted class"
    ax.set_xlabel(f"SHAP value (log-odds {target})")
    ax.set_title(f"SHAP values per validation row — {run_name}")
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("feature value (low → high)")
    return save(fig, path)
