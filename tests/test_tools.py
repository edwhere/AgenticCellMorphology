"""Deterministic tool functions (no LLM involved)."""

import numpy as np
import pytest

from agenticcellmorph.tools import eda, explain, modeling, selection
from conftest import make_df

FEATURES = ["a", "b", "c", "d", "color"]


def test_describe_and_correlations(df):
    stats = {s.name: s for s in eda.describe_features(df, FEATURES)}
    assert stats["a"].kind == "numeric" and stats["a"].count == len(df)
    assert stats["color"].kind == "categorical" and set(stats["color"].top_categories) == {"red", "blue"}

    corr = eda.correlation_matrix(df, FEATURES, "pearson")
    assert list(corr.columns) == ["a", "b", "c", "d"]  # categoricals excluded
    pairs = eda.correlated_pairs(corr, 0.8)
    assert [(p.feature_a, p.feature_b) for p in pairs] == [("a", "b")]


def test_split_is_disjoint_and_stratified(df):
    split = modeling.make_split(df, "y", 0.2, 5, 42)
    assert split.stratified
    assert not set(split.train_idx) & set(split.val_idx)
    assert len(split.train_idx) + len(split.val_idx) == len(df)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_fit_cv_evaluate_shap(tmp_path, n_classes):
    df = make_df(n_classes=n_classes)
    split = modeling.make_split(df, "y", 0.2, 5, 42)

    cv = modeling.cross_validate_model(df, FEATURES, "y", split, 2000)
    assert len(cv.accuracy.folds) == 5 and 0.6 < cv.accuracy.mean <= 1
    assert cv.roc_auc is not None

    model = modeling.fit_model(df, FEATURES, "y", split, 2000)
    assert model.terms[-2:] == ["color=blue", "color=red"]
    metrics = modeling.evaluate(model, df, "y", split)
    assert metrics.accuracy > modeling.baseline_accuracy(df, "y", split)
    assert np.sum(metrics.confusion_matrix) == len(split.val_idx)

    coefs = modeling.coefficients(model)
    assert len(coefs) == len(model.terms) * (1 if n_classes == 2 else n_classes)

    summary = explain.compute_shap(model, df, "y", split, tmp_path, "run", 200, 4, 3)
    assert {i.feature for i in summary.global_importance} == set(FEATURES)  # one-hot re-aggregated
    assert abs(sum(i.share for i in summary.global_importance) - 1) < 1e-3
    assert summary.global_importance[0].feature in {"a", "b", "c"}
    assert len(summary.explained_classes) == (1 if n_classes == 2 else n_classes)
    assert 1 <= len(summary.local_explanations) <= 4
    assert len({e.row_index for e in summary.local_explanations}) == len(summary.local_explanations)
    for path in summary.figures:
        assert (tmp_path / path.split("/")[-1]).exists()


def test_shap_matches_linear_closed_form(df, tmp_path):
    """For a linear model, SHAP (interventional) = coef * (x - background mean)."""
    feats = ["a", "c", "d"]
    split = modeling.make_split(df, "y", 0.2, 5, 42)
    model = modeling.fit_model(df, feats, "y", split, 2000)
    summary = explain.compute_shap(model, df, "y", split, tmp_path, "r", 10_000, 1, 3)
    pre, clf = model.pipeline.named_steps["pre"], model.pipeline.named_steps["clf"]
    Xt_tr = pre.transform(df.iloc[split.train_idx][feats])
    expl = summary.local_explanations[0]
    xt = pre.transform(df.iloc[[expl.row_index]][feats])[0]
    expected = clf.coef_[0] * (xt - Xt_tr.mean(axis=0))
    got = {c.feature: c.shap for c in expl.top_contributions}
    for f, e in zip(feats, expected):
        assert got[f] == pytest.approx(e, abs=1e-3)


def test_select_uncorrelated_drops_one_of_pair(df):
    split = modeling.make_split(df, "y", 0.2, 5, 42)
    sel = selection.select_uncorrelated_features(df, FEATURES, "y", split, 0.8, "pearson")
    assert [d.feature for d in sel.dropped] in (["a"], ["b"])
    assert "color" in sel.kept and len(sel.kept) == len(FEATURES) - 1


def test_select_chain_of_correlations():
    rng = np.random.default_rng(1)
    df = make_df()
    df["e"] = df["a"] + rng.normal(0, 0.1, len(df))  # a, b, e all mutually correlated
    split = modeling.make_split(df, "y", 0.2, 5, 42)
    sel = selection.select_uncorrelated_features(df, ["a", "b", "e", "c"], "y", split, 0.8, "pearson")
    assert len(sel.dropped) == 2
    assert "c" in sel.kept
