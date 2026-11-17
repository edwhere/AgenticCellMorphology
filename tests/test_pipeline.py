"""End-to-end orchestration with pydantic-ai's TestModel (no API calls)."""

import json
from contextlib import ExitStack

import pytest
from pydantic_ai.models.test import TestModel

from agenticcellmorph.agents import (
    eda_agent, modeling_agent, parser_agent, summary_agent, validate_request,
)
from agenticcellmorph.orchestrator import run_analysis
from agenticcellmorph.schemas import AnalysisRequest

FEATURES = ["a", "b", "c", "d", "color"]


@pytest.fixture
def test_models():
    with ExitStack() as stack:
        for agent in (eda_agent, modeling_agent, summary_agent):
            stack.enter_context(agent.override(model=TestModel()))
        yield


async def test_two_runs_when_features_correlated(csv_path, tmp_path, test_models):
    req = AnalysisRequest(csv_path=str(csv_path), features=FEATURES, target="y")
    result = await run_analysis(request=req, output_dir=tmp_path / "out", log=lambda m: None)

    assert [(p.feature_a, p.feature_b) for p in result.eda.correlated_pairs] == [("a", "b")]
    assert [r.run_name for r in result.runs] == ["all_features", "uncorrelated_features"]
    dropped = result.comparison.dropped_features
    assert len(dropped) == 1 and dropped[0] not in result.runs[1].features_used
    assert len(result.comparison.cv_accuracy_fold_diffs) == 5
    # Both runs share the split: same validation size, same baseline.
    assert result.runs[0].baseline_accuracy == result.runs[1].baseline_accuracy

    out = tmp_path / "out"
    assert (out / "report.md").exists()
    saved = json.loads((out / "results.json").read_text())
    assert saved["runs"][1]["shap"]["global_importance"]
    assert (out / "figures" / "correlation_heatmap.png").exists()
    assert (out / "figures" / "uncorrelated_features_shap_beeswarm.png").exists()


async def test_one_run_without_correlated_features(csv_path, tmp_path, test_models):
    req = AnalysisRequest(csv_path=str(csv_path), features=["a", "c", "d", "color"], target="y")
    result = await run_analysis(request=req, output_dir=tmp_path / "out", log=lambda m: None)
    assert result.eda.correlated_pairs == []
    assert [r.run_name for r in result.runs] == ["all_features"]
    assert result.selection is None and result.comparison is None


def test_validate_request(csv_path):
    ok = AnalysisRequest(csv_path=str(csv_path), features=["a"], target="y")
    assert validate_request(ok) == []
    assert "not in the file" in validate_request(ok.model_copy(update={"features": ["zzz"]}))[0]
    assert "also listed" in validate_request(ok.model_copy(update={"features": ["a", "y"]}))[0]
    assert "not found" in validate_request(ok.model_copy(update={"csv_path": "/nope.csv"}))[0]


async def test_parser_agent_accepts_valid_output(csv_path):
    args = {"csv_path": str(csv_path), "features": ["a", "c"], "target": "y"}
    with parser_agent.override(model=TestModel(custom_output_args=args)):
        result = await parser_agent.run(f"Use {csv_path}, features a and c, target y")
    assert result.output.features == ["a", "c"]
