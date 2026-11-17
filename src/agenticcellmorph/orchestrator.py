"""Runs the agents in order and makes the branching decisions in plain Python.

    prompt -> parser -> EDA -> [correlated pairs?] -> modeling x1 or x2 -> comparison -> summary
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from . import config
from .agents import eda_agent, modeling_agent, parser_agent, summary_agent, validate_request
from .deps import AnalysisDeps
from .schemas import AnalysisRequest, AnalysisResult, EDAReport, FeatureSelection, ModelReport
from .tools.modeling import DataSplit, baseline_accuracy, make_split
from .tools.selection import compare_runs, select_uncorrelated_features

Log = Callable[[str], None]


def _stderr(msg: str) -> None:
    print(f"[agenticcellmorph] {msg}", file=sys.stderr, flush=True)


def load_data(request: AnalysisRequest) -> pd.DataFrame:
    df = pd.read_csv(Path(request.csv_path).expanduser())
    df = df[[*request.features, request.target]]
    df = df.dropna(subset=[request.target]).reset_index(drop=True)
    for col in request.features:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
    n_classes = df[request.target].nunique()
    if n_classes < 2:
        raise ValueError(f"Target '{request.target}' has {n_classes} class(es); need at least 2.")
    return df


async def parse_request(prompt: str) -> AnalysisRequest:
    result = await parser_agent.run(prompt, model=config.LLM_MODEL)
    return result.output


async def run_eda(df: pd.DataFrame, request: AnalysisRequest, output_dir: Path) -> EDAReport:
    deps = AnalysisDeps(df=df, request=request, output_dir=output_dir,
                        features=list(request.features))
    prompt = (f"Analyse the {len(request.features)} feature variables {request.features} "
              f"of {request.csv_path} ({len(df)} rows). Target variable: {request.target}.")
    notes = (await eda_agent.run(prompt, deps=deps, model=config.LLM_MODEL)).output
    r = deps.results
    return EDAReport(
        n_rows=len(df),
        n_features=len(request.features),
        target=request.target,
        target_distribution=r["target_distribution"],
        feature_stats=r["feature_stats"],
        correlation_method=config.CORR_METHOD,
        correlation_threshold=config.CORR_THRESHOLD,
        correlation_matrix=r["correlation_matrix"],
        correlated_pairs=r["correlated_pairs"],
        figures=r.get("figures", []),
        notes=notes.notes,
    )


async def run_model(df: pd.DataFrame, request: AnalysisRequest, output_dir: Path,
                    split: DataSplit, run_name: str, features: list[str]) -> ModelReport:
    deps = AnalysisDeps(df=df, request=request, output_dir=output_dir, features=features,
                        run_name=run_name, split=split)
    prompt = (f"Run '{run_name}': predict {request.target} from {len(features)} features "
              f"{features} with logistic regression.")
    notes = (await modeling_agent.run(prompt, deps=deps, model=config.LLM_MODEL)).output
    r = deps.results
    model = r["model"]
    return ModelReport(
        run_name=run_name,
        features_used=features,
        classes=[str(c) for c in model.classes],
        n_train=len(split.train_idx),
        n_validation=len(split.val_idx),
        baseline_accuracy=round(baseline_accuracy(df, request.target, split), 4),
        cv=r["cv"],
        validation=r["validation"],
        coefficients=r["coefficients"],
        shap=r["shap"],
        notes=notes.notes,
    )


def _summary_payload(request, eda, selection, runs, comparison) -> str:
    payload = {
        "request": request.model_dump(),
        "methodology": {
            "rows": "rows with a missing target are dropped",
            "split": f"{1 - config.TEST_SIZE:.0%} train / {config.TEST_SIZE:.0%} validation, "
                     f"stratified when possible, random_state={config.RANDOM_STATE}; "
                     "both runs use the identical split",
            "cross_validation": f"{config.CV_FOLDS}-fold (stratified) on the training rows only; "
                                "both runs use identical folds",
            "preprocessing": "numeric: median imputation then standardization; categorical: "
                             "most-frequent imputation then one-hot encoding of every level "
                             "(no dropped reference level). Fitted on training rows only.",
            "model": "scikit-learn LogisticRegression, L2 penalty, C=1.0",
            "feature_selection": "greedy: drop the feature with the highest mean |r| to the "
                                 "others until no pair has |r| >= threshold (training rows)",
            "shap": "values are log-odds contributions toward `explained_class`; "
                    "`percentile` is where a row's value sits among training rows (0 = lowest)",
        },
        "eda": eda.model_dump(),
        "feature_selection": selection.model_dump() if selection else None,
        "runs": [
            # keep the prompt compact: the 15 largest coefficients are plenty
            {**run.model_dump(exclude={"coefficients"}),
             "top_coefficients": [c.model_dump() for c in run.coefficients[:15]]}
            for run in runs
        ],
        "comparison": comparison.model_dump() if comparison else None,
    }
    return json.dumps(payload, indent=1)


async def run_analysis(
    prompt: str | None = None,
    *,
    request: AnalysisRequest | None = None,
    output_dir: str | Path = "outputs",
    log: Log = _stderr,
) -> AnalysisResult:
    """Run the full pipeline. Pass either a free-text ``prompt`` or a ready ``request``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Model: {config.LLM_MODEL}")

    if request is None:
        if not prompt:
            raise ValueError("Provide a prompt or a request.")
        log("Parsing the request…")
        request = await parse_request(prompt)
    elif problems := validate_request(request):
        raise ValueError("; ".join(problems))
    log(f"CSV: {request.csv_path} | features: {request.features} | target: {request.target}")

    df = load_data(request)
    split = make_split(df, request.target, config.TEST_SIZE, config.CV_FOLDS, config.RANDOM_STATE)

    log("Agent 1 (EDA): statistics, correlations, figures…")
    eda = await run_eda(df, request, output_dir)

    runs_spec: list[tuple[str, list[str]]] = [("all_features", list(request.features))]
    selection: FeatureSelection | None = None
    if eda.correlated_pairs:
        pairs = ", ".join(f"{p.feature_a}~{p.feature_b} (r={p.r})" for p in eda.correlated_pairs)
        log(f"Strongly correlated pairs: {pairs}")
        selection = select_uncorrelated_features(df, request.features, request.target, split,
                                                 config.CORR_THRESHOLD, config.CORR_METHOD)
        if selection.dropped:
            log(f"Second run will drop {[d.feature for d in selection.dropped]}")
            runs_spec.append(("uncorrelated_features", selection.kept))
        else:
            log("No feature needed dropping on the training rows; running one model.")
    else:
        log("No strongly correlated pairs; running one model.")

    log(f"Agent 2 (modeling): {len(runs_spec)} run(s)…")
    runs = list(await asyncio.gather(
        *(run_model(df, request, output_dir, split, name, feats) for name, feats in runs_spec)
    ))
    comparison = compare_runs(runs[0], runs[1], selection) if len(runs) == 2 else None

    log("Agent 3 (summary): writing the report…")
    final = (await summary_agent.run(
        _summary_payload(request, eda, selection, runs, comparison),
        model=config.LLM_MODEL,
        model_settings={"max_tokens": config.SUMMARY_MAX_TOKENS},
    )).output

    result = AnalysisResult(request=request, eda=eda, selection=selection, runs=runs,
                            comparison=comparison, final=final)
    (output_dir / "results.json").write_text(result.model_dump_json(indent=2))
    (output_dir / "report.md").write_text(final.markdown)
    log(f"Wrote {output_dir / 'report.md'} and {output_dir / 'results.json'}")
    return result
