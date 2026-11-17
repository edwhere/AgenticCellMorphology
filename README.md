# AgenticCellMorph

A pydantic-ai multi-agent system that analyses a CSV file and predicts a target
variable with logistic regression, then explains the predictions with SHAP.

```
prompt → parser agent → EDA agent → [correlated features?] → modeling agent ×1 or ×2 → summary agent
```

| Agent | Does | Output |
|---|---|---|
| Parser | Extracts the CSV path, feature list and target from free text; validates them against the file | `AnalysisRequest` |
| EDA | Statistics, target balance, correlation matrix, strongly correlated pairs (|r| ≥ 0.8), figures | `EDAReport` |
| Modeling | Stratified 80/20 split, 5-fold CV on the training rows, validation metrics, coefficients, SHAP | `ModelReport` |
| Summary | Interprets everything, compares the runs, explains the predictions | `FinalReport` |

The orchestrator (`orchestrator.py`) is plain Python. If the EDA finds correlated
pairs, it greedily drops features until none remain above the threshold (using
training rows only) and runs the modeling agent twice, with all features and with
the uncorrelated subset, on the **same split and CV folds**. It then computes
a paired comparison for the summary agent.

The agents decide which tools to call and interpret the results, but every
number comes from the tools (pandas, scikit-learn, shap). Tools store their
results in the run's deps and the orchestrator builds the reports from there,
so the LLM never transcribes metrics.

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=...
```

The model is set in one place: `LLM_MODEL` in `src/agenticcellmorph/config.py`
(default `anthropic:claude-haiku-5-5`). Thresholds, the split, the fold count and
the SHAP settings are set there too.

## Usage

```bash
uv run agenticcellmorph "Analyze data/sample.csv. Features: area, perimeter, nucleus_area, circularity, mean_intensity, texture_contrast, cell_line. Target: phenotype."
```

Skip the parser agent with explicit flags:

```bash
uv run agenticcellmorph --csv data/sample.csv --features area,perimeter,circularity --target phenotype
```

Other options: `--output-dir` (default `outputs/`) and `--model` (overrides `config.LLM_MODEL`).

Outputs in the output directory:
- `report.md`: the final report
- `results.json`: every report and metric
- `figures/`: distributions, correlation heatmap, features by target, and the SHAP importance and beeswarm plots for each run

`data/sample.csv` is synthetic cell-morphology data with a correlated group
(area, perimeter, nucleus_area). Regenerate it with `uv run python scripts/make_sample_data.py`.

## Tests

```bash
uv run pytest
```

The tests use pydantic-ai's `TestModel`, so they make no API calls.
