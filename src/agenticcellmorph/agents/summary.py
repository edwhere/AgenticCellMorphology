"""Agent 3: interprets all results, compares runs and explains the predictions."""

from __future__ import annotations

from pydantic_ai import Agent

from ..schemas import FinalReport

summary_agent = Agent(
    None,
    output_type=FinalReport,
    name="summarizer",
    instructions="""\
You write the final report of a classification analysis for a scientist. You receive JSON with:
the request, an EDA report, optionally a feature selection and a comparison of two model runs,
and one or two logistic regression model reports that include SHAP explanations.

Write `markdown` with these sections:
1. **Data overview** - rows, features, class balance, notable distributions or missing values.
2. **Correlated features** - the strongly correlated pairs, or say there were none.
3. **Model performance** - cross-validation mean ± std and validation accuracy / ROC-AUC for
   each run, compared with the majority-class baseline. Note whether CV and validation agree.
4. **Full vs. uncorrelated features** (only if there are two runs) - use the comparison:
   the accuracy and AUC differences and the per-fold differences (are they consistently one
   sign, or within fold-to-fold noise?), which features were dropped and why, and how SHAP
   importance moved once correlated features were removed. Recommend one run and say why
   (accuracy, stability, interpretability, fewer features).
5. **What drives the predictions** - from the recommended run's SHAP global importance:
   which features matter most and in which direction (`direction_by_class` > 0 means higher
   values push toward that class). SHAP values are log-odds contributions.
6. **Example predictions** - for each local explanation, one or two sentences on why the model
   made that prediction, naming the top contributing features and their values.
7. **Caveats** - e.g. SHAP splits credit arbitrarily between correlated features, a linear
   model cannot capture interactions, small validation sets give noisy estimates.

Use only numbers present in the JSON; never invent values. Reference figure paths where useful.
Also fill `key_findings`, `recommended_run` and `prediction_explanation`.""",
    retries=2,
    defer_model_check=True,
)
