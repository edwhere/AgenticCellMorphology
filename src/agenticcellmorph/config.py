"""Global settings for the analysis system.

Change ``LLM_MODEL`` to switch every agent to a different provider/model.
Any pydantic-ai model string works, e.g. ``"anthropic:claude-sonnet-5-5"`` or
``"openai:gpt-5"``. The value is read each time an agent runs, so it can also
be changed at runtime (the CLI's ``--model`` flag does this).
"""

from __future__ import annotations

LLM_MODEL: str = "anthropic:claude-haiku-5-5"

# Correlation analysis
CORR_METHOD: str = "pearson"  # "pearson" or "spearman"
CORR_THRESHOLD: float = 0.8  # |r| at or above this counts as "strongly correlated"

# Modeling
TEST_SIZE: float = 0.2  # fraction held out for validation
CV_FOLDS: int = 5
RANDOM_STATE: int = 42
MAX_ITER: int = 2000  # LogisticRegression solver iterations

# Explainability
SHAP_BACKGROUND_SAMPLES: int = 500
N_LOCAL_EXPLANATIONS: int = 4
TOP_CONTRIBUTIONS: int = 5

# Max output tokens for the agent that writes the final report
SUMMARY_MAX_TOKENS: int = 16000
