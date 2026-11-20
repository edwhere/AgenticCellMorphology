"""Agentic CSV analysis: EDA, logistic regression with SHAP, and a written summary."""

from __future__ import annotations

import os

# Remove the banner from Pydantic runs
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sys  # noqa: E402

from . import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agenticcellmorph",
        description="Analyse a CSV file and predict a target with logistic regression.",
        epilog='Example: agenticcellmorph "Use data/sample.csv. Features: area, perimeter, '
        'circularity. Target: phenotype."',
    )
    parser.add_argument("prompt", nargs="?", help="Free-text request (read from stdin if omitted).")
    parser.add_argument("--csv", help="CSV path (skips the request-parser agent).")
    parser.add_argument("--features", help="Comma-separated feature columns (with --csv).")
    parser.add_argument("--target", help="Target column (with --csv).")
    parser.add_argument("--output-dir", default="outputs", help="Where to write figures and reports.")
    parser.add_argument("--model", help=f"Override config.LLM_MODEL (default {config.LLM_MODEL}).")
    args = parser.parse_args()

    if args.model:
        config.LLM_MODEL = args.model

    from .orchestrator import run_analysis
    from .schemas import AnalysisRequest

    request = None
    prompt = None
    if args.csv:
        if not (args.features and args.target):
            parser.error("--csv needs --features and --target")
        request = AnalysisRequest(
            csv_path=args.csv,
            features=[f.strip() for f in args.features.split(",") if f.strip()],
            target=args.target,
        )
    else:
        prompt = args.prompt or (sys.stdin.read() if not sys.stdin.isatty() else None)
        if not prompt:
            parser.error("give a prompt, pipe one on stdin, or use --csv/--features/--target")

    try:
        result = asyncio.run(run_analysis(prompt, request=request, output_dir=args.output_dir))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(result.final.markdown)
