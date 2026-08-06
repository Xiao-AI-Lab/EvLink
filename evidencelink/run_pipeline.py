"""Run the public EvLink paper-facing pipeline end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencelink.api import PaperPipelineConfig, run_paper_pipeline
from evidencelink.contract import (
    DEFAULT_BINDING_MAX_CANDIDATES,
    DEFAULT_MIN_COVERAGE_GAIN,
    DEFAULT_MIN_SWAP_GAIN,
    DEFAULT_POOL_K,
    DEFAULT_READER_BUDGET_K,
    DEFAULT_STABILITY_WINDOW_M,
)


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Backward-compatible wrapper around :func:`run_paper_pipeline`."""

    return run_paper_pipeline(
        corpus_path=args.corpus,
        questions_path=args.questions,
        workdir=args.workdir,
        config=PaperPipelineConfig.from_namespace(args),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--top-k", "--reader-budget-k", dest="reader_budget_k", type=int, default=DEFAULT_READER_BUDGET_K)
    parser.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--stability-window-m", type=int, default=DEFAULT_STABILITY_WINDOW_M)
    parser.add_argument("--openie-mode", choices=("simple", "llm"), default="simple")
    parser.add_argument("--evidence-need-mode", choices=("llm", "whole_question", "anchor_list"), default="whole_question")
    parser.add_argument("--binding-mode", choices=("simple", "llm"), default="simple")
    parser.add_argument("--openie-model", default="gpt-4o-mini")
    parser.add_argument("--evidence-need-model", default="gpt-4o-mini")
    parser.add_argument("--binding-model", default="simple-binding")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--binding-max-candidates", type=int, default=DEFAULT_BINDING_MAX_CANDIDATES)
    parser.add_argument("--embedding-name", default="deterministic-hash")
    parser.add_argument("--embedding-base-url", default="offline")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--min-coverage-gain", type=float, default=DEFAULT_MIN_COVERAGE_GAIN)
    parser.add_argument("--min-swap-gain", type=float, default=DEFAULT_MIN_SWAP_GAIN)
    parser.add_argument("--allow-missing-requirements", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_pipeline(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
