#!/usr/bin/env python3
"""Run EvLink coverage-aware evidence selection over an exported pool.

The runner expects an exported candidate pool, a frozen evidence-need report,
and a frozen binding cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidencelink.contract import (  # noqa: E402
    DEFAULT_BINDING_MAX_CANDIDATES,
    DEFAULT_POOL_K,
    DEFAULT_STABILITY_WINDOW_M,
    DEFAULT_READER_BUDGET_K,
    DEFAULT_MIN_COVERAGE_GAIN,
    DEFAULT_MIN_SWAP_GAIN,
    METHOD_CONTRACT,
    METHOD_NAME,
    MAIN_UPSTREAM_RETRIEVER_NAME,
    PAPER_FACING_METHOD_NAME,
    is_main_evidencelink_pool,
    pool_upstream_retriever,
)
from evidencelink.coverage_utility import (  # noqa: E402
    CoverageUtilityProvider,
    load_binding_cache,
)
from evidencelink.artifacts import read_json_or_jsonl  # noqa: E402
from evidencelink.evidence_selection import compose_evidence_selection  # noqa: E402
from evidencelink.io_utils import (  # noqa: E402
    title_all_covered,
    title_recall,
    write_json,
)
from evidencelink.requirements import FrozenReportRequirementProvider  # noqa: E402
from evidencelink.types import EvidenceQueryState  # noqa: E402
from evidencelink.embedding import DeterministicHashEmbeddingClient, OpenAIEmbeddingClient  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("run_logs/evidencelink_selection")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pool_payload(path: str | Path) -> dict[str, Any]:
    payload = read_json_or_jsonl(path)
    if isinstance(payload, list):
        return {"artifact_type": "candidate_pool", "records": payload}
    return dict(payload or {})


def extract_title(doc_text: str) -> str:
    return str(doc_text or "").split("\n", 1)[0].strip()


def default_output_json(
    output_root: Path,
    dataset: str,
    stability_window_m: int,
    reader_budget_k: int,
    pool_k: int,
    limit: int,
) -> Path:
    _ = int(stability_window_m)
    return output_root / "evals" / f"{dataset}_evidence_selection_budget{reader_budget_k}_pool{pool_k}_limit{limit}.json"


def source_retrieval_payload(pool_payload: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_report = str(pool_payload.get("retrieval_report") or "").strip()
    if not retrieval_report:
        return {}
    path = Path(retrieval_report)
    if not path.exists():
        return {}
    return dict(read_json(path))


def row_by_query_index(source_payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = {}
    for row in list(source_payload.get("rows") or []):
        if not isinstance(row, Mapping):
            continue
        try:
            query_idx = int(row.get("query_index", row.get("query_idx")))
        except (TypeError, ValueError):
            continue
        rows[query_idx] = row
    return rows


def build_query_trace(
    *,
    state: EvidenceQueryState,
    result: Any,
) -> dict[str, Any]:
    baseline_titles = list(state.pool_titles[: int(state.reader_budget_k)])
    final_titles = list(result.final_titles)
    return {
        "query_id": state.query_id,
        "query_index": int(state.query_idx),
        "question": state.question,
        "gold_answers": list(state.gold_answers),
        "gold_titles": list(state.gold_titles),
        "baseline_top_titles": baseline_titles,
        "selected_top_titles": final_titles,
        "changed_from_baseline": bool(baseline_titles != final_titles),
        "selection_trace": dict(result.raw_selection_trace),
        "evidence_selection": dict(result.trace),
    }


def build_report_row(
    *,
    state: EvidenceQueryState,
    result: Any,
    source_row: Mapping[str, Any],
    requirement_count: int,
) -> dict[str, Any]:
    reader_doc_ids_source = list(
        (state.pool_trace or {}).get("reader_pool_doc_ids")
        or (state.pool_trace or {}).get("external_pool_doc_ids")
        or []
    )
    if len(reader_doc_ids_source) < len(state.pool_doc_ids):
        reader_doc_ids_source = list(state.pool_doc_ids)
    final_doc_ids = [
        reader_doc_ids_source[pos]
        for pos in result.final_positions
        if pos < len(reader_doc_ids_source) and reader_doc_ids_source[pos] is not None
    ]
    baseline_titles = list(state.pool_titles[: int(state.reader_budget_k)])
    final_titles = list(result.final_titles)
    return {
        "query_id": state.query_id,
        "query_index": int(state.query_idx),
        "question": state.question,
        "gold_answers": list(state.gold_answers),
        "gold_docs": list(state.gold_docs),
        "gold_titles": list(state.gold_titles),
        "gold_doc_indices": list(source_row.get("gold_doc_indices") or []),
        "retrieved_doc_indices_top5": final_doc_ids,
        "retrieved_docs_top5": list(getattr(result, "final_docs", ())),
        "retrieved_titles_top5": final_titles,
        "baseline_titles_top5": baseline_titles,
        "title_recall_at5": title_recall(list(state.gold_titles), final_titles, k=int(state.reader_budget_k)),
        "title_all_gold_at5": title_all_covered(list(state.gold_titles), final_titles, k=int(state.reader_budget_k)),
        "evidence_need_count": int(requirement_count),
        "evidence_selection": dict(result.trace),
        "raw_selection_trace": dict(result.raw_selection_trace),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]], traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = len(rows)
    return {
        "count": int(count),
        "changed_count": int(sum(bool(trace.get("changed_from_baseline")) for trace in traces)),
        "title_recall_top5": round(mean(float(row.get("title_recall_at5", 0.0)) for row in rows), 4),
        "evidence_selection_title_all_gold_top5": round(
            sum(bool(row.get("title_all_gold_at5")) for row in rows) / count, 4
        ),
        "rank_stability_held_count": int(
            sum(bool((row.get("evidence_selection") or {}).get("rank_stability_held")) for row in rows)
        ),
        "admit_count": int(
            sum(str((row.get("evidence_selection") or {}).get("decision")) == "admit" for row in rows)
        ),
    }


def run_evidence_selection(args: argparse.Namespace) -> dict[str, Any]:
    pool_payload = load_pool_payload(args.pool_json)
    records = list(pool_payload.get("records") or [])
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]

    source_payload = source_retrieval_payload(pool_payload)
    source_rows = row_by_query_index(source_payload)
    if str(args.embedding_name).lower() in {"deterministic-hash", "offline"}:
        embedding_client = DeterministicHashEmbeddingClient()
    else:
        embedding_client = OpenAIEmbeddingClient(
            base_url=str(args.embedding_base_url),
            model=str(args.embedding_name),
            batch_size=int(args.embedding_batch_size),
            api_key=str(args.embedding_api_key or os.environ.get("OPENAI_API_KEY", "")),
            timeout=float(args.embedding_timeout),
        )

    requirement_provider = FrozenReportRequirementProvider(
        args.requirement_report,
        allow_missing=bool(args.allow_missing_requirements),
    )
    utility_provider = CoverageUtilityProvider(
        embedding_client=embedding_client,
        binding_cache=load_binding_cache(args.binding_cache_path),
        binding_model=str(args.llm_binding_model or "qwen3-32b-judge"),
        binding_top_m=int(args.binding_max_candidates),
        min_objective_gain=float(args.min_coverage_gain),
        min_swap_gain=float(args.min_swap_gain),
        llm_binding_title_match_mode=str(args.llm_binding_title_match_mode),
    )

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    missing_requirement_count = 0
    for record in records:
        state = EvidenceQueryState.from_pool_record(
            record,
            dataset=str(args.dataset),
            reader_budget_k=int(args.reader_budget_k),
            stability_window_m=int(args.stability_window_m),
        )
        requirements = requirement_provider.get_requirements(state)
        if not requirements:
            missing_requirement_count += 1
        result = compose_evidence_selection(
            state,
            requirements=requirements,
            utility_provider=utility_provider,
        )
        source_row = source_rows.get(int(state.query_idx), {})
        rows.append(
            build_report_row(
                state=state,
                result=result,
                source_row=source_row,
                requirement_count=len(requirements),
            )
        )
        traces.append(build_query_trace(state=state, result=result))

    retrieval_report = str(pool_payload.get("retrieval_report") or "")
    openie_path = str(pool_payload.get("openie_path") or source_payload.get("openie_path") or "")
    input_report = str(source_payload.get("input_report") or "")
    upstream_retriever = pool_upstream_retriever(pool_payload)
    is_main_protocol = is_main_evidencelink_pool(pool_payload)
    return {
        "method": METHOD_NAME,
        "paper_facing_method": PAPER_FACING_METHOD_NAME,
        "mode": "coverage_aware_evidence_selection",
        "pool_protocol": {
            "role": "main_evidencelink" if is_main_protocol else "compatibility_or_ablation",
            "is_main_table_evidencelink": bool(is_main_protocol),
            "upstream_retriever": upstream_retriever,
            "expected_main_upstream_retriever": MAIN_UPSTREAM_RETRIEVER_NAME,
            "pool_provenance_key": "retrieval.input_method",
        },
        "dataset": str(args.dataset),
        "limit": int(len(records)),
        "pool_json": str(args.pool_json),
        "requirement_report": str(args.requirement_report),
        "binding_cache_path": str(args.binding_cache_path),
        "retrieval_report": retrieval_report,
        "openie_path": openie_path,
        "input_report": input_report,
        "method_contract": dict(METHOD_CONTRACT),
        "evidence_selection_config": {
            "reader_budget_k": int(args.reader_budget_k),
            "stability_window_m": int(args.stability_window_m),
            "pool_k": int(args.pool_k),
            "min_coverage_gain": float(args.min_coverage_gain),
            "min_swap_gain": float(args.min_swap_gain),
        },
        "requirement_provider": {
            **requirement_provider.summary(),
            "missing_requirement_count": int(missing_requirement_count),
        },
        "utility_provider": utility_provider.summary(),
        "summary": summarize_rows(rows, traces),
        "rows": rows,
        "evidence_selection_query_traces": traces,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K)
    parser.add_argument("--reader-budget-k", type=int, default=DEFAULT_READER_BUDGET_K)
    parser.add_argument("--stability-window-m", type=int, default=DEFAULT_STABILITY_WINDOW_M)
    parser.add_argument("--pool-json", type=Path, default=None)
    parser.add_argument("--requirement-report", type=Path, default=None)
    parser.add_argument("--binding-cache-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--llm-binding-model", default="")
    parser.add_argument("--embedding-name", default="deterministic-hash")
    parser.add_argument("--embedding-base-url", default="offline")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    parser.add_argument("--binding-max-candidates", type=int, default=DEFAULT_BINDING_MAX_CANDIDATES)
    parser.add_argument("--llm-binding-title-match-mode", default="wiki_title")
    parser.add_argument("--min-coverage-gain", type=float, default=DEFAULT_MIN_COVERAGE_GAIN)
    parser.add_argument("--min-swap-gain", type=float, default=DEFAULT_MIN_SWAP_GAIN)
    parser.add_argument("--allow-missing-requirements", action="store_true")
    return parser


def resolve_default_paths(args: argparse.Namespace) -> None:
    if args.pool_json is None:
        raise ValueError("--pool-json is required")
    if args.requirement_report is None:
        raise ValueError("--requirement-report is required")
    if args.binding_cache_path is None:
        raise ValueError("--binding-cache-path is required")
    if args.output_json is None:
        args.output_json = default_output_json(
            Path(args.output_root),
            str(args.dataset),
            int(args.stability_window_m),
            int(args.reader_budget_k),
            int(args.pool_k),
            int(args.limit),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolve_default_paths(args)
    payload = run_evidence_selection(args)
    write_json(payload, args.output_json)
    summary = payload.get("summary", {})
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "dataset": payload.get("dataset"),
                "count": summary.get("count"),
                "changed": summary.get("changed_count"),
                "title_all": summary.get("evidence_selection_title_all_gold_top5"),
                "binding_misses": payload.get("utility_provider", {}).get("misses"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
