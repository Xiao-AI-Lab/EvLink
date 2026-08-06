#!/usr/bin/env python3
"""Build naive requirement reports for EvLink ablations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


QUESTION_WORDS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = re.sub(r"\b([A-Za-z0-9.'’ -]+)'s\b", r"\1", str(item or ""))
        clean = re.sub(r"\s+", " ", clean.strip(" ,.;:?!'\""))
        key = normalize(clean)
        if not clean or not key or key in seen:
            continue
        if key in QUESTION_WORDS or key.split(" ", 1)[0] in QUESTION_WORDS:
            continue
        if len(key) < 3:
            continue
        output.append(clean)
        seen.add(key)
    return output


def split_question(question: str) -> list[str]:
    base = str(question or "").strip()
    if not base:
        return []
    clauses: list[str] = []
    seen: set[str] = set()
    base_clean = base.strip(" ,.;:?!")
    if base_clean:
        clauses.append(base_clean)
        seen.add(normalize(base_clean))
    for part in re.split(r"\s+(?:and|or|then|after|before|while)\s+|[,;]", base):
        clean = part.strip(" ,.;:?!")
        key = normalize(clean)
        if len(clean.split()) >= 4 and key and key not in seen:
            clauses.append(clean)
            seen.add(key)
    return clauses[:4]


def regex_anchors(question: str) -> list[str]:
    spans: list[str] = []
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'’.-]*|[0-9]+|[IVX]{2,})"
        r"(?:\s+(?:[A-Z][A-Za-z0-9'’.-]*|[0-9]+|[IVX]{2,}|of|Of|the|The|and|And|de|De|von|Von|van|Van))*"
    )
    for match in pattern.finditer(str(question or "")):
        spans.append(match.group(0))
    return dedupe(spans)


def title_anchors(question: str, pool_titles: list[str]) -> list[str]:
    q_norm = f" {normalize(question)} "
    anchors: list[str] = []
    for title in pool_titles[:100]:
        base = str(title or "").split("(", 1)[0].strip()
        key = normalize(base)
        if not key or len(key) < 3:
            continue
        if f" {key} " in q_norm:
            anchors.append(base)
    return dedupe(anchors)


def build_whole_question_requirements(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    question = str(record.get("question") or "").strip()
    if not question:
        return []
    return [
        {
            "unit_id": "q0",
            "subquery": question,
            "depends_on": [],
            "expected_answer_type": "unknown",
            "anchor_mentions": [],
            "role": "query",
            "satisfiable_by": "unknown",
        }
    ]


def build_simple_question_anchor_requirements(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    question = str(record.get("question") or "").strip()
    anchors = dedupe(regex_anchors(question) + title_anchors(question, list(record.get("pool_titles") or [])))[:6]
    requirements: list[dict[str, Any]] = []
    for idx, clause in enumerate(split_question(question)):
        requirements.append(
            {
                "unit_id": f"q{idx}",
                "subquery": clause,
                "depends_on": [],
                "expected_answer_type": "unknown",
                "anchor_mentions": anchors,
                "role": "query" if idx == 0 else "clause",
                "satisfiable_by": "unknown",
            }
        )
    for idx, anchor in enumerate(anchors):
        requirements.append(
            {
                "unit_id": f"a{idx}",
                "subquery": anchor,
                "depends_on": [],
                "expected_answer_type": "entity",
                "anchor_mentions": [anchor],
                "role": "anchor",
                "satisfiable_by": "document",
            }
        )
    return requirements or [
        {
            "unit_id": "q0",
            "subquery": question,
            "depends_on": [],
            "expected_answer_type": "unknown",
            "anchor_mentions": anchors,
            "role": "query",
            "satisfiable_by": "unknown",
        }
    ]


def build_requirements(record: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "whole_question":
        return build_whole_question_requirements(record)
    if mode == "simple_question_anchors":
        return build_simple_question_anchor_requirements(record)
    raise ValueError(f"Unsupported requirement mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--binding-cache-json", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument(
        "--mode",
        choices=("whole_question", "simple_question_anchors"),
        default="simple_question_anchors",
        help=(
            "Naive requirement construction mode. whole_question emits one "
            "requirement per query; simple_question_anchors emits split clauses "
            "plus regex/title anchor requirements."
        ),
    )
    args = parser.parse_args()

    pool_payload = json.loads(args.pool_json.read_text(encoding="utf-8"))
    traces = []
    requirement_counts: list[int] = []
    anchor_counts: list[int] = []
    for record in list(pool_payload.get("records") or []):
        requirements = build_requirements(record, args.mode)
        requirement_counts.append(len(requirements))
        anchor_counts.append(sum(1 for req in requirements if str(req.get("role")) == "anchor"))
        traces.append(
            {
                "question": str(record.get("question") or ""),
                "query_index": int(record.get("query_idx", record.get("query_index", len(traces))) or 0),
                "selection_trace": {
                    "selection": args.mode,
                    "requirements": requirements,
                },
            }
        )

    payload = {
        "mode": args.mode,
        "dataset": str(args.dataset or pool_payload.get("dataset") or ""),
        "pool_json": str(args.pool_json),
        "summary": {
            "query_count": len(traces),
            "avg_requirement_count": sum(requirement_counts) / max(len(requirement_counts), 1),
            "avg_anchor_requirement_count": sum(anchor_counts) / max(len(anchor_counts), 1),
        },
        "evidence_selection_query_traces": traces,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.binding_cache_json.parent.mkdir(parents=True, exist_ok=True)
    args.binding_cache_json.write_text("{}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
