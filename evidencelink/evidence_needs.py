"""Evidence-need mining for EvidenceLink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
import urllib.request

from evidencelink.anchors import extract_question_anchors
from evidencelink.artifacts import load_questions, write_jsonl
from evidencelink.evidence_need_utils import (
    EvidenceNeed,
    build_fallback_evidence_needs,
    parse_evidence_need_response,
)


def requirement_to_mapping(requirement: EvidenceNeed) -> dict[str, object]:
    return requirement.to_trace()


def simple_evidence_needs(question: str, *, mode: str) -> list[EvidenceNeed]:
    if mode == "anchor_list":
        anchors = extract_question_anchors(question)
        if anchors:
            return [
                EvidenceNeed(
                    unit_id=f"a{idx + 1}",
                    subquery=anchor,
                    anchor_mentions=(anchor,),
                    role="support",
                    satisfiable_by="document",
                )
                for idx, anchor in enumerate(anchors)
            ]
    return build_fallback_evidence_needs(question)


def chat_completion(*, base_url: str, model: str, api_key: str, messages: Sequence[Mapping[str, str]], timeout: float) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = json.dumps({"model": model, "messages": list(messages), "temperature": 0}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def mine_evidence_needs_for_question(*, question: str, base_url: str, model: str, api_key: str, timeout: float, max_steps: int) -> tuple[list[EvidenceNeed], dict[str, object]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Decompose a multi-hop question into evidence needs. Return only JSON with a 'requirements' array. "
                "Each item must contain id, subquery, anchor_mentions, expected_answer_type, role, satisfiable_by, depends_on."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    response = chat_completion(base_url=base_url, model=model, api_key=api_key, messages=messages, timeout=timeout)
    requirements, trace = parse_evidence_need_response(response, max_steps=max_steps)
    if not requirements:
        requirements = build_fallback_evidence_needs(question)
        trace["fallback_used"] = True
    else:
        trace["fallback_used"] = False
    return requirements, trace


def build_evidence_need_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for question in load_questions(args.questions):
        if args.mode in {"whole_question", "anchor_list"}:
            requirements = simple_evidence_needs(question.question, mode=args.mode)
            trace = {"provider": args.mode, "fallback_used": False}
        else:
            requirements, trace = mine_evidence_needs_for_question(
                question=question.question,
                base_url=str(args.llm_base_url),
                model=str(args.llm_model),
                api_key=str(args.api_key or os.environ.get("OPENAI_API_KEY", "")),
                timeout=float(args.timeout),
                max_steps=int(args.max_steps),
            )
        rows.append(
            {
                "artifact_type": "evidence_needs",
                "query_id": question.query_id,
                "question": question.question,
                "B_q": [requirement_to_mapping(req) for req in requirements],
                "requirements": [requirement_to_mapping(req) for req in requirements],
                "trace": trace,
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EvidenceLink evidence needs B(q).")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("llm", "whole_question", "anchor_list"), default="llm")
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rows = build_evidence_need_rows(args)
    write_jsonl(rows, args.output)
    print(json.dumps({"output": str(args.output), "query_count": len(rows), "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
