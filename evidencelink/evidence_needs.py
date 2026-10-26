"""Evidence-need mining for EvidenceLink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence
import urllib.error
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


def qwen_no_think_enabled(model: str) -> bool:
    flag = str(os.environ.get("YUNWU_QWEN_NO_THINK", "")).strip().lower()
    return "qwen" in str(model).lower() and flag in {"1", "true", "yes", "y", "on"}


def with_qwen_no_think_messages(messages: Sequence[Mapping[str, str]], *, model: str) -> list[dict[str, str]]:
    patched: list[dict[str, str]] = []
    prefix = str(os.environ.get("YUNWU_QWEN_NO_THINK_PREFIX", "/no_think") or "/no_think")
    for message in messages:
        item = dict(message)
        content = str(item.get("content") or "")
        if str(item.get("role", "")).lower() == "user" and qwen_no_think_enabled(model) and prefix not in content[:128]:
            item["content"] = f"{prefix}\n{content}"
        patched.append(item)
    return patched


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


def chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: Sequence[Mapping[str, str]],
    timeout: float,
    max_retries: int,
    retry_delay_seconds: float,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload_obj: dict[str, object] = {
        "model": model,
        "messages": with_qwen_no_think_messages(messages, model=model),
        "temperature": 0,
    }
    if qwen_no_think_enabled(model):
        payload_obj["enable_thinking"] = False
    payload = json.dumps(payload_obj).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max(int(max_retries), 1)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt + 1 >= max(int(max_retries), 1):
                raise
            time.sleep(float(retry_delay_seconds) * float(attempt + 1))
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt + 1 >= max(int(max_retries), 1):
                raise
            time.sleep(float(retry_delay_seconds) * float(attempt + 1))
    else:
        raise RuntimeError("chat completion failed without response") from last_error
    return str(body["choices"][0]["message"]["content"])


def mine_evidence_needs_for_question(
    *,
    question: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    max_steps: int,
    max_retries: int,
    retry_delay_seconds: float,
) -> tuple[list[EvidenceNeed], dict[str, object]]:
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
    response = chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        messages=messages,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    requirements, trace = parse_evidence_need_response(response, max_steps=max_steps)
    if not requirements:
        requirements = build_fallback_evidence_needs(question)
        trace["fallback_used"] = True
    else:
        trace["fallback_used"] = False
    return requirements, trace


def build_evidence_need_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    existing_by_query_id: dict[str, dict[str, object]] = {}
    output_path = Path(args.output)
    if bool(getattr(args, "resume", False)) and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if not clean:
                continue
            try:
                row = json.loads(clean)
            except json.JSONDecodeError:
                continue
            query_id = str(row.get("query_id", ""))
            if query_id:
                existing_by_query_id[query_id] = row
    questions = load_questions(args.questions)
    max_queries = int(getattr(args, "max_queries", 0) or 0)
    if max_queries > 0:
        questions = questions[:max_queries]
    for question in questions:
        if question.query_id in existing_by_query_id:
            rows.append(existing_by_query_id[question.query_id])
            continue
        mode = str(getattr(args, "mode", "llm"))
        if mode in {"whole_question", "anchor_list"}:
            requirements = simple_evidence_needs(question.question, mode=mode)
            trace = {"provider": mode, "fallback_used": False}
        else:
            requirements, trace = mine_evidence_needs_for_question(
                question=question.question,
                base_url=str(args.llm_base_url),
                model=str(args.llm_model),
                api_key=str(args.api_key or os.environ.get("OPENAI_API_KEY", "")),
                timeout=float(args.timeout),
                max_steps=int(args.max_steps),
                max_retries=int(args.max_retries),
                retry_delay_seconds=float(args.retry_delay_seconds),
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
        if bool(getattr(args, "incremental_write", False)):
            write_jsonl(rows, output_path)
        request_delay_seconds = float(getattr(args, "request_delay_seconds", 0.0) or 0.0)
        if request_delay_seconds > 0.0:
            time.sleep(request_delay_seconds)
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
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--incremental-write", action="store_true")
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
