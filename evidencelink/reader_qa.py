#!/usr/bin/env python3
"""Run a lightweight OpenAI-compatible reader over EvidenceLink top-5 reports."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from evidencelink.utils.text import normalize_answer


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gold) for gold in gold_answers))


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = {}
        for token in pred_tokens:
            common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def build_prompt(question: str, passages: Sequence[str]) -> list[dict[str, str]]:
    context = []
    for idx, passage in enumerate(passages, start=1):
        context.append(f"[{idx}] {str(passage).strip()}")
    user = (
        "Answer the question using only the provided Wikipedia passages. "
        "Return a concise answer after 'Answer:'.\n\n"
        + "\n\n".join(context)
        + f"\n\nQuestion: {question}\nThought: "
    )
    return [
        {"role": "system", "content": "You are a careful question-answering assistant."},
        {"role": "user", "content": user},
    ]


def chat_completion(*, base_url: str, model: str, api_key: str, messages: Sequence[Mapping[str, str]], timeout: float) -> str:
    payload = json.dumps({"model": model, "messages": list(messages), "temperature": 0}).encode("utf-8")
    request = urllib.request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def extract_answer(response: str) -> str:
    match = re.search(r"Answer:\s*(.*)", str(response), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else str(response).strip()


def run_reader(args: argparse.Namespace) -> dict[str, Any]:
    payload = read_json(args.retrieval_report)
    rows = list(payload.get("rows") or [])
    if int(args.max_queries) > 0:
        rows = rows[: int(args.max_queries)]
    outputs = []
    for idx, row in enumerate(rows):
        if idx % 10 == 0:
            print(f"[reader] infer {idx + 1}/{len(rows)}", flush=True)
        passages = list(row.get("retrieved_docs_top5") or [])[: int(args.qa_top_k)]
        response = chat_completion(
            base_url=str(args.llm_base_url),
            model=str(args.llm_name),
            api_key=str(args.api_key or os.environ.get("OPENAI_API_KEY", "")),
            messages=build_prompt(str(row.get("question") or ""), passages),
            timeout=float(args.timeout),
        )
        answer = extract_answer(response)
        gold_answers = [str(item) for item in list(row.get("gold_answers") or [])]
        outputs.append(
            {
                "query_index": int(row.get("query_index", idx)),
                "question": str(row.get("question") or ""),
                "gold_answers": gold_answers,
                "prediction": answer,
                "em": exact_match(answer, gold_answers),
                "f1": token_f1(answer, gold_answers),
                "retrieved_titles_top5": list(row.get("retrieved_titles_top5") or [])[: int(args.qa_top_k)],
            }
        )
    return {
        "method": str(payload.get("method") or "evidencelink"),
        "dataset": str(payload.get("dataset") or ""),
        "reader": {"llm_name": str(args.llm_name), "qa_top_k": int(args.qa_top_k)},
        "summary": {
            "count": len(outputs),
            "em": round(mean(float(row["em"]) for row in outputs), 6) if outputs else 0.0,
            "f1": round(mean(float(row["f1"]) for row in outputs), 6) if outputs else 0.0,
        },
        "rows": outputs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--llm-name", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_reader(args)
    write_json(payload, args.output_json)
    print(json.dumps({"output_json": str(args.output_json), **payload["summary"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
