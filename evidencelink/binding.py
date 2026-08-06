"""Binding-cache generation for coverage-aware EvidenceLink selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request

from evidencelink.artifacts import read_jsonl
from evidencelink.contract import DEFAULT_BINDING_MAX_CANDIDATES
from evidencelink.requirements import requirement_from_trace
from evidencelink.utils.hash import compute_mdhash_id


def binding_cache_key(*, model: str, subquery: str, passage: str) -> str:
    payload = {
        "version": "coverage_llm_binding_v1",
        "model": str(model),
        "subquery": str(subquery),
        "passage": str(passage[:1500]),
    }
    return compute_mdhash_id(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def simple_bindings(subquery: str, passage: str, title: str = "") -> list[str]:
    haystack = f"{title}\n{passage}"
    bindings: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5}\b", haystack):
        text = match.group(0).strip()
        if text and text not in bindings:
            bindings.append(text)
    for token in re.findall(r"\b[a-zA-Z0-9]{4,}\b", subquery):
        if re.search(re.escape(token), haystack, re.I) and token not in bindings:
            bindings.append(token)
    return bindings[:8]


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


def llm_bindings(*, subquery: str, passage: str, base_url: str, model: str, api_key: str, timeout: float) -> list[str]:
    messages = [
        {"role": "system", "content": "Extract short entity/value bindings in the passage that support the evidence need. Return only a JSON array of strings."},
        {"role": "user", "content": f"Evidence need: {subquery}\nPassage: {passage[:1500]}"},
    ]
    response = chat_completion(base_url=base_url, model=model, api_key=api_key, messages=messages, timeout=timeout)
    start = response.find("[")
    end = response.rfind("]")
    if start >= 0 and end > start:
        try:
            payload = json.loads(response[start : end + 1])
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return []


def build_binding_cache(args: argparse.Namespace) -> dict[str, list[str]]:
    pool_rows = read_jsonl(args.candidate_pool)
    need_rows = read_jsonl(args.evidence_needs)
    needs_by_query = {
        str(row.get("query_id", row.get("query_idx", ""))): [
            requirement_from_trace(item)
            for item in list(row.get("requirements") or row.get("B_q") or [])
            if isinstance(item, Mapping)
        ]
        for row in need_rows
    }
    cache: dict[str, list[str]] = {}
    for pool_row in pool_rows:
        query_id = str(pool_row.get("query_id", pool_row.get("query_idx", "")))
        requirements = needs_by_query.get(query_id, [])
        candidates = list(pool_row.get("candidate_pool") or [])
        if not candidates:
            docs = list(pool_row.get("pool_docs") or [])
            titles = list(pool_row.get("pool_titles") or [])
            candidates = [{"text": str(doc).split("\n", 1)[1] if "\n" in str(doc) else str(doc), "title": titles[idx] if idx < len(titles) else ""} for idx, doc in enumerate(docs)]
        for req in requirements:
            for candidate in candidates[: int(args.max_candidates)]:
                if not isinstance(candidate, Mapping):
                    continue
                passage = f"{candidate.get('title', '')}\n{candidate.get('text', '')}".strip()
                key = binding_cache_key(model=str(args.binding_model), subquery=req.subquery, passage=passage)
                if args.mode == "llm":
                    values = llm_bindings(
                        subquery=req.subquery,
                        passage=passage,
                        base_url=str(args.llm_base_url),
                        model=str(args.binding_model),
                        api_key=str(args.api_key or os.environ.get("OPENAI_API_KEY", "")),
                        timeout=float(args.timeout),
                    )
                    if not values and args.fallback:
                        values = simple_bindings(req.subquery, passage, str(candidate.get("title", "")))
                else:
                    values = simple_bindings(req.subquery, passage, str(candidate.get("title", "")))
                cache[key] = values
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an EvidenceLink binding cache for coverage-aware selection.")
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--evidence-needs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("simple", "llm"), default="simple")
    parser.add_argument("--binding-model", default="simple-binding")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_BINDING_MAX_CANDIDATES)
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--fallback", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cache = build_binding_cache(args)
    print(json.dumps({"output": str(args.output), "entry_count": len(cache)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
