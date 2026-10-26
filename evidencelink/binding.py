"""Binding-cache generation for coverage-aware EvidenceLink selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

from evidencelink.artifacts import read_json_or_jsonl, read_jsonl
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


def chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: Sequence[Mapping[str, str]],
    timeout: float,
    max_retries: int = 5,
    retry_delay_seconds: float = 2.0,
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
    attempts = max(int(max_retries), 0) + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            last_error = exc
            if int(getattr(exc, "code", 0) or 0) not in {408, 409, 425, 429, 500, 502, 503, 504}:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < attempts - 1:
            time.sleep(float(retry_delay_seconds) * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("chat completion failed without an exception")


def llm_bindings(
    *,
    subquery: str,
    passage: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    max_retries: int,
    retry_delay_seconds: float,
    request_delay_seconds: float,
) -> list[str]:
    messages = [
        {"role": "system", "content": "Extract short entity/value bindings in the passage that support the evidence need. Return only a JSON array of strings."},
        {"role": "user", "content": f"Evidence need: {subquery}\nPassage: {passage[:1500]}"},
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
    if float(request_delay_seconds) > 0:
        time.sleep(float(request_delay_seconds))
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


def load_candidate_pool_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = read_json_or_jsonl(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        rows = payload.get("records", payload.get("rows", []))
        return [row for row in list(rows or []) if isinstance(row, Mapping)]
    return []


def write_cache(path: Path, cache: Mapping[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def build_binding_cache(args: argparse.Namespace) -> dict[str, list[str]]:
    pool_rows = load_candidate_pool_rows(args.candidate_pool)
    query_start = int(getattr(args, "query_start", 0) or 0)
    query_end = int(getattr(args, "query_end", 0) or 0)
    if query_start > 0 or query_end > 0:
        start = max(query_start, 0)
        end = query_end if query_end > 0 else len(pool_rows)
        pool_rows = pool_rows[start:end]
    need_rows = read_jsonl(args.evidence_needs)
    needs_by_query = {
        str(row.get("query_id", row.get("query_idx", ""))): [
            requirement_from_trace(item)
            for item in list(row.get("requirements") or row.get("B_q") or [])
            if isinstance(item, Mapping)
        ]
        for row in need_rows
    }
    output_path = Path(args.output)
    cache: dict[str, list[str]] = {}
    if bool(getattr(args, "resume", False)) and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping):
                cache.update({str(key): [str(item) for item in list(value or [])] for key, value in existing.items()})
        except json.JSONDecodeError:
            pass
    writes_since_flush = 0
    incremental_write_every = max(int(getattr(args, "incremental_write_every", 1) or 1), 1)
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
                if key in cache:
                    continue
                if str(getattr(args, "mode", "simple")) == "llm":
                    values = llm_bindings(
                        subquery=req.subquery,
                        passage=passage,
                        base_url=str(getattr(args, "llm_base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))),
                        model=str(args.binding_model),
                        api_key=str(getattr(args, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")),
                        timeout=float(getattr(args, "timeout", 120.0)),
                        max_retries=int(getattr(args, "max_retries", 5)),
                        retry_delay_seconds=float(getattr(args, "retry_delay_seconds", 2.0)),
                        request_delay_seconds=float(getattr(args, "request_delay_seconds", 0.0)),
                    )
                    if not values and bool(getattr(args, "fallback", False)):
                        values = simple_bindings(req.subquery, passage, str(candidate.get("title", "")))
                else:
                    values = simple_bindings(req.subquery, passage, str(candidate.get("title", "")))
                cache[key] = values
                writes_since_flush += 1
                if bool(getattr(args, "incremental_write", False)):
                    if writes_since_flush >= incremental_write_every:
                        write_cache(output_path, cache)
                        writes_since_flush = 0
    write_cache(output_path, cache)
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
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("EVLINK_BINDING_MAX_RETRIES", "5")))
    parser.add_argument("--retry-delay-seconds", type=float, default=float(os.environ.get("EVLINK_BINDING_RETRY_DELAY_SECONDS", "2.0")))
    parser.add_argument("--request-delay-seconds", type=float, default=float(os.environ.get("EVLINK_BINDING_REQUEST_DELAY_SECONDS", "0.5")))
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--incremental-write", action="store_true")
    parser.add_argument("--incremental-write-every", type=int, default=int(os.environ.get("EVLINK_BINDING_INCREMENTAL_WRITE_EVERY", "1")))
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cache = build_binding_cache(args)
    print(json.dumps({"output": str(args.output), "entry_count": len(cache)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
