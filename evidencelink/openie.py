"""OpenIE fact extraction entrypoint for EvLink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request

from evidencelink.artifacts import OpenIEFact, load_documents, write_jsonl


def simple_sentence_facts(doc_id: str, text: str) -> list[OpenIEFact]:
    """Deterministic fallback extractor for demos and tests."""
    facts: list[OpenIEFact] = []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]
    patterns = [
        re.compile(r"^(?P<s>.+?)\s+(?P<r>was born in|is in|is a|is an|flows into|is part of|is located in)\s+(?P<o>.+?)[.!]?$", re.I),
        re.compile(r"^(?P<s>.+?)\s+(?P<r>main tributary of|tributary of|branch of)\s+(?P<o>.+?)[.!]?$", re.I),
    ]
    for idx, sentence in enumerate(sentences):
        for pattern in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            facts.append(
                OpenIEFact(
                    doc_id=str(doc_id),
                    fact_id=f"{doc_id}:f{idx}",
                    subject=match.group("s").strip(),
                    relation=match.group("r").strip(),
                    object=match.group("o").strip(),
                    source_span=sentence,
                    raw_text=sentence,
                )
            )
            break
    return facts


def chat_completion(*, base_url: str, model: str, api_key: str, messages: Sequence[Mapping[str, str]], timeout: float) -> str:
    payload = json.dumps({"model": model, "messages": list(messages), "temperature": 0}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions" if not base_url.rstrip("/").endswith("/chat/completions") else base_url,
        data=payload,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def extract_json_array(text: str) -> list[Any]:
    start = str(text or "").find("[")
    end = str(text or "").rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(str(text)[start : end + 1])
    except json.JSONDecodeError:
        return []
    return list(payload) if isinstance(payload, list) else []


def llm_sentence_facts(*, doc_id: str, title: str, text: str, base_url: str, model: str, api_key: str, timeout: float) -> list[OpenIEFact]:
    messages = [
        {"role": "system", "content": "Extract source-grounded OpenIE triples. Return only a JSON array."},
        {
            "role": "user",
            "content": (
                "Return triples as objects with subject, relation, object, source_span. "
                f"Title: {title}\nPassage: {text[:3000]}"
            ),
        },
    ]
    response = chat_completion(base_url=base_url, model=model, api_key=api_key, messages=messages, timeout=timeout)
    facts: list[OpenIEFact] = []
    for idx, row in enumerate(extract_json_array(response)):
        if not isinstance(row, Mapping):
            continue
        facts.append(
            OpenIEFact(
                doc_id=str(doc_id),
                fact_id=f"{doc_id}:f{idx}",
                subject=str(row.get("subject") or ""),
                relation=str(row.get("relation") or ""),
                object=str(row.get("object") or ""),
                source_span=str(row.get("source_span") or ""),
                raw_text=json.dumps(row, ensure_ascii=False),
            )
        )
    return [fact for fact in facts if fact.subject and fact.object]


def build_openie_facts(args: argparse.Namespace) -> list[OpenIEFact]:
    docs = load_documents(args.corpus)
    all_facts: list[OpenIEFact] = []
    for doc in docs:
        if args.mode == "llm":
            facts = llm_sentence_facts(
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.text,
                base_url=str(args.llm_base_url),
                model=str(args.llm_model),
                api_key=str(args.api_key or os.environ.get("OPENAI_API_KEY", "")),
                timeout=float(args.timeout),
            )
            if not facts and args.fallback:
                facts = simple_sentence_facts(doc.doc_id, doc.text)
        else:
            facts = simple_sentence_facts(doc.doc_id, doc.text)
        all_facts.extend(facts)
    write_jsonl((fact.to_mapping() for fact in all_facts), args.output)
    return all_facts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build OpenIE fact artifacts for EvLink.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("simple", "llm"), default="simple")
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--fallback", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    facts = build_openie_facts(args)
    print(json.dumps({"output": str(args.output), "fact_count": len(facts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
