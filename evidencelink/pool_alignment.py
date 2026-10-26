"""Pool document alignment used by EvidenceLink evidence selection."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from evidencelink.utils.hash import compute_mdhash_id


def extract_title(doc_text: Any) -> str:
    return str(doc_text or "").split("\n", 1)[0].strip()


def normalize_title_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_doc_text_to_chunk_id(corpus: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    doc_text_to_chunk_id: dict[str, str] = {}
    for row in corpus:
        doc_text = f"{row['title']}\n{row['text']}"
        doc_text_to_chunk_id[doc_text] = compute_mdhash_id(doc_text, prefix="chunk-")
    return doc_text_to_chunk_id


def build_unique_title_to_doc_text(corpus: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    title_to_docs: dict[str, list[str]] = {}
    for row in corpus:
        title_key = normalize_title_key(row.get("title"))
        if not title_key:
            continue
        title_to_docs.setdefault(title_key, []).append(f"{row['title']}\n{row['text']}")
    return {title_key: docs[0] for title_key, docs in title_to_docs.items() if len(docs) == 1}


def align_pool_docs_to_corpus(
    pool_docs: Sequence[Any],
    pool_titles: Sequence[Any],
    *,
    corpus: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    canonical_docs = {f"{row['title']}\n{row['text']}" for row in corpus}
    unique_title_to_doc = build_unique_title_to_doc_text(corpus)
    aligned_docs: list[str] = []
    aligned_titles: list[str] = []
    for pos, raw_doc in enumerate(pool_docs):
        doc_text = str(raw_doc or "")
        raw_title = (
            str(pool_titles[pos]).strip()
            if pos < len(pool_titles)
            else extract_title(doc_text)
        )
        aligned_doc = doc_text
        if doc_text not in canonical_docs:
            title_doc = unique_title_to_doc.get(normalize_title_key(raw_title or extract_title(doc_text)))
            if title_doc is not None:
                aligned_doc = title_doc
        aligned_docs.append(aligned_doc)
        aligned_titles.append(extract_title(aligned_doc) or raw_title)
    return aligned_docs, aligned_titles


def remap_pool_doc_ids(
    pool_docs: Sequence[Any],
    *,
    passage_index: Any,
    doc_text_to_chunk_id: Mapping[str, str],
) -> list[int | None]:
    chunk_text_to_hash = getattr(getattr(passage_index, "chunk_embedding_store", None), "text_to_hash_id", {}) or {}
    passage_node_key_to_doc_idx = getattr(passage_index, "passage_node_key_to_doc_idx", {}) or {}
    mapped_doc_ids: list[int | None] = []
    for raw_doc in pool_docs:
        doc_text = str(raw_doc or "")
        chunk_id = doc_text_to_chunk_id.get(doc_text) or chunk_text_to_hash.get(doc_text)
        mapped_doc_id = passage_node_key_to_doc_idx.get(chunk_id) if chunk_id is not None else None
        try:
            mapped_doc_ids.append(int(mapped_doc_id) if mapped_doc_id is not None else None)
        except (TypeError, ValueError):
            mapped_doc_ids.append(None)
    return mapped_doc_ids


def align_pool_record(
    record: Mapping[str, Any],
    *,
    corpus: Sequence[Mapping[str, Any]],
    passage_index: Any,
    doc_text_to_chunk_id: Mapping[str, str],
) -> dict[str, Any]:
    pool_docs = list(record.get("pool_docs") or [])
    pool_titles = list(record.get("pool_titles") or [])
    reader_pool_doc_ids = list(record.get("pool_doc_ids") or [])
    if not pool_docs and pool_titles:
        pool_docs = ["" for _ in pool_titles]
    if len(pool_titles) < len(pool_docs):
        pool_titles.extend(extract_title(doc_text) for doc_text in pool_docs[len(pool_titles):])
    aligned_docs, aligned_titles = align_pool_docs_to_corpus(
        pool_docs,
        pool_titles,
        corpus=corpus,
    )
    output = dict(record)
    output["pool_docs"] = aligned_docs
    output["pool_titles"] = aligned_titles
    output["pool_doc_ids"] = remap_pool_doc_ids(
        aligned_docs,
        passage_index=passage_index,
        doc_text_to_chunk_id=doc_text_to_chunk_id,
    )
    pool_trace = dict(record.get("pool_trace") or record.get("pool_trace") or {})
    if reader_pool_doc_ids:
        pool_trace.setdefault("external_pool_doc_ids", reader_pool_doc_ids)
        pool_trace.setdefault("reader_pool_doc_ids", reader_pool_doc_ids)
    if pool_trace:
        output["pool_trace"] = pool_trace
    return output
