"""Dependency-free adapters for external retriever result objects.

The adapters in this module use duck typing so EvLink does not require
HippoRAG or LightRAG at installation time. They only normalize upstream
retrieval results into the candidate mapping accepted by :class:`EvidenceSelector`.
Upstream graph seeds, references, and file paths remain upstream metadata; they
are not relabeled as EvLink source-grounded edge witnesses.
"""

from __future__ import annotations

import math
from pathlib import PurePath
from typing import Any, Mapping, Sequence


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field_name} must be a sequence, got {type(value).__name__}")
    try:
        return list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence") from exc


def _score(value: Any, index: int) -> float:
    if value is None or value == "":
        return float(-index)
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate {index} has an invalid retrieval score: {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"candidate {index} has a non-finite retrieval score: {value!r}")
    return score


def _title_and_text(text: Any, metadata: Mapping[str, Any], fallback: str) -> tuple[str, str]:
    passage = str(text or "").strip()
    title = str(
        metadata.get("title")
        or metadata.get("document_title")
        or metadata.get("name")
        or ""
    ).strip()
    if not title and "\n" in passage:
        title, passage = passage.split("\n", 1)
        title = title.strip()
        passage = passage.strip()
    return title or fallback, passage


def candidates_from_hipporag(result: Any, *, source: str = "hipporag") -> list[dict[str, Any]]:
    """Convert one HippoRAG ``RetrievalResult`` or ``QuerySolution``.

    Supported fields are ``docs``, ``scores`` or ``doc_scores``,
    ``doc_metadata``, and ``graph_seeds``. Metadata identifiers take precedence
    over the stable positional fallback. Query-level graph seeds are retained
    under ``upstream_metadata`` instead of being presented as link evidence.
    """

    docs = _sequence(_field(result, "docs"), field_name="HippoRAG result.docs")
    if not docs:
        raise ValueError("HippoRAG result.docs must not be empty")
    raw_scores = _field(result, "scores")
    if raw_scores is None:
        raw_scores = _field(result, "doc_scores")
    scores = _sequence(raw_scores, field_name="HippoRAG result.scores") if raw_scores is not None else []
    metadata_rows = _sequence(
        _field(result, "doc_metadata"),
        field_name="HippoRAG result.doc_metadata",
    )
    graph_seeds = _sequence(
        _field(result, "graph_seeds"),
        field_name="HippoRAG result.graph_seeds",
    )

    candidates: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        raw_metadata = metadata_rows[index] if index < len(metadata_rows) else {}
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        doc_id = next(
            (
                str(metadata[key]).strip()
                for key in ("doc_id", "id", "idx", "chunk_id", "source_id")
                if metadata.get(key) is not None and str(metadata.get(key)).strip()
            ),
            f"hipporag:{index}",
        )
        title, text = _title_and_text(doc, metadata, f"doc_{doc_id}")
        upstream_metadata = dict(metadata)
        if graph_seeds:
            upstream_metadata["graph_seeds"] = graph_seeds
        candidates.append(
            {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "rank": index + 1,
                "score": _score(scores[index] if index < len(scores) else None, index),
                "source": source,
                "metadata": upstream_metadata,
            }
        )
    return candidates


def candidates_from_lightrag(result: Any, *, source: str = "lightrag") -> list[dict[str, Any]]:
    """Convert a LightRAG ``QueryResult`` or its ``raw_data`` mapping.

    LightRAG's user-format chunks live under ``raw_data["data"]["chunks"]``.
    ``reference_id`` values are joined to ``data.references`` so file
    provenance survives normalization. LightRAG does not expose a chunk score
    in this result shape, so input order is represented by deterministic
    fallback scores.
    """

    raw_data = _field(result, "raw_data")
    if raw_data is None and isinstance(result, Mapping):
        raw_data = result
    if not isinstance(raw_data, Mapping):
        raise TypeError("LightRAG result must expose a raw_data mapping")
    data = raw_data.get("data")
    if not isinstance(data, Mapping):
        raise ValueError('LightRAG raw_data must contain a "data" mapping')
    chunks = _sequence(data.get("chunks"), field_name="LightRAG raw_data.data.chunks")
    if not chunks:
        raise ValueError("LightRAG raw_data.data.chunks must not be empty")
    references = _sequence(
        data.get("references"),
        field_name="LightRAG raw_data.data.references",
    )
    reference_paths = {
        str(row.get("reference_id") or ""): str(row.get("file_path") or "")
        for row in references
        if isinstance(row, Mapping)
    }

    candidates: list[dict[str, Any]] = []
    for index, raw_chunk in enumerate(chunks):
        if not isinstance(raw_chunk, Mapping):
            raise TypeError(f"LightRAG chunk {index} must be a mapping")
        chunk = dict(raw_chunk)
        reference_id = str(chunk.get("reference_id") or "").strip()
        file_path = str(chunk.get("file_path") or reference_paths.get(reference_id) or "").strip()
        chunk_id = str(chunk.get("chunk_id") or f"lightrag:{index}").strip()
        title = str(chunk.get("title") or "").strip()
        if not title and file_path:
            title = PurePath(file_path).name or file_path
        text = str(chunk.get("content") or chunk.get("text") or "").strip()
        if not title and "\n" in text:
            title, text = text.split("\n", 1)
            title = title.strip()
            text = text.strip()
        candidates.append(
            {
                "doc_id": chunk_id,
                "title": title or f"doc_{chunk_id}",
                "text": text,
                "rank": index + 1,
                "score": _score(chunk.get("score"), index),
                "source": source,
                "metadata": {
                    "reference_id": reference_id,
                    "file_path": file_path,
                },
            }
        )
    return candidates
