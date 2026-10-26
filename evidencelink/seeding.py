"""Dense and anchor seeding for EvidenceLink candidate-pool construction."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from evidencelink.artifacts import Document
from evidencelink.embedding import DeterministicHashEmbeddingClient
from evidencelink.index import normalize_endpoint


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def anchor_seed_doc_ids(documents: Sequence[Document], anchors: Sequence[str], *, limit: int = 20) -> list[str]:
    patterns = [
        re.compile(r"\b%s\b" % re.escape(anchor))
        for anchor in (normalize_endpoint(a) for a in anchors)
        if anchor
    ]
    scored: list[tuple[str, int, int]] = []
    for doc in documents:
        title = normalize_endpoint(doc.title)
        haystack = normalize_endpoint(f"{doc.title} {doc.text}")
        title_hits = sum(1 for pattern in patterns if pattern.search(title))
        text_hits = sum(1 for pattern in patterns if pattern.search(haystack))
        if title_hits or text_hits:
            scored.append((doc.doc_id, title_hits, text_hits))
    scored.sort(key=lambda item: (-item[1], -item[2]))
    return [doc_id for doc_id, _, _ in scored[: int(limit)]]


def dense_seed_doc_ids(
    documents: Sequence[Document],
    question: str,
    *,
    top_k: int = 20,
    embedding_client=None,
) -> list[tuple[str, float]]:
    client = embedding_client or DeterministicHashEmbeddingClient()
    texts = [str(question or "")] + [doc.passage for doc in documents]
    embeddings = client.embed(texts)
    query_vec = np.asarray(embeddings.get(str(question or ""), np.array([])), dtype=float)
    scored: list[tuple[str, float]] = []
    for doc in documents:
        doc_vec = np.asarray(embeddings.get(doc.passage, np.array([])), dtype=float)
        scored.append((doc.doc_id, cosine(query_vec, doc_vec)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[: int(top_k)]
