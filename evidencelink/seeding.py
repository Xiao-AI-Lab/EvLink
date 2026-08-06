"""Dense and anchor seeding for EvLink candidate-pool construction."""

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
    seeds: list[str] = []
    seen: set[str] = set()
    normalized_anchors = [normalize_endpoint(anchor) for anchor in anchors if normalize_endpoint(anchor)]
    for doc in documents:
        haystack = normalize_endpoint(f"{doc.title} {doc.text}")
        title = normalize_endpoint(doc.title)
        if any(anchor in title or anchor in haystack for anchor in normalized_anchors):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                seeds.append(doc.doc_id)
        if len(seeds) >= int(limit):
            break
    return seeds


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
        lexical_bonus = 0.05 * len(set(re.findall(r"\w+", question.lower())).intersection(re.findall(r"\w+", doc.passage.lower())))
        scored.append((doc.doc_id, cosine(query_vec, doc_vec) + lexical_bonus))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[: int(top_k)]
