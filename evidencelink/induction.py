"""Query-local evidence induction and candidate-pool construction."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Any

from evidencelink.anchors import extract_question_anchors
from evidencelink.artifacts import (
    CandidateEvidence,
    CandidatePoolRecord,
    load_documents,
    load_questions,
    write_jsonl,
)
from evidencelink.embedding import DeterministicHashEmbeddingClient, OpenAIEmbeddingClient
from evidencelink.index import EvidenceLinkIndex
from evidencelink.seeding import anchor_seed_doc_ids, dense_seed_doc_ids


def _ordered_unique(values) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def bfs_candidate_doc_ids(
    index: EvidenceLinkIndex,
    seed_doc_ids: list[str],
    *,
    max_hops: int,
    pool_k: int,
    allowed_doc_ids: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    visited: set[str] = set()
    ordered: list[str] = []
    parents: dict[str, list[str]] = {}
    queue = deque(
        (doc_id, 0, [doc_id])
        for doc_id in seed_doc_ids
        if doc_id in index.documents and (allowed_doc_ids is None or doc_id in allowed_doc_ids)
    )
    while queue and len(ordered) < int(pool_k):
        doc_id, depth, path = queue.popleft()
        if doc_id in visited:
            continue
        visited.add(doc_id)
        ordered.append(doc_id)
        parents[doc_id] = path
        if depth >= int(max_hops):
            continue
        for link in index.neighbors(doc_id):
            if link.target_doc_id not in visited and (allowed_doc_ids is None or link.target_doc_id in allowed_doc_ids):
                queue.append((link.target_doc_id, depth + 1, path + [link.target_doc_id]))
    return ordered, {
        "node_count": len(visited),
        "seed_doc_ids": list(seed_doc_ids),
        "paths": parents,
        "scope_doc_count": None if allowed_doc_ids is None else len(allowed_doc_ids),
    }


def documents_for_question_scope(documents, question):
    """Return the corpus slice explicitly assigned to a closed-domain question."""

    metadata = dict(question.metadata or {})
    scope_id = str(metadata.get("scope_id") or "").strip()
    if not scope_id:
        return list(documents), None
    scoped_documents = [doc for doc in documents if str((doc.metadata or {}).get("scope_id") or "") == scope_id]
    if not scoped_documents:
        raise ValueError(f"question {question.query_id!r} requests missing corpus scope {scope_id!r}")
    return scoped_documents, scope_id


def build_candidate_pool_records(
    *,
    questions_path: str | Path,
    corpus_path: str | Path,
    index_path: str | Path,
    dense_top_k: int = 20,
    max_hops: int = 2,
    pool_k: int = 100,
    max_queries: int = 0,
    embedding_client=None,
) -> list[CandidatePoolRecord]:
    documents = load_documents(corpus_path)
    docs_by_id = {doc.doc_id: doc for doc in documents}
    index = EvidenceLinkIndex.load(index_path)
    records: list[CandidatePoolRecord] = []
    questions = load_questions(questions_path)
    if int(max_queries) > 0:
        questions = questions[: int(max_queries)]
    for query_index, question in enumerate(questions):
        scoped_documents, scope_id = documents_for_question_scope(documents, question)
        scoped_doc_ids = {doc.doc_id for doc in scoped_documents}
        anchors = extract_question_anchors(question.question)
        anchor_seeds = anchor_seed_doc_ids(scoped_documents, anchors, limit=dense_top_k)
        dense_seeds = [
            doc_id
            for doc_id, _ in dense_seed_doc_ids(
                scoped_documents,
                question.question,
                top_k=dense_top_k,
                embedding_client=embedding_client,
            )
        ]
        seed_doc_ids = _ordered_unique(anchor_seeds + dense_seeds)
        candidate_doc_ids, local_subgraph = bfs_candidate_doc_ids(
            index,
            seed_doc_ids,
            max_hops=max_hops,
            pool_k=pool_k,
            allowed_doc_ids=scoped_doc_ids if scope_id else None,
        )
        if len(candidate_doc_ids) < int(pool_k):
            candidate_doc_ids = _ordered_unique(candidate_doc_ids + dense_seeds)[: int(pool_k)]
        dense_score = {
            doc_id: score
            for doc_id, score in dense_seed_doc_ids(
                scoped_documents,
                question.question,
                top_k=len(scoped_documents),
                embedding_client=embedding_client,
            )
        }
        bfs_positions = {doc_id: position for position, doc_id in enumerate(candidate_doc_ids)}
        candidate_doc_ids = sorted(
            candidate_doc_ids,
            key=lambda doc_id: (-dense_score.get(doc_id, 0.0), bfs_positions[doc_id]),
        )
        local_subgraph["scope_id"] = scope_id
        candidates: list[CandidateEvidence] = []
        for rank, doc_id in enumerate(candidate_doc_ids[: int(pool_k)], start=1):
            doc = docs_by_id.get(doc_id)
            if doc is None:
                continue
            path = tuple(str(item) for item in (local_subgraph.get("paths") or {}).get(doc_id, [doc_id]))
            edge_evidence = []
            if len(path) >= 2:
                edge_evidence = index.edge_witnesses(path[-2], path[-1])
            candidates.append(
                CandidateEvidence(
                    rank=rank,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    source="query_local_evidence_induction",
                    score=float(dense_score.get(doc_id, max(0.0, 1.0 / rank))),
                    path=path,
                    edge_evidence=tuple(edge_evidence),
                )
            )
        records.append(
            CandidatePoolRecord(
                query_id=question.query_id,
                question=question.question,
                anchors=tuple(anchors),
                seed_doc_ids=tuple(seed_doc_ids),
                candidate_pool=tuple(candidates),
                gold_doc_ids=question.gold_doc_ids,
                gold_titles=question.gold_titles,
                gold_answers=question.gold_answers,
                local_subgraph=local_subgraph,
                query_index=query_index,
                metadata=question.metadata,
            )
        )
    return records


def write_candidate_pool(records: list[CandidatePoolRecord], output: str | Path) -> None:
    write_jsonl((record.to_mapping() for record in records), output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build query-local EvidenceLink candidate pools C_q.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--embedding-name", default="deterministic-hash")
    parser.add_argument("--embedding-base-url", default="offline")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if str(args.embedding_name).lower() in {"deterministic-hash", "offline"}:
        embedding_client = DeterministicHashEmbeddingClient()
    else:
        embedding_client = OpenAIEmbeddingClient(
            base_url=str(args.embedding_base_url),
            model=str(args.embedding_name),
            batch_size=int(args.embedding_batch_size),
            api_key=str(args.embedding_api_key),
            timeout=float(args.embedding_timeout),
        )
    records = build_candidate_pool_records(
        questions_path=args.questions,
        corpus_path=args.corpus,
        index_path=args.index,
        dense_top_k=args.dense_top_k,
        max_hops=args.max_hops,
        pool_k=args.pool_k,
        max_queries=args.max_queries,
        embedding_client=embedding_client,
    )
    write_candidate_pool(records, args.output)
    print(json.dumps({"output": str(args.output), "query_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
