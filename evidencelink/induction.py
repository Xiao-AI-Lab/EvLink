"""Query-local evidence induction and candidate-pool construction."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Any, Sequence

from evidencelink.anchors import extract_question_anchors
from evidencelink.artifacts import (
    CandidateEvidence,
    CandidatePoolRecord,
    load_documents,
    load_questions,
    write_jsonl,
)
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


def _seed_roles(
    doc_id: str,
    *,
    anchor_seed_doc_ids: set[str],
    dense_seed_doc_ids: set[str],
) -> list[str]:
    roles: list[str] = []
    if doc_id in anchor_seed_doc_ids:
        roles.append("anchor")
    if doc_id in dense_seed_doc_ids:
        roles.append("dense")
    return roles


def bfs_candidate_doc_ids(
    index: EvidenceLinkIndex,
    seed_doc_ids: list[str],
    *,
    max_hops: int,
    pool_k: int,
    anchor_seed_doc_ids: Sequence[str] = (),
    dense_seed_doc_ids: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    visited: set[str] = set()
    ordered: list[str] = []
    parents: dict[str, list[str]] = {}
    discovery_events: list[dict[str, Any]] = []
    anchor_seed_set = {str(doc_id) for doc_id in anchor_seed_doc_ids}
    dense_seed_set = {str(doc_id) for doc_id in dense_seed_doc_ids}
    queue = deque(
        (doc_id, 0, [doc_id], None, None, [])
        for doc_id in seed_doc_ids
        if doc_id in index.documents
    )
    while queue and len(ordered) < int(pool_k):
        doc_id, depth, path, parent_doc_id, incoming_link, path_links = queue.popleft()
        if doc_id in visited:
            continue
        visited.add(doc_id)
        ordered.append(doc_id)
        parents[doc_id] = path
        incoming_links = (
            [
                link.to_mapping()
                for link in index.neighbors(str(parent_doc_id))
                if str(link.target_doc_id) == str(doc_id)
            ]
            if parent_doc_id is not None
            else []
        )
        path_hops = [
            {
                "source_doc_id": str(source_doc_id),
                "target_doc_id": str(target_doc_id),
                "links": [
                    link.to_mapping()
                    for link in index.neighbors(str(source_doc_id))
                    if str(link.target_doc_id) == str(target_doc_id)
                ],
            }
            for source_doc_id, target_doc_id in zip(path, path[1:])
        ]
        seed_roles = _seed_roles(
            doc_id,
            anchor_seed_doc_ids=anchor_seed_set,
            dense_seed_doc_ids=dense_seed_set,
        )
        discovery_events.append(
            {
                "step": len(discovery_events) + 1,
                "doc_id": str(doc_id),
                "depth": int(depth),
                "parent_doc_id": str(parent_doc_id) if parent_doc_id is not None else None,
                "discovery_method": "seed" if int(depth) == 0 else "bfs",
                "seed_roles": seed_roles,
                "path": list(path),
                "incoming_link": dict(incoming_link) if isinstance(incoming_link, dict) else None,
                "incoming_links": incoming_links,
                "path_links": [dict(link) for link in path_links],
                "path_hops": path_hops,
            }
        )
        if depth >= int(max_hops):
            continue
        for link in index.neighbors(doc_id):
            if link.target_doc_id not in visited:
                link_payload = link.to_mapping()
                queue.append(
                    (
                        link.target_doc_id,
                        depth + 1,
                        path + [link.target_doc_id],
                        doc_id,
                        link_payload,
                        path_links + [link_payload],
                    )
                )
    return ordered, {
        "node_count": len(visited),
        "seed_doc_ids": list(seed_doc_ids),
        "anchor_seed_doc_ids": list(anchor_seed_doc_ids),
        "dense_seed_doc_ids": list(dense_seed_doc_ids),
        "paths": parents,
        "discovery_events": discovery_events,
    }


def build_candidate_pool_records(
    *,
    questions_path: str | Path,
    corpus_path: str | Path,
    index_path: str | Path,
    dense_top_k: int = 20,
    max_hops: int = 2,
    pool_k: int = 100,
) -> list[CandidatePoolRecord]:
    documents = load_documents(corpus_path)
    docs_by_id = {doc.doc_id: doc for doc in documents}
    index = EvidenceLinkIndex.load(index_path)
    records: list[CandidatePoolRecord] = []
    for question in load_questions(questions_path):
        anchors = extract_question_anchors(question.question)
        anchor_seeds = anchor_seed_doc_ids(documents, anchors, limit=dense_top_k)
        dense_seeds = [doc_id for doc_id, _ in dense_seed_doc_ids(documents, question.question, top_k=dense_top_k)]
        seed_doc_ids = _ordered_unique(anchor_seeds + dense_seeds)
        candidate_doc_ids, local_subgraph = bfs_candidate_doc_ids(
            index,
            seed_doc_ids,
            max_hops=max_hops,
            pool_k=pool_k,
            anchor_seed_doc_ids=anchor_seeds,
            dense_seed_doc_ids=dense_seeds,
        )
        if len(candidate_doc_ids) < int(pool_k):
            before_fallback = set(candidate_doc_ids)
            candidate_doc_ids = _ordered_unique(candidate_doc_ids + dense_seeds)[: int(pool_k)]
            events = list(local_subgraph.get("discovery_events") or [])
            paths = dict(local_subgraph.get("paths") or {})
            for doc_id in candidate_doc_ids:
                if doc_id in before_fallback:
                    continue
                paths[doc_id] = [doc_id]
                events.append(
                    {
                        "step": len(events) + 1,
                        "doc_id": str(doc_id),
                        "depth": 0,
                        "parent_doc_id": None,
                        "discovery_method": "dense_fallback",
                        "seed_roles": _seed_roles(
                            doc_id,
                            anchor_seed_doc_ids=set(anchor_seeds),
                            dense_seed_doc_ids=set(dense_seeds),
                        ),
                        "path": [doc_id],
                        "incoming_link": None,
                        "incoming_links": [],
                        "path_links": [],
                        "path_hops": [],
                    }
                )
            local_subgraph["paths"] = paths
            local_subgraph["discovery_events"] = events
            local_subgraph["node_count"] = len(candidate_doc_ids)
        dense_score = {doc_id: score for doc_id, score in dense_seed_doc_ids(documents, question.question, top_k=len(documents))}
        event_by_doc_id = {
            str(event.get("doc_id")): event
            for event in list(local_subgraph.get("discovery_events") or [])
            if isinstance(event, dict)
        }
        candidates: list[CandidateEvidence] = []
        for rank, doc_id in enumerate(candidate_doc_ids[: int(pool_k)], start=1):
            doc = docs_by_id.get(doc_id)
            if doc is None:
                continue
            path = tuple(str(item) for item in (local_subgraph.get("paths") or {}).get(doc_id, [doc_id]))
            edge_evidence = []
            if len(path) >= 2:
                edge_evidence = index.edge_witnesses(path[-2], path[-1])
            discovery_event = event_by_doc_id.get(str(doc_id), {})
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
                    metadata={
                        "discovery_method": str(discovery_event.get("discovery_method") or "unknown"),
                        "discovery_depth": int(discovery_event.get("depth", 0) or 0),
                        "seed_roles": list(discovery_event.get("seed_roles") or []),
                    },
                )
            )
        records.append(
            CandidatePoolRecord(
                query_id=question.query_id,
                question=question.question,
                anchors=tuple(anchors),
                seed_doc_ids=tuple(seed_doc_ids),
                anchor_seed_doc_ids=tuple(anchor_seeds),
                dense_seed_doc_ids=tuple(dense_seeds),
                candidate_pool=tuple(candidates),
                gold_doc_ids=question.gold_doc_ids,
                gold_titles=question.gold_titles,
                gold_answers=question.gold_answers,
                local_subgraph=local_subgraph,
            )
        )
    return records


def write_candidate_pool(records: list[CandidatePoolRecord], output: str | Path) -> None:
    write_jsonl((record.to_mapping() for record in records), output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build query-local EvLink candidate pools C_q.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--pool-k", type=int, default=100)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    records = build_candidate_pool_records(
        questions_path=args.questions,
        corpus_path=args.corpus,
        index_path=args.index,
        dense_top_k=args.dense_top_k,
        max_hops=args.max_hops,
        pool_k=args.pool_k,
    )
    write_candidate_pool(records, args.output)
    print(json.dumps({"output": str(args.output), "query_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
