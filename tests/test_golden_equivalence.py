from __future__ import annotations

import json
from pathlib import Path

from evidencelink.index import build_evidence_link_index
from evidencelink.induction import build_candidate_pool_records


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "equivalence"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_evidence_link_index_matches_golden_summary(tmp_path: Path) -> None:
    index_path = tmp_path / "evidence_link_index.json"
    index = build_evidence_link_index(
        corpus_path=FIXTURE_DIR / "corpus.jsonl",
        openie_path=FIXTURE_DIR / "openie_facts.jsonl",
        output_path=index_path,
    )
    golden = read_json(FIXTURE_DIR / "golden_index_summary.json")

    assert len(index.documents) == golden["document_count"]
    assert sum(link.link_type == "relation_grounded" for link in index.links) == golden["relation_grounded_link_count"]
    assert sum(link.link_type == "endpoint_aligned" for link in index.links) == golden["endpoint_aligned_link_count"]
    assert any(
        link.source_doc_id == "0"
        and link.target_doc_id == "1"
        and link.link_type == "relation_grounded"
        and link.relation == "drains to"
        for link in index.links
    )
    assert any(
        link.source_doc_id == "1"
        and link.target_doc_id == "2"
        and link.link_type == "relation_grounded"
        and link.relation == "main tributary of"
        for link in index.links
    )


def test_query_local_candidate_pool_matches_golden_path(tmp_path: Path) -> None:
    index_path = tmp_path / "evidence_link_index.json"
    build_evidence_link_index(
        corpus_path=FIXTURE_DIR / "corpus.jsonl",
        openie_path=FIXTURE_DIR / "openie_facts.jsonl",
        output_path=index_path,
    )
    records = build_candidate_pool_records(
        questions_path=FIXTURE_DIR / "questions.jsonl",
        corpus_path=FIXTURE_DIR / "corpus.jsonl",
        index_path=index_path,
        dense_top_k=1,
        max_hops=2,
        pool_k=5,
    )
    golden = read_json(FIXTURE_DIR / "golden_candidate_pool.json")
    record = records[0].to_mapping()

    assert [item["doc_id"] for item in record["candidate_pool"][:3]] == golden["candidate_doc_ids"]
    assert [item["title"] for item in record["candidate_pool"][:3]] == golden["candidate_titles"]
    assert record["local_subgraph"]["paths"] == golden["paths"]
    assert record["anchor_seed_doc_ids"]
    assert record["dense_seed_doc_ids"]
    events = record["local_subgraph"]["discovery_events"]
    assert [event["step"] for event in events] == list(range(1, len(events) + 1))
    bridge_event = next(event for event in events if event["doc_id"] == "2")
    assert bridge_event["discovery_method"] == "bfs"
    assert bridge_event["depth"] == 2
    assert bridge_event["parent_doc_id"] == "1"
    assert len(bridge_event["path_links"]) == 2
    assert bridge_event["path_links"][-1]["witnesses"]
    assert len(bridge_event["path_hops"]) == 2
    assert all(hop["links"] for hop in bridge_event["path_hops"])
    assert any(
        link["link_type"] == "relation_grounded"
        for link in bridge_event["incoming_links"]
    )
    relations = {
        str(witness.get("relation"))
        for item in record["candidate_pool"]
        for witness in item.get("edge_evidence", [])
        if witness.get("relation")
    }
    assert set(golden["required_witness_relations"]).issubset(relations)
