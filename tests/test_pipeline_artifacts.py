from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evidencelink.binding import build_binding_cache
from evidencelink.evidence_needs import build_evidence_need_rows
from evidencelink.index import EvidenceLinkIndex, build_evidence_link_index
from evidencelink.induction import build_candidate_pool_records, write_candidate_pool
from evidencelink.openie import build_openie_facts
from evidencelink.run_evidence_selection import run_evidence_selection
from evidencelink.io_utils import write_json
from evidencelink.artifacts import read_jsonl


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    questions = tmp_path / "questions.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"doc_id": "0", "title": "Alice", "text": "Alice was born in Paris."}),
                json.dumps({"doc_id": "1", "title": "Paris", "text": "Paris is in France."}),
                json.dumps({"doc_id": "2", "title": "France", "text": "France is in Europe."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            {
                "query_id": "0",
                "question": "Where was Alice born?",
                "gold_doc_ids": ["0"],
                "gold_titles": ["Alice"],
                "gold_answers": ["Paris"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return corpus, questions


def test_evidence_link_index_and_candidate_pool(tmp_path: Path) -> None:
    corpus, questions = write_fixture(tmp_path)
    openie = tmp_path / "openie.jsonl"
    index_path = tmp_path / "index.json"

    facts = build_openie_facts(SimpleNamespace(corpus=corpus, output=openie, mode="simple"))
    index = build_evidence_link_index(corpus_path=corpus, openie_path=openie, output_path=index_path)
    records = build_candidate_pool_records(
        questions_path=questions,
        corpus_path=corpus,
        index_path=index_path,
        dense_top_k=2,
        max_hops=2,
        pool_k=5,
    )

    assert [fact.object for fact in facts][:2] == ["Paris", "France"]
    assert len(index.links) >= 2
    assert records[0].candidate_pool[0].title == "Alice"
    assert "Paris" in [candidate.title for candidate in records[0].candidate_pool]
    loaded = EvidenceLinkIndex.load(index_path)
    assert loaded.edge_witnesses("0", "1")


def test_public_pipeline_smoke_runs_to_selection(tmp_path: Path) -> None:
    corpus, questions = write_fixture(tmp_path)
    openie = tmp_path / "openie.jsonl"
    index_path = tmp_path / "index.json"
    pool = tmp_path / "pool.jsonl"
    needs = tmp_path / "needs.jsonl"
    binding_cache = tmp_path / "binding.json"

    build_openie_facts(SimpleNamespace(corpus=corpus, output=openie, mode="simple"))
    build_evidence_link_index(corpus_path=corpus, openie_path=openie, output_path=index_path)
    records = build_candidate_pool_records(
        questions_path=questions,
        corpus_path=corpus,
        index_path=index_path,
        dense_top_k=2,
        max_hops=2,
        pool_k=5,
    )
    write_candidate_pool(records, pool)
    need_rows = build_evidence_need_rows(
        SimpleNamespace(questions=questions, output=needs, mode="whole_question")
    )
    from evidencelink.artifacts import write_jsonl

    write_jsonl(need_rows, needs)
    build_binding_cache(
        SimpleNamespace(
            candidate_pool=pool,
            evidence_needs=needs,
            output=binding_cache,
            mode="simple",
            binding_model="simple-binding",
            max_candidates=5,
        )
    )
    report = run_evidence_selection(
        SimpleNamespace(
            dataset="toy",
            limit=0,
            max_queries=0,
            pool_k=5,
            reader_budget_k=2,
            stability_window_m=1,
            pool_json=pool,
            requirement_report=needs,
            binding_cache_path=binding_cache,
            output_json=tmp_path / "selection.json",
            output_root=tmp_path,
            llm_binding_model="simple-binding",
            embedding_name="deterministic-hash",
            embedding_base_url="offline",
            embedding_batch_size=8,
            embedding_api_key="",
            embedding_timeout=10.0,
            binding_max_candidates=5,
            llm_binding_title_match_mode="wiki_title",
            min_coverage_gain=0.0,
            min_swap_gain=0.0,
            allow_missing_requirements=True,
        )
    )
    write_json(report, tmp_path / "selection.json")

    assert report["summary"]["count"] == 1
    assert report["rows"][0]["retrieved_titles_top5"]


def test_jsonl_candidate_pool_matches_records_json_for_evidence_selection(tmp_path: Path) -> None:
    corpus, questions = write_fixture(tmp_path)
    openie = tmp_path / "openie.jsonl"
    index_path = tmp_path / "index.json"
    pool_jsonl = tmp_path / "pool.jsonl"
    pool_json = tmp_path / "pool.json"
    needs = tmp_path / "needs.jsonl"
    binding_cache = tmp_path / "binding.json"

    build_openie_facts(SimpleNamespace(corpus=corpus, output=openie, mode="simple"))
    build_evidence_link_index(corpus_path=corpus, openie_path=openie, output_path=index_path)
    records = build_candidate_pool_records(
        questions_path=questions,
        corpus_path=corpus,
        index_path=index_path,
        dense_top_k=2,
        max_hops=2,
        pool_k=5,
    )
    write_candidate_pool(records, pool_jsonl)
    pool_json.write_text(
        json.dumps({"artifact_type": "candidate_pool", "records": read_jsonl(pool_jsonl)}, ensure_ascii=False),
        encoding="utf-8",
    )

    from evidencelink.artifacts import write_jsonl

    write_jsonl(
        build_evidence_need_rows(SimpleNamespace(questions=questions, output=needs, mode="whole_question")),
        needs,
    )
    build_binding_cache(
        SimpleNamespace(
            candidate_pool=pool_jsonl,
            evidence_needs=needs,
            output=binding_cache,
            mode="simple",
            binding_model="simple-binding",
            max_candidates=5,
        )
    )

    def run(pool_path: Path) -> dict:
        return run_evidence_selection(
            SimpleNamespace(
                dataset="toy",
                limit=0,
                max_queries=0,
                pool_k=5,
                reader_budget_k=2,
                stability_window_m=1,
                pool_json=pool_path,
                requirement_report=needs,
                binding_cache_path=binding_cache,
                output_json=tmp_path / f"{pool_path.stem}.selection.json",
                output_root=tmp_path,
                llm_binding_model="simple-binding",
                embedding_name="deterministic-hash",
                embedding_base_url="offline",
                embedding_batch_size=8,
                embedding_api_key="",
                embedding_timeout=10.0,
                binding_max_candidates=5,
                llm_binding_title_match_mode="wiki_title",
                min_coverage_gain=0.0,
                min_swap_gain=0.0,
                allow_missing_requirements=True,
            )
        )

    from_jsonl = run(pool_jsonl)
    from_json = run(pool_json)

    assert from_jsonl["summary"] == from_json["summary"]
    assert from_jsonl["rows"] == from_json["rows"]
