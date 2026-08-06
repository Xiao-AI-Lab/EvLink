from __future__ import annotations

from pathlib import Path

from evidencelink import PaperPipelineConfig, run_paper_pipeline
from evidencelink.api import (
    build_candidate_pool_cq,
    build_evidence_needs_bq,
    build_openie_artifact,
    build_support_cache,
    compose_final_evidence_rq,
)
from evidencelink.index import build_evidence_link_index


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    questions = tmp_path / "questions.jsonl"
    corpus.write_text(
        "\n".join(
            [
                '{"doc_id": "0", "title": "Alice", "text": "Alice was born in Paris."}',
                '{"doc_id": "1", "title": "Paris", "text": "Paris is in France."}',
                '{"doc_id": "2", "title": "France", "text": "France is in Europe."}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    questions.write_text(
        (
            '{"query_id": "0", "question": "Where was Alice born?", '
            '"gold_doc_ids": ["0"], "gold_titles": ["Alice"], "gold_answers": ["Paris"]}'
        )
        + "\n",
        encoding="utf-8",
    )
    return corpus, questions


def test_paper_facing_pipeline_api_runs_to_rq(tmp_path: Path) -> None:
    corpus, questions = write_fixture(tmp_path)
    payload = run_paper_pipeline(
        corpus_path=corpus,
        questions_path=questions,
        workdir=tmp_path / "paper_run",
        config=PaperPipelineConfig(
            dataset="toy",
            reader_budget_k=2,
            stability_window_m=1,
            pool_k=5,
            dense_top_k=2,
            binding_max_candidates=5,
            min_swap_gain=0.0,
            force=True,
        ),
    )

    artifacts = payload["artifacts"]
    assert payload["method"] == "EvLink"
    assert payload["selection_summary"]["count"] == 1
    assert Path(artifacts["candidate_pool"]).exists()
    assert Path(artifacts["evidence_needs"]).exists()
    assert Path(artifacts["binding_cache"]).exists()
    assert Path(artifacts["evidence_selection"]).exists()
    assert [step["name"] for step in payload["steps"]] == [
        "build_openie_facts",
        "build_evidence_link_index",
        "build_candidate_pool_cq",
        "build_evidence_needs_bq",
        "build_support_cache",
        "compose_final_evidence_rq",
    ]


def test_paper_facing_stage_api_names_match_notation(tmp_path: Path) -> None:
    corpus, questions = write_fixture(tmp_path)
    openie = tmp_path / "openie_facts.jsonl"
    index = tmp_path / "evidence_link_index.json"
    candidate_pool = tmp_path / "candidate_pool.jsonl"
    evidence_needs = tmp_path / "evidence_needs.jsonl"
    binding_cache = tmp_path / "binding_cache.json"
    evidence_selection = tmp_path / "evidence_selection.json"

    facts = build_openie_artifact(corpus_path=corpus, output_path=openie)
    built_index = build_evidence_link_index(corpus_path=corpus, openie_path=openie, output_path=index)
    cq_records = build_candidate_pool_cq(
        questions_path=questions,
        corpus_path=corpus,
        index_path=index,
        output_path=candidate_pool,
        dense_top_k=2,
        pool_k=5,
    )
    bq_rows = build_evidence_needs_bq(questions_path=questions, output_path=evidence_needs)
    cache = build_support_cache(
        candidate_pool_path=candidate_pool,
        evidence_needs_path=evidence_needs,
        output_path=binding_cache,
        max_candidates=5,
    )
    rq_payload = compose_final_evidence_rq(
        dataset="toy",
        candidate_pool_path=candidate_pool,
        evidence_needs_path=evidence_needs,
        binding_cache_path=binding_cache,
        output_path=evidence_selection,
        reader_budget_k=2,
        stability_window_m=1,
        pool_k=5,
        min_swap_gain=0.0,
    )

    assert facts
    assert built_index.links
    assert cq_records[0].question == "Where was Alice born?"
    assert bq_rows[0]["B_q"]
    assert cache
    assert rq_payload["rows"][0]["retrieved_titles_top5"]
