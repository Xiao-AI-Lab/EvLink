from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from evidencelink import (
    build_candidate_pool_cq,
    build_evidence_link_index,
    build_evidence_needs_bq,
    build_query_result_view,
    build_query_result_view_from_files,
    build_support_cache,
    compose_final_evidence_rq,
    load_query_result_view_schema,
)
from evidencelink.artifacts import read_jsonl


EQUIVALENCE_DIR = Path(__file__).resolve().parent / "fixtures" / "equivalence"
def build_fixture_artifacts(tmp_path: Path) -> tuple[dict, list[dict], dict, dict]:
    workdir = tmp_path / "run"
    workdir.mkdir(parents=True)
    index_path = workdir / "evidence_link_index.json"
    candidate_pool_path = workdir / "candidate_pool.jsonl"
    evidence_needs_path = workdir / "evidence_needs.jsonl"
    binding_cache_path = workdir / "binding_cache.json"
    selection_path = workdir / "evidence_selection.json"
    build_evidence_link_index(
        corpus_path=EQUIVALENCE_DIR / "corpus.jsonl",
        openie_path=EQUIVALENCE_DIR / "openie_facts.jsonl",
        output_path=index_path,
    )
    build_candidate_pool_cq(
        questions_path=EQUIVALENCE_DIR / "questions.jsonl",
        corpus_path=EQUIVALENCE_DIR / "corpus.jsonl",
        index_path=index_path,
        output_path=candidate_pool_path,
        dense_top_k=1,
        max_hops=2,
        pool_k=5,
    )
    build_evidence_needs_bq(
        questions_path=EQUIVALENCE_DIR / "questions.jsonl",
        output_path=evidence_needs_path,
        mode="whole_question",
    )
    build_support_cache(
        candidate_pool_path=candidate_pool_path,
        evidence_needs_path=evidence_needs_path,
        output_path=binding_cache_path,
        mode="simple",
        binding_model="simple-binding",
    )
    selection = compose_final_evidence_rq(
        dataset="equivalence",
        candidate_pool_path=candidate_pool_path,
        evidence_needs_path=evidence_needs_path,
        binding_cache_path=binding_cache_path,
        output_path=selection_path,
        reader_budget_k=2,
        stability_window_m=1,
        pool_k=5,
        binding_model="simple-binding",
        min_swap_gain=0.0,
    )
    paths = {
        "candidate_pool": str(candidate_pool_path),
        "evidence_selection": str(selection_path),
    }
    result = {"artifacts": paths}
    pool_rows = read_jsonl(candidate_pool_path)
    selected_ids = list(selection["rows"][0]["retrieved_doc_indices_top5"])
    selected_titles = list(selection["rows"][0]["retrieved_titles_top5"])
    reader = {
        "method": "evlink",
        "dataset": "equivalence",
        "reader": {"llm_name": "deterministic-fixture-reader", "qa_top_k": 2},
        "rows": [
            {
                "query_id": "0",
                "query_index": 0,
                "question": pool_rows[0]["question"],
                "prediction": "Sydney Harbour",
                "citations": [
                    {
                        "marker": "[1]",
                        "passage_position": 0,
                        "passage_id": str(selected_ids[0]),
                        "passage_title": str(selected_titles[0]),
                    }
                ],
            }
        ],
    }
    return result, pool_rows, selection, reader


def test_deterministic_fixture_produces_schema_validated_query_result_view(tmp_path: Path) -> None:
    _, pool_rows, selection, reader = build_fixture_artifacts(tmp_path)

    view = build_query_result_view(
        query_id="0",
        pool_payload=pool_rows,
        selection_payload=selection,
        reader_payload=reader,
    )
    Draft202012Validator(load_query_result_view_schema()).validate(view)
    repeated = build_query_result_view(
        query_id="0",
        pool_payload=pool_rows,
        selection_payload=selection,
        reader_payload=reader,
    )
    assert view == repeated
    assert view["artifact_schema_version"] == "query_result_view/v1"
    assert view["query_id"] == "0"
    assert view["answer"]["claims"] is None
    assert view["answer"]["citations"][0]["passage_id"] in {
        passage["passage_id"] for passage in view["passages"]
    }
    assert view["retrieval_trace"]["capabilities"]["witnesses"] == "available"
    assert any(event["depth"] == 2 for event in view["retrieval_trace"]["discovery_events"])
    assert any("bridge" in passage["roles"] for passage in view["passages"])
    assert view["support_matrix"]
    assert all("supporting_bindings" in row for row in view["support_matrix"])
    assert any(edge["type"] == "relation_grounded" for edge in view["evidence_graph"]["edges"])
    assert any(edge["type"] == "citation" for edge in view["evidence_graph"]["edges"])


def test_query_result_view_file_api_and_cli_inputs(tmp_path: Path) -> None:
    result, _, _, reader = build_fixture_artifacts(tmp_path)
    reader_path = tmp_path / "reader.json"
    reader_path.write_text(json.dumps(reader), encoding="utf-8")
    output_path = tmp_path / "query-view.json"

    view = build_query_result_view_from_files(
        query_id="0",
        candidate_pool_path=result["artifacts"]["candidate_pool"],
        selection_path=result["artifacts"]["evidence_selection"],
        reader_path=reader_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == view
    cli_output = tmp_path / "query-view-cli.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evidencelink.view.cli",
            "--query-id",
            "0",
            "--candidate-pool",
            result["artifacts"]["candidate_pool"],
            "--selection",
            result["artifacts"]["evidence_selection"],
            "--reader",
            str(reader_path),
            "--output",
            str(cli_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["query_id"] == "0"
    assert json.loads(cli_output.read_text(encoding="utf-8")) == view


def test_query_result_view_requires_exact_query_identity(tmp_path: Path) -> None:
    _, pool_rows, selection, reader = build_fixture_artifacts(tmp_path)

    with pytest.raises(ValueError, match="exactly one row"):
        build_query_result_view(
            query_id="missing",
            pool_payload=pool_rows,
            selection_payload=selection,
            reader_payload=reader,
        )


def test_external_pool_reports_witnesses_unavailable() -> None:
    pool = [
        {
            "query_id": "external-1",
            "question": "Question?",
            "candidate_pool": [
                {"doc_id": "p1", "rank": 1, "title": "P1", "text": "Body"}
            ],
            "pool_trace": {"input_method": "external_retriever"},
        }
    ]
    selection = {
        "method": "evlink",
        "rows": [
            {
                "query_id": "external-1",
                "question": "Question?",
                "evidence_selection": {
                    "baseline_positions": [0],
                    "final_positions": [0],
                    "protected_baseline_positions": [],
                    "admitted_positions": [],
                    "rank_stability_held": True,
                },
                "raw_selection_trace": {"requirements": [], "support_matrix": []},
            }
        ],
    }
    reader = {
        "rows": [
            {
                "query_id": "external-1",
                "question": "Question?",
                "prediction": "Answer",
                "citations": [{"marker": "[1]", "passage_id": "p1"}],
            }
        ]
    }

    view = build_query_result_view(
        query_id="external-1",
        pool_payload=pool,
        selection_payload=selection,
        reader_payload=reader,
    )

    assert view["retrieval_trace"]["capabilities"] == {
        "witnesses": "unavailable",
        "faithful_step_through": False,
    }
    assert not any(
        edge["type"] in {"relation_grounded", "endpoint_aligned"}
        for edge in view["evidence_graph"]["edges"]
    )
