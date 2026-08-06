from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from evidencelink import EvidenceSelector, EvidenceSelectorConfig
from evidencelink.integrations import candidates_from_hipporag, candidates_from_lightrag


def test_hipporag_adapter_accepts_retrieval_result_shape() -> None:
    result = SimpleNamespace(
        docs=["Alpha\nAlpha points to Beta.", "Beta\nBeta supplies the answer."],
        scores=[0.91, 0.83],
        doc_metadata=[{"doc_id": "a", "collection": "wiki"}, {"id": "b"}],
        graph_seeds=[("Alpha", "points_to", "Beta")],
    )

    candidates = candidates_from_hipporag(result)

    assert [row["doc_id"] for row in candidates] == ["a", "b"]
    assert candidates[0]["title"] == "Alpha"
    assert candidates[0]["text"] == "Alpha points to Beta."
    assert candidates[0]["score"] == 0.91
    assert candidates[0]["source"] == "hipporag"
    assert candidates[0]["metadata"]["graph_seeds"]
    assert "edge_evidence" not in candidates[0]

    selection = EvidenceSelector(EvidenceSelectorConfig(reader_budget_k=1)).select(
        question="What supplies the answer?", candidates=candidates
    )
    assert selection.baseline_evidence[0].source == "hipporag"
    assert selection.baseline_evidence[0].metadata["collection"] == "wiki"


def test_hipporag_adapter_accepts_query_solution_scores_and_positional_ids() -> None:
    result = {"docs": ["Only passage"], "doc_scores": ["0.5"]}

    candidates = candidates_from_hipporag(result, source="hipporag-mainline")

    assert candidates[0]["doc_id"] == "hipporag:0"
    assert candidates[0]["score"] == 0.5
    assert candidates[0]["source"] == "hipporag-mainline"


def test_lightrag_adapter_joins_references_without_claiming_link_evidence() -> None:
    result = SimpleNamespace(
        raw_data={
            "data": {
                "chunks": [
                    {
                        "reference_id": "1",
                        "content": "Alice founded Acme.",
                        "chunk_id": "chunk-a",
                    },
                    {
                        "reference_id": "2",
                        "content": "Singapore\nAlice was born in Singapore.",
                        "chunk_id": "chunk-b",
                        "file_path": "/data/singapore.md",
                    },
                ],
                "references": [
                    {"reference_id": "1", "file_path": "/data/acme.pdf"},
                    {"reference_id": "2", "file_path": "/data/singapore.md"},
                ],
            }
        }
    )

    candidates = candidates_from_lightrag(result)

    assert candidates[0]["doc_id"] == "chunk-a"
    assert candidates[0]["title"] == "acme.pdf"
    assert candidates[0]["metadata"] == {
        "reference_id": "1",
        "file_path": "/data/acme.pdf",
    }
    assert candidates[1]["title"] == "singapore.md"
    assert candidates[1]["score"] == -1.0
    assert "edge_evidence" not in candidates[0]
    selection = EvidenceSelector(EvidenceSelectorConfig(reader_budget_k=1)).select(
        question="Who founded Acme?", candidates=candidates
    )
    assert selection.baseline_evidence[0].metadata["file_path"] == "/data/acme.pdf"


def test_external_adapters_reject_missing_or_malformed_candidate_lists() -> None:
    with pytest.raises(ValueError, match="docs must not be empty"):
        candidates_from_hipporag({"docs": []})
    with pytest.raises(TypeError, match="raw_data mapping"):
        candidates_from_lightrag(SimpleNamespace(raw_data=None))
    with pytest.raises(TypeError, match="chunk 0"):
        candidates_from_lightrag({"data": {"chunks": ["bad"], "references": []}})


@pytest.mark.parametrize(
    "script",
    [
        "examples/integrations/hipporag.py",
        "examples/integrations/lightrag.py",
        "examples/end_to_end.py",
    ],
)
def test_public_examples_run_offline(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload
    if script.endswith(("hipporag.py", "lightrag.py")):
        assert payload["evidence"]
        assert payload["trace"]["input_method"] == "external_retriever"


def test_end_to_end_example_does_not_leave_checkout_artifacts() -> None:
    assert not Path("runs/example-end-to-end").exists()
