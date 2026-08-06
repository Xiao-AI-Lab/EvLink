from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from evidencelink import (
    EvidenceSelection,
    EvidenceSelector,
    EvidenceSelectorConfig,
    select_evidence,
)


CANDIDATES = [
    {
        "doc_id": "d0",
        "title": "Acme Corporation",
        "text": "Acme Corporation was founded by Alice Chen in 2012.",
        "score": 0.98,
    },
    {
        "doc_id": "d1",
        "title": "Alice Chen",
        "text": "Alice Chen was born in Singapore.",
        "score": 0.94,
    },
    {
        "doc_id": "d2",
        "title": "Company history",
        "text": "Acme opened its first European office in 2018.",
        "score": 0.72,
    },
]


def test_external_selector_returns_structured_evidence_without_persisting_artifacts() -> None:
    result = EvidenceSelector(
        EvidenceSelectorConfig(reader_budget_k=2, evidence_need_mode="anchor_list")
    ).select(
        question="Who founded Acme Corporation and where was the founder born?",
        candidates=CANDIDATES,
    )

    assert isinstance(result, EvidenceSelection)
    assert len(result.evidence) == 2
    assert [item.doc_id for item in result.evidence] == ["d0", "d1"]
    assert result.evidence_needs
    assert result.trace["reader_budget_k"] == 2
    assert result.trace["input_method"] == "external_retriever"
    assert result.artifacts == {}
    assert result.to_mapping()["evidence"][0]["source"] == "external_retriever"


def test_external_selector_can_retain_standard_artifacts(tmp_path: Path) -> None:
    result = select_evidence(
        question="Who founded Acme Corporation?",
        candidates=CANDIDATES,
        config=EvidenceSelectorConfig(reader_budget_k=2),
        query_id="acme-1",
        workdir=tmp_path / "selection",
    )

    assert set(result.artifacts) == {
        "questions",
        "candidate_pool",
        "evidence_needs",
        "binding_cache",
        "evidence_selection",
    }
    assert all(Path(path).exists() for path in result.artifacts.values())
    pool_row = json.loads(Path(result.artifacts["candidate_pool"]).read_text().splitlines()[0])
    assert pool_row["pool_trace"]["input_method"] == "external_retriever"
    assert pool_row["pool_doc_ids"] == ["d0", "d1", "d2"]
    selection = json.loads(Path(result.artifacts["evidence_selection"]).read_text())
    selection_row = selection["rows"][0]
    assert selection_row["query_id"] == "acme-1"
    assert selection_row["evidence_selection"]["input_method"] == "external_retriever"


def test_external_selector_preserves_upstream_candidate_provenance() -> None:
    result = EvidenceSelector(EvidenceSelectorConfig(reader_budget_k=1)).select(
        question="What does the linked passage establish?",
        candidates=[
            {
                "id": "upstream-7",
                "passage": "Linked passage\nThe passage establishes the bridge fact.",
                "rank": 7,
                "score": 0.37,
                "source": "hipporag",
                "path": ["seed", "bridge"],
                "edge_evidence": [{"relation": "founded_by", "witness": "source span"}],
                "metadata": {"collection": "wiki"},
            }
        ],
    )

    evidence = result.evidence[0]
    assert evidence.doc_id == "upstream-7"
    assert evidence.title == "Linked passage"
    assert evidence.text == "The passage establishes the bridge fact."
    assert evidence.rank == 7
    assert evidence.score == 0.37
    assert evidence.source == "hipporag"
    assert evidence.path == ("seed", "bridge")
    assert evidence.edge_evidence[0]["witness"] == "source span"
    assert evidence.metadata["collection"] == "wiki"


def test_external_selector_validates_public_inputs() -> None:
    selector = EvidenceSelector()
    with pytest.raises(ValueError, match="question"):
        selector.select(question="", candidates=CANDIDATES)
    with pytest.raises(ValueError, match="candidates"):
        selector.select(question="Question?", candidates=[])
    with pytest.raises(TypeError, match="candidate 0"):
        selector.select(question="Question?", candidates=["invalid"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="pool_k"):
        EvidenceSelectorConfig(pool_k=0)
    with pytest.raises(ValueError, match="stability_window_m"):
        EvidenceSelectorConfig(stability_window_m=-1)
    with pytest.raises(ValueError, match="evidence_need_mode"):
        EvidenceSelectorConfig(evidence_need_mode="unknown")


def test_external_selector_normalizes_none_values_and_rejects_duplicate_ids() -> None:
    result = EvidenceSelector(EvidenceSelectorConfig(reader_budget_k=2)).select(
        question="Question?",
        candidates=[
            {"doc_id": None, "rank": None, "score": None, "title": "A", "text": "A text"},
            {"id": "b", "rank": "2", "score": "0.5", "title": "B", "text": "B text"},
        ],
    )

    assert [item.doc_id for item in result.baseline_evidence] == ["0", "b"]
    assert result.baseline_evidence[0].rank == 1
    assert result.baseline_evidence[0].score == 0.0
    assert result.baseline_evidence[1].rank == 2
    assert result.baseline_evidence[1].score == 0.5

    with pytest.raises(ValueError, match="duplicates candidate 0"):
        EvidenceSelector().select(
            question="Question?",
            candidates=[
                {"doc_id": "same", "title": "A", "text": "A"},
                {"id": "same", "title": "B", "text": "B"},
            ],
        )


def test_external_selector_reports_truncation_and_clamps_budget() -> None:
    result = EvidenceSelector(EvidenceSelectorConfig(reader_budget_k=5, pool_k=2)).select(
        question="Question?",
        candidates=CANDIDATES,
    )

    assert len(result.evidence) == 2
    assert result.trace["input_candidate_count"] == 3
    assert result.trace["used_candidate_count"] == 2
    assert result.trace["truncated"] is True


def test_sdk_artifacts_round_trip_through_stage_cli_defaults(tmp_path: Path) -> None:
    result = EvidenceSelector(EvidenceSelectorConfig(reader_budget_k=2)).select(
        question="Who founded Acme Corporation?",
        candidates=CANDIDATES,
        query_id="acme-round-trip",
        workdir=tmp_path / "sdk",
    )
    cli_output = tmp_path / "cli-selection.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evidencelink.run_evidence_selection",
            "--dataset",
            "custom",
            "--pool-json",
            result.artifacts["candidate_pool"],
            "--requirement-report",
            result.artifacts["evidence_needs"],
            "--binding-cache-path",
            result.artifacts["binding_cache"],
            "--output-json",
            str(cli_output),
            "--reader-budget-k",
            "2",
            "--stability-window-m",
            "1",
            "--pool-k",
            "3",
            "--allow-missing-requirements",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(cli_output.read_text())

    assert payload["rows"][0]["query_id"] == "acme-round-trip"
    assert payload["rows"][0]["evidence_selection"]["final_positions"] == result.trace["final_positions"]
    assert payload["rows"][0]["evidence_selection"]["input_method"] == "external_retriever"


def test_external_retriever_example_runs_from_source_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/external_retriever.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["evidence"]
    assert payload["trace"]["input_method"] == "external_retriever"
