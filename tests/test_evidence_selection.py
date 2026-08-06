from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evidencelink.evidence_selection import compose_evidence_selection
from evidencelink.pool_alignment import (
    align_pool_record,
    build_doc_text_to_chunk_id,
)
from evidencelink.requirements import FrozenReportRequirementProvider
from evidencelink.types import EvidenceQueryState
from evidencelink.run_evidence_selection import build_report_row


class DummyUtilityProvider:
    def __init__(self, selected_positions):
        self.selected_positions = list(selected_positions)

    def select_positions(self, state, requirements):
        return self.selected_positions, {
            "selection": "dummy",
            "baseline_objective": 0.4,
            "selected_binding_id": "b0",
            "safe_projection_trace": {
                "safe_decision": "minimal_edit_applied",
                "baseline_objective": 0.4,
                "admission_steps": [
                    {
                        "step": 1,
                        "mode": "coverage_aware_admission",
                        "out_position": 4,
                        "out_title": "Noise",
                        "in_position": 7,
                        "in_title": "Gold Bridge",
                        "objective_gain": 0.11,
                        "objective": 0.51,
                    }
                ],
            },
        }


def make_state() -> EvidenceQueryState:
    return EvidenceQueryState(
        dataset="toy",
        query_idx=0,
        question="Question?",
        gold_titles=("A", "Gold Bridge"),
        pool_docs=tuple(f"Doc {idx}\nBody" for idx in range(10)),
        pool_titles=("A", "B", "C", "D", "Noise", "F", "G", "Gold Bridge", "I", "J"),
        pool_doc_ids=tuple(range(10)),
        pool_doc_scores=tuple(float(10 - idx) for idx in range(10)),
        reader_budget_k=3,
        stability_window_m=2,
    )


def test_compose_evidence_selection_uses_rank_stability_and_candidate_admission() -> None:
    result = compose_evidence_selection(
        make_state(),
        requirements=[],
        utility_provider=DummyUtilityProvider([0, 1, 7]),
    )

    assert result.final_positions == (0, 1, 7)
    assert result.stable_seed_positions == (0, 1)
    assert result.admitted_positions == (7,)
    assert result.trace["decision"] == "admit"
    assert result.trace["rank_stability_held"] is True
    assert result.trace["best_candidate_title"] == "Gold Bridge"


def test_frozen_report_requirement_provider_loads_by_question(tmp_path: Path) -> None:
    report = {
        "evidence_selection_query_traces": [
            {
                "question": "Question?",
                "selection_trace": {
                    "requirements": [
                        {
                            "unit_id": "s1",
                            "subquery": "Find A",
                            "depends_on": [],
                            "anchor_mentions": ["A"],
                        }
                    ]
                },
            }
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    provider = FrozenReportRequirementProvider(path)

    requirements = provider.get_requirements(make_state())

    assert len(requirements) == 1
    assert requirements[0].unit_id == "s1"
    assert requirements[0].subquery == "Find A"


def test_align_pool_record_remaps_doc_ids_to_passage_indices() -> None:
    corpus = [
        {"title": "Doc A", "text": "Body A"},
        {"title": "Doc B", "text": "Body B"},
    ]
    doc_text_to_chunk_id = build_doc_text_to_chunk_id(corpus)
    chunk_id = doc_text_to_chunk_id["Doc B\nBody B"]

    class DummyChunkStore:
        text_to_hash_id = {}

    class DummyIndex:
        chunk_embedding_store = DummyChunkStore()
        passage_node_key_to_doc_idx = {chunk_id: 17}

    aligned = align_pool_record(
        {
            "pool_docs": ["Doc B\nBody B"],
            "pool_titles": ["Doc B"],
            "pool_doc_ids": [1],
        },
        corpus=corpus,
        passage_index=DummyIndex(),
        doc_text_to_chunk_id=doc_text_to_chunk_id,
    )

    assert aligned["pool_doc_ids"] == [17]
    assert aligned["pool_titles"] == ["Doc B"]


def test_build_report_row_uses_external_pool_doc_ids_for_reader_context() -> None:
    state = EvidenceQueryState(
        dataset="toy",
        query_idx=0,
        question="Question?",
        gold_titles=("A",),
        pool_docs=("Doc A\nBody", "Doc B\nBody"),
        pool_titles=("Doc A", "Doc B"),
        pool_doc_ids=(1017, 1018),
        pool_doc_scores=(1.0, 0.5),
        reader_budget_k=2,
        stability_window_m=1,
        pool_trace={"external_pool_doc_ids": [7, 8]},
    )
    result = SimpleNamespace(
        final_positions=(0, 1),
        final_titles=("Doc A", "Doc B"),
        trace={},
        raw_selection_trace={},
    )

    row = build_report_row(
        state=state,
        result=result,
        source_row={"gold_doc_indices": [7]},
        requirement_count=0,
    )

    assert row["retrieved_doc_indices_top5"] == [7, 8]
