from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from evidencelink import reader_qa


def test_reader_separates_same_line_citations_from_answer() -> None:
    response = "Answer: Paris Citations: [1]"

    assert reader_qa.extract_answer(response) == "Paris"
    assert reader_qa.extract_citation_positions(response, passage_count=1) == [0]


def test_reader_exports_passage_level_citations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = {
        "method": "evlink",
        "dataset": "toy",
        "rows": [
            {
                "query_id": "q-1",
                "query_index": 0,
                "question": "Where was Alice born?",
                "gold_answers": ["Paris"],
                "retrieved_doc_indices_top5": ["doc-a", "doc-b"],
                "retrieved_docs_top5": [
                    "Alice\nAlice was born in Paris.",
                    "Paris\nParis is in France.",
                ],
                "retrieved_titles_top5": ["Alice", "Paris"],
            }
        ],
    }
    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        reader_qa,
        "chat_completion",
        lambda **_: "Answer: Paris\nCitations: [1] [2] [1] [9]",
    )

    payload = reader_qa.run_reader(
        Namespace(
            retrieval_report=report_path,
            max_queries=0,
            qa_top_k=2,
            llm_base_url="offline",
            llm_name="fixture-reader",
            api_key="",
            timeout=1.0,
        )
    )

    row = payload["rows"][0]
    assert row["query_id"] == "q-1"
    assert row["prediction"] == "Paris"
    assert row["em"] == 1.0
    assert row["citations"] == [
        {
            "marker": "[1]",
            "passage_position": 0,
            "passage_id": "doc-a",
            "passage_title": "Alice",
        },
        {
            "marker": "[2]",
            "passage_position": 1,
            "passage_id": "doc-b",
            "passage_title": "Paris",
        },
    ]
    assert "Citations:" in reader_qa.build_prompt("Question?", ["Passage"])[1]["content"]
