from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidencelink.artifacts import read_jsonl
from evidencelink.datasets import canonical_dataset_name, dataset_spec, supported_dataset_names
from evidencelink.prepare_dataset import prepare_benchmark_dataset


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def write_source_fixtures(source_root: Path) -> None:
    write_json(
        source_root / "hotpotqa_corpus.json",
        [
            {"idx": 0, "title": "Alpha", "text": "Alpha text."},
            {"idx": 1, "title": "Beta", "text": "Beta supporting sentence."},
        ],
    )
    write_json(
        source_root / "hotpotqa.json",
        [
            {
                "_id": "hotpot-1",
                "question": "Which page supports Beta?",
                "answer": "Beta",
                "context": [["Alpha", ["Alpha text."]], ["Beta", ["Beta supporting sentence."]]],
                "supporting_facts": [["Beta", 0]],
            }
        ],
    )

    write_json(
        source_root / "2wikimultihopqa_corpus.json",
        [
            {"title": "Lothair II", "text": "Lothair's mother was Ermengarde."},
            {"title": "Ermengarde of Tours", "text": "Ermengarde died in 851."},
        ],
    )
    write_json(
        source_root / "2wikimultihopqa.json",
        [
            {
                "_id": "2wiki-1",
                "question": "When did Lothair II's mother die?",
                "answer": "851",
                "context": [
                    ["Lothair II", ["Lothair's mother was Ermengarde."]],
                    ["Ermengarde of Tours", ["Ermengarde died in 851."]],
                ],
                "supporting_facts": [["Lothair II", 0], ["Ermengarde of Tours", 0]],
            }
        ],
    )

    write_json(
        source_root / "musique_corpus.json",
        [
            {"title": "FC Barcelona", "text": "Barcelona signed Diego Maradona."},
            {"title": "Diego Maradona", "text": "He joined Barcelona in June 1982."},
        ],
    )
    write_json(
        source_root / "musique.json",
        [
            {
                "id": "musique-1",
                "question": "When did Maradona join Barcelona?",
                "answer": "June 1982",
                "answer_aliases": ["1982"],
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": "FC Barcelona",
                        "paragraph_text": "Barcelona signed Diego Maradona.",
                        "is_supporting": False,
                    },
                    {
                        "idx": 1,
                        "title": "Diego Maradona",
                        "paragraph_text": "He joined Barcelona in June 1982.",
                        "is_supporting": True,
                    },
                ],
            }
        ],
    )

    write_json(
        source_root / "nq_rear_corpus.json",
        [{"idx": 0, "title": "Eddie Gottlieb", "text": "Original NBA teams include the Knicks."}],
    )
    write_json(
        source_root / "nq_rear.json",
        [
            {
                "question": "Who has one of the oldest teams in the NBA?",
                "reference": ["New York Knickerbockers"],
                "contexts": [
                    {
                        "title": "Eddie Gottlieb",
                        "text": "Original NBA teams include the Knicks.",
                        "is_supporting": True,
                    }
                ],
            }
        ],
    )

    write_json(
        source_root / "popqa_corpus.json",
        [
            {"title": "George Rankin", "text": "George Rankin was an Australian politician."},
            {"title": "Politician", "text": "A politician participates in policy-making."},
        ],
    )
    write_json(
        source_root / "popqa.json",
        [
            {
                "id": 4222362,
                "question": "What was George Rankin's occupation?",
                "obj": "politician",
                "possible_answers": '["politician", "political leader"]',
                "o_wiki_title": "Politician",
                "o_aliases": '["pol"]',
                "paragraphs": [
                    {
                        "title": "George Rankin",
                        "text": "George Rankin was an Australian politician.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Politician",
                        "text": "A politician participates in policy-making.",
                        "is_supporting": True,
                    },
                ],
            }
        ],
    )


def test_dataset_registry_covers_paper_benchmarks() -> None:
    assert supported_dataset_names() == ("hotpotqa", "2wikimultihopqa", "musique", "nq_rear", "popqa")
    assert canonical_dataset_name("nq") == "nq_rear"
    assert dataset_spec("natural_questions").display_name == "NQ"


def test_prepare_hotpotqa_dataset_to_standard_inputs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_source_fixtures(source_root)

    manifest = prepare_benchmark_dataset(
        dataset="hotpotqa",
        source_root=source_root,
        output_root=tmp_path / "prepared" / "hotpotqa",
    )
    questions = read_jsonl(manifest["questions_path"])
    corpus = read_jsonl(manifest["corpus_path"])

    assert manifest["dataset"] == "hotpotqa"
    assert manifest["question_count"] == 1
    assert corpus[1]["doc_id"] == "1"
    assert corpus[1]["metadata"]["source_idx"] == 1
    assert questions[0]["query_id"] == "hotpot-1"
    assert questions[0]["gold_doc_ids"] == ["1"]
    assert questions[0]["gold_titles"] == ["Beta"]
    assert questions[0]["gold_answers"] == ["Beta"]


def test_prepare_all_supported_dataset_formats(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_source_fixtures(source_root)

    datasets = ["2wikimultihopqa", "musique", "nq", "popqa"]
    manifests = {
        name: prepare_benchmark_dataset(
            dataset=name,
            source_root=source_root,
            output_root=tmp_path / "prepared" / name,
        )
        for name in datasets
    }

    wiki_questions = read_jsonl(manifests["2wikimultihopqa"]["questions_path"])
    musique_questions = read_jsonl(manifests["musique"]["questions_path"])
    nq_questions = read_jsonl(manifests["nq"]["questions_path"])
    popqa_questions = read_jsonl(manifests["popqa"]["questions_path"])

    assert wiki_questions[0]["gold_doc_ids"] == ["0", "1"]
    assert musique_questions[0]["gold_doc_ids"] == ["1"]
    assert musique_questions[0]["gold_answers"] == ["June 1982", "1982"]
    assert manifests["nq"]["dataset"] == "nq_rear"
    assert nq_questions[0]["gold_doc_ids"] == ["0"]
    assert nq_questions[0]["gold_answers"] == ["New York Knickerbockers"]
    assert popqa_questions[0]["gold_doc_ids"] == ["0", "1"]
    assert popqa_questions[0]["gold_answers"] == ["politician", "political leader", "Politician", "pol"]
