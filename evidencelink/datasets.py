"""Benchmark dataset registry for EvLink reproductions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for a benchmark dataset used in the paper."""

    name: str
    display_name: str
    task_family: str
    file_stem: str
    question_format: str
    corpus_format: str
    notes: str = ""

    @property
    def question_filename(self) -> str:
        return f"{self.file_stem}.json"

    @property
    def corpus_filename(self) -> str:
        return f"{self.file_stem}_corpus.json"

    def source_paths(self, source_root: str | Path) -> tuple[Path, Path]:
        root = Path(source_root)
        return root / self.question_filename, root / self.corpus_filename

    def to_mapping(self) -> dict[str, str]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "task_family": self.task_family,
            "file_stem": self.file_stem,
            "question_filename": self.question_filename,
            "corpus_filename": self.corpus_filename,
            "question_format": self.question_format,
            "corpus_format": self.corpus_format,
            "notes": self.notes,
        }


SUPPORTED_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="hotpotqa",
        display_name="HotpotQA",
        task_family="multi-hop QA",
        file_stem="hotpotqa",
        question_format="context/supporting_facts",
        corpus_format="title/text rows",
        notes="Uses supporting_facts titles to identify gold passages.",
    ),
    DatasetSpec(
        name="2wikimultihopqa",
        display_name="2WikiMultiHopQA",
        task_family="multi-hop QA",
        file_stem="2wikimultihopqa",
        question_format="context/supporting_facts",
        corpus_format="title/text rows",
        notes="Uses the same source format family as HotpotQA.",
    ),
    DatasetSpec(
        name="musique",
        display_name="MuSiQue",
        task_family="multi-hop QA",
        file_stem="musique",
        question_format="paragraphs/is_supporting",
        corpus_format="title/text rows",
        notes="Includes answer_aliases when present.",
    ),
    DatasetSpec(
        name="nq_rear",
        display_name="NQ",
        task_family="open-domain QA",
        file_stem="nq_rear",
        question_format="contexts/is_supporting",
        corpus_format="title/text rows",
        notes="Canonical local file stem for Natural Questions is nq_rear.",
    ),
    DatasetSpec(
        name="popqa",
        display_name="PopQA",
        task_family="open-domain QA",
        file_stem="popqa",
        question_format="paragraphs/is_supporting",
        corpus_format="title/text rows",
        notes="Includes object labels and aliases as accepted answers.",
    ),
)

DATASET_ALIASES: dict[str, str] = {
    "hotpot": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "2wiki_multihop_qa": "2wikimultihopqa",
    "2wiki-multihop-qa": "2wikimultihopqa",
    "2wikimultihop": "2wikimultihopqa",
    "musiqueqa": "musique",
    "nq": "nq_rear",
    "natural_questions": "nq_rear",
    "natural-questions": "nq_rear",
    "naturalquestions": "nq_rear",
}

_SPECS_BY_NAME = {spec.name: spec for spec in SUPPORTED_DATASETS}


def canonical_dataset_name(name: str) -> str:
    """Return the local canonical dataset name used by EvLink."""

    raw = str(name).strip().lower()
    return DATASET_ALIASES.get(raw, raw)


def dataset_spec(name: str) -> DatasetSpec:
    """Return metadata for a supported dataset."""

    canonical = canonical_dataset_name(name)
    try:
        return _SPECS_BY_NAME[canonical]
    except KeyError as exc:
        supported = ", ".join(spec.name for spec in SUPPORTED_DATASETS)
        raise ValueError(f"unsupported dataset {name!r}; supported datasets: {supported}") from exc


def iter_supported_datasets() -> Iterable[DatasetSpec]:
    """Iterate over the paper benchmark datasets in their reporting order."""

    return iter(SUPPORTED_DATASETS)


def supported_dataset_names() -> tuple[str, ...]:
    """Return canonical names for all paper benchmark datasets."""

    return tuple(spec.name for spec in SUPPORTED_DATASETS)
