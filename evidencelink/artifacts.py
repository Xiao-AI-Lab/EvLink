"""Artifact schemas and JSON helpers for the public EvidenceLink pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passage(self) -> str:
        return f"{self.title}\n{self.text}".strip()

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], fallback_id: int = 0) -> "Document":
        raw_doc_id = row.get("doc_id", row.get("id", row.get("idx", fallback_id)))
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or row.get("passage") or "").strip()
        if not title and "\n" in text:
            title, text = text.split("\n", 1)
            title = title.strip()
            text = text.strip()
        if not title:
            title = f"doc_{raw_doc_id}"
        return cls(
            doc_id=str(raw_doc_id),
            title=title,
            text=text,
            metadata=dict(row.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Question:
    query_id: str
    question: str
    gold_doc_ids: tuple[str, ...] = ()
    gold_titles: tuple[str, ...] = ()
    gold_answers: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], fallback_id: int = 0) -> "Question":
        query_id = str(row.get("query_id", row.get("query_idx", row.get("query_index", fallback_id))))
        return cls(
            query_id=query_id,
            question=str(row.get("question") or row.get("query") or "").strip(),
            gold_doc_ids=tuple(str(item) for item in list(row.get("gold_doc_ids") or row.get("gold_doc_indices") or [])),
            gold_titles=tuple(str(item) for item in list(row.get("gold_titles") or [])),
            gold_answers=tuple(str(item) for item in list(row.get("gold_answers") or row.get("answers") or [])),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_doc_ids"] = list(self.gold_doc_ids)
        payload["gold_titles"] = list(self.gold_titles)
        payload["gold_answers"] = list(self.gold_answers)
        return payload


@dataclass(frozen=True)
class OpenIEFact:
    doc_id: str
    fact_id: str
    subject: str
    relation: str
    object: str
    source_span: str = ""
    confidence: float = 1.0
    raw_text: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], fallback_id: int = 0) -> "OpenIEFact":
        triple = row.get("triple") or row.get("extracted_triple")
        if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)) and len(triple) >= 3:
            subject, relation, obj = triple[:3]
        else:
            subject = row.get("subject", row.get("head", ""))
            relation = row.get("relation", row.get("predicate", ""))
            obj = row.get("object", row.get("tail", ""))
        doc_id = str(row.get("doc_id", row.get("idx", "")))
        fact_id = str(row.get("fact_id", f"{doc_id}:f{fallback_id}"))
        return cls(
            doc_id=doc_id,
            fact_id=fact_id,
            subject=str(subject or "").strip(),
            relation=str(relation or "").strip(),
            object=str(obj or "").strip(),
            source_span=str(row.get("source_span") or row.get("sentence") or ""),
            confidence=float(row.get("confidence", 1.0) or 1.0),
            raw_text=str(row.get("raw_text") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvidence:
    rank: int
    doc_id: str
    title: str
    text: str
    source: str
    score: float = 0.0
    path: tuple[str, ...] = ()
    edge_evidence: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = list(self.path)
        payload["edge_evidence"] = [dict(item) for item in self.edge_evidence]
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class CandidatePoolRecord:
    query_id: str
    question: str
    anchors: tuple[str, ...]
    seed_doc_ids: tuple[str, ...]
    candidate_pool: tuple[CandidateEvidence, ...]
    gold_doc_ids: tuple[str, ...] = ()
    gold_titles: tuple[str, ...] = ()
    gold_answers: tuple[str, ...] = ()
    local_subgraph: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_idx": self.query_id,
            "question": self.question,
            "anchors": list(self.anchors),
            "seed_doc_ids": list(self.seed_doc_ids),
            "candidate_pool": [item.to_mapping() for item in self.candidate_pool],
            "gold_doc_ids": list(self.gold_doc_ids),
            "gold_docs": list(self.gold_doc_ids),
            "gold_titles": list(self.gold_titles),
            "gold_answers": list(self.gold_answers),
            "local_subgraph": dict(self.local_subgraph),
            "pool_docs": [f"{item.title}\n{item.text}".strip() for item in self.candidate_pool],
            "pool_titles": [item.title for item in self.candidate_pool],
            "pool_doc_ids": [item.doc_id for item in self.candidate_pool],
            "pool_doc_scores": [item.score for item in self.candidate_pool],
            "pool_trace": {
                "anchors": list(self.anchors),
                "seed_doc_ids": list(self.seed_doc_ids),
                "local_subgraph": dict(self.local_subgraph),
            },
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            rows.append(json.loads(clean))
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_json_or_jsonl(path: str | Path) -> Any:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if input_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_documents(path: str | Path) -> list[Document]:
    payload = read_json_or_jsonl(path)
    if isinstance(payload, Mapping):
        if "doc_id" in payload or "text" in payload or "passage" in payload:
            raw_rows = [payload]
        else:
            raw_rows = payload.get("documents", payload.get("docs", payload.get("corpus", [])))
    else:
        raw_rows = payload
    return [Document.from_mapping(row, idx) for idx, row in enumerate(list(raw_rows or [])) if isinstance(row, Mapping)]


def load_questions(path: str | Path) -> list[Question]:
    payload = read_json_or_jsonl(path)
    if isinstance(payload, Mapping):
        if "question" in payload or "query" in payload:
            raw_rows = [payload]
        else:
            raw_rows = payload.get("questions", payload.get("rows", payload.get("queries", [])))
    else:
        raw_rows = payload
    return [Question.from_mapping(row, idx) for idx, row in enumerate(list(raw_rows or [])) if isinstance(row, Mapping)]


def load_openie_facts(path: str | Path) -> list[OpenIEFact]:
    payload = read_json_or_jsonl(path)
    if isinstance(payload, Mapping):
        if "subject" in payload or "triple" in payload or "extracted_triples" in payload:
            raw_rows = [payload]
        else:
            raw_rows = payload.get("facts", payload.get("openie_facts", []))
    else:
        raw_rows = payload
    facts: list[OpenIEFact] = []
    for idx, row in enumerate(list(raw_rows or [])):
        if not isinstance(row, Mapping):
            continue
        if isinstance(row.get("extracted_triples"), list):
            doc_id = str(row.get("doc_id", row.get("idx", idx)))
            for fact_idx, triple in enumerate(row.get("extracted_triples") or []):
                facts.append(OpenIEFact.from_mapping({"doc_id": doc_id, "fact_id": f"{doc_id}:f{fact_idx}", "triple": triple}, fact_idx))
        else:
            facts.append(OpenIEFact.from_mapping(row, idx))
    return facts
