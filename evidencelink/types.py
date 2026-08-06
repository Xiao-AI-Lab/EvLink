"""Typed data boundaries for native coverage-aware evidence selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from evidencelink.contract import DEFAULT_READER_BUDGET_K, DEFAULT_STABILITY_WINDOW_M


@dataclass(frozen=True)
class EvidenceQueryState:
    dataset: str
    query_idx: int
    question: str
    query_id: str = ""
    gold_answers: tuple[str, ...] = ()
    gold_docs: tuple[str, ...] = ()
    gold_titles: tuple[str, ...] = ()
    pool_docs: tuple[str, ...] = ()
    pool_titles: tuple[str, ...] = ()
    pool_doc_ids: tuple[Any, ...] = ()
    pool_doc_scores: tuple[float, ...] = ()
    reader_budget_k: int = DEFAULT_READER_BUDGET_K
    stability_window_m: int = DEFAULT_STABILITY_WINDOW_M
    pool_trace: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pool_record(
        cls,
        record: Mapping[str, Any],
        *,
        dataset: str,
        reader_budget_k: int,
        stability_window_m: int,
    ) -> "EvidenceQueryState":
        pool_docs = tuple(str(item) for item in list(record.get("pool_docs") or []))
        pool_titles = tuple(
            str(item)
            for item in (
                list(record.get("pool_titles") or [])
                or [str(doc).split("\n", 1)[0].strip() for doc in pool_docs]
            )
        )
        raw_scores = list(record.get("pool_doc_scores") or [])
        if len(raw_scores) != len(pool_docs):
            raw_scores = [float(len(pool_docs) - idx) for idx in range(len(pool_docs))]
        raw_doc_ids = list(record.get("pool_doc_ids") or [])
        if len(raw_doc_ids) < len(pool_docs):
            raw_doc_ids.extend([None] * (len(pool_docs) - len(raw_doc_ids)))
        doc_ids: list[Any] = []
        for value in raw_doc_ids[: len(pool_docs)]:
            try:
                doc_ids.append(int(value) if value is not None and str(value).isdigit() else value)
            except (TypeError, ValueError):
                doc_ids.append(value)
        raw_query_id = record.get("query_id", record.get("query_idx", record.get("query_index", 0)))
        query_id = str(raw_query_id if raw_query_id is not None else "")
        raw_query_idx = record.get("query_index")
        if raw_query_idx is None:
            raw_query_idx = record.get("query_idx", 0)
        try:
            query_idx = int(raw_query_idx)
        except (TypeError, ValueError):
            query_idx = 0
        return cls(
            dataset=str(dataset),
            query_idx=query_idx,
            question=str(record.get("question") or ""),
            query_id=query_id,
            gold_answers=tuple(str(item) for item in list(record.get("gold_answers") or [])),
            gold_docs=tuple(str(item) for item in list(record.get("gold_docs") or [])),
            gold_titles=tuple(str(item) for item in list(record.get("gold_titles") or [])),
            pool_docs=pool_docs,
            pool_titles=pool_titles,
            pool_doc_ids=tuple(doc_ids),
            pool_doc_scores=tuple(float(value) for value in raw_scores[: len(pool_docs)]),
            reader_budget_k=int(reader_budget_k),
            stability_window_m=int(stability_window_m),
            pool_trace=dict(record.get("pool_trace") or {}),
        )


@dataclass(frozen=True)
class EvidenceSetScore:
    objective: float
    coverage_by_requirement: Mapping[str, float] = field(default_factory=dict)
    selected_binding_id: str | None = None
    binding_trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSelectionResult:
    final_positions: tuple[int, ...]
    final_docs: tuple[str, ...]
    final_titles: tuple[str, ...]
    baseline_positions: tuple[int, ...]
    stable_seed_positions: tuple[int, ...]
    admitted_positions: tuple[int, ...]
    trace: Mapping[str, Any]
    raw_selection_trace: Mapping[str, Any] = field(default_factory=dict)

    @property
    def protected_baseline_positions(self) -> tuple[int, ...]:
        """Baseline positions protected by the rank-stability guard."""
        return self.stable_seed_positions

    @property
    def protected_baseline_titles(self) -> tuple[str, ...]:
        """Baseline titles protected by the rank-stability guard."""
        return tuple(str(item) for item in self.trace.get("protected_baseline_titles", ()))


def ordered_unique_positions(values: Sequence[Any], *, pool_size: int, limit: int) -> list[int]:
    positions: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            pos = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < int(pool_size) and pos not in seen:
            seen.add(pos)
            positions.append(pos)
        if len(positions) >= int(limit):
            break
    return positions
