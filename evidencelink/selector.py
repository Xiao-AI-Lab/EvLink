"""Public integration API for selecting evidence from external retrievers.

This module adapts an ordered candidate list to the EvLink artifact
contract, then runs evidence-need mining, support binding, and fixed-budget
coverage-aware selection. It does not claim that arbitrary external candidates
have source-grounded EvLink edges; that provenance remains the upstream
retriever's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from evidencelink.api import (
    build_evidence_needs_bq,
    build_support_cache,
    compose_final_evidence_rq,
)
from evidencelink.artifacts import CandidateEvidence, write_jsonl
from evidencelink.contract import (
    DEFAULT_BINDING_MAX_CANDIDATES,
    DEFAULT_MIN_COVERAGE_GAIN,
    DEFAULT_MIN_SWAP_GAIN,
    DEFAULT_POOL_K,
    DEFAULT_READER_BUDGET_K,
    validate_budgets,
)


@dataclass(frozen=True)
class EvidenceSelectorConfig:
    """Configuration for the external-retriever selection API."""

    dataset: str = "custom"
    reader_budget_k: int = DEFAULT_READER_BUDGET_K
    stability_window_m: int | None = None
    pool_k: int = DEFAULT_POOL_K
    evidence_need_mode: str = "whole_question"
    binding_mode: str = "simple"
    evidence_need_model: str = "gpt-4o-mini"
    binding_model: str = "simple-binding"
    llm_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout: float = 120.0
    max_steps: int = 5
    binding_max_candidates: int = DEFAULT_BINDING_MAX_CANDIDATES
    embedding_name: str = "deterministic-hash"
    embedding_base_url: str = "offline"
    embedding_batch_size: int = 8
    min_coverage_gain: float = DEFAULT_MIN_COVERAGE_GAIN
    min_swap_gain: float = DEFAULT_MIN_SWAP_GAIN

    def __post_init__(self) -> None:
        try:
            reader_budget_k = int(self.reader_budget_k)
            pool_k = int(self.pool_k)
            binding_max_candidates = int(self.binding_max_candidates)
            embedding_batch_size = int(self.embedding_batch_size)
            max_steps = int(self.max_steps)
            timeout = float(self.timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("selector numeric configuration values must be valid numbers") from exc

        validate_budgets(
            reader_budget_k=reader_budget_k,
            stability_window_m=self.resolved_stability_window_m,
        )
        if pool_k <= 0:
            raise ValueError(f"pool_k must be positive, got {self.pool_k}")
        if binding_max_candidates <= 0:
            raise ValueError(
                f"binding_max_candidates must be positive, got {self.binding_max_candidates}"
            )
        if embedding_batch_size <= 0:
            raise ValueError(f"embedding_batch_size must be positive, got {self.embedding_batch_size}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.evidence_need_mode not in {"llm", "whole_question", "anchor_list"}:
            raise ValueError(f"unsupported evidence_need_mode: {self.evidence_need_mode!r}")
        if self.binding_mode not in {"simple", "llm"}:
            raise ValueError(f"unsupported binding_mode: {self.binding_mode!r}")

    @property
    def resolved_stability_window_m(self) -> int:
        if self.stability_window_m is None:
            return max(0, int(self.reader_budget_k) - 1)
        return int(self.stability_window_m)


@dataclass(frozen=True)
class EvidenceSelection:
    """Structured result returned by :class:`EvidenceSelector`."""

    query_id: str
    question: str
    evidence: tuple[CandidateEvidence, ...]
    baseline_evidence: tuple[CandidateEvidence, ...]
    evidence_needs: tuple[Mapping[str, Any], ...]
    trace: Mapping[str, Any]
    summary: Mapping[str, Any]
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "evidence": [item.to_mapping() for item in self.evidence],
            "baseline_evidence": [item.to_mapping() for item in self.baseline_evidence],
            "evidence_needs": [dict(item) for item in self.evidence_needs],
            "trace": dict(self.trace),
            "summary": dict(self.summary),
            "artifacts": dict(self.artifacts),
        }


def _candidate_number(
    value: Mapping[str, Any],
    *,
    field_name: str,
    candidate_index: int,
    default: int | float,
    cast: type[int] | type[float],
) -> int | float:
    raw = value.get(field_name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"candidate {candidate_index} field {field_name!r} must be a valid {cast.__name__}, got {raw!r}"
        ) from exc


def _candidate_from_value(value: CandidateEvidence | Mapping[str, Any], index: int) -> CandidateEvidence:
    if isinstance(value, CandidateEvidence):
        value = value.to_mapping()
    if not isinstance(value, Mapping):
        raise TypeError(f"candidate {index} must be a CandidateEvidence or mapping")

    raw_text = str(value.get("text") or value.get("passage") or "").strip()
    title = str(value.get("title") or "").strip()
    if not title and "\n" in raw_text:
        title, raw_text = raw_text.split("\n", 1)
        title = title.strip()
        raw_text = raw_text.strip()
    raw_doc_id = next(
        (value.get(key) for key in ("doc_id", "id") if value.get(key) is not None and str(value.get(key)).strip()),
        index,
    )
    doc_id = str(raw_doc_id).strip()
    return CandidateEvidence(
        rank=int(
            _candidate_number(
                value,
                field_name="rank",
                candidate_index=index,
                default=index + 1,
                cast=int,
            )
        ),
        doc_id=doc_id,
        title=title or f"doc_{doc_id}",
        text=raw_text,
        source=str(value.get("source") or "external_retriever"),
        score=float(
            _candidate_number(
                value,
                field_name="score",
                candidate_index=index,
                default=float(-index),
                cast=float,
            )
        ),
        path=tuple(str(item) for item in list(value.get("path") or [])),
        edge_evidence=tuple(
            dict(item) for item in list(value.get("edge_evidence") or []) if isinstance(item, Mapping)
        ),
        metadata=dict(value.get("metadata") or value.get("upstream_metadata") or {}),
    )


def _pool_row(
    *,
    query_id: str,
    question: str,
    candidates: Sequence[CandidateEvidence],
    gold_doc_ids: Sequence[str],
    gold_titles: Sequence[str],
    gold_answers: Sequence[str],
    input_candidate_count: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "candidate_pool",
        "query_id": query_id,
        "query_idx": query_id,
        "question": question,
        "candidate_pool": [item.to_mapping() for item in candidates],
        "pool_docs": [f"{item.title}\n{item.text}".strip() for item in candidates],
        "pool_titles": [item.title for item in candidates],
        "pool_doc_ids": [item.doc_id for item in candidates],
        "pool_doc_scores": [item.score for item in candidates],
        "gold_doc_ids": [str(item) for item in gold_doc_ids],
        "gold_docs": [str(item) for item in gold_doc_ids],
        "gold_titles": [str(item) for item in gold_titles],
        "gold_answers": [str(item) for item in gold_answers],
        "pool_trace": {
            "input_method": "external_retriever",
            "external_pool_doc_ids": [item.doc_id for item in candidates],
            "input_candidate_count": int(input_candidate_count),
            "used_candidate_count": int(len(candidates)),
            "truncated": bool(input_candidate_count > len(candidates)),
        },
    }


class EvidenceSelector:
    """Select a traceable fixed-budget evidence set from external candidates."""

    def __init__(self, config: EvidenceSelectorConfig | None = None) -> None:
        self.config = config or EvidenceSelectorConfig()

    def select(
        self,
        *,
        question: str,
        candidates: Sequence[CandidateEvidence | Mapping[str, Any]],
        query_id: str = "query-0",
        workdir: str | Path | None = None,
        gold_doc_ids: Sequence[str] = (),
        gold_titles: Sequence[str] = (),
        gold_answers: Sequence[str] = (),
    ) -> EvidenceSelection:
        """Run coverage-aware selection over one ordered candidate list.

        When ``workdir`` is provided, the intermediate EvLink artifacts
        are retained for inspection. Otherwise they are created in a temporary
        directory and only the structured result is returned.
        """

        clean_question = str(question).strip()
        if not clean_question:
            raise ValueError("question must not be empty")
        if not candidates:
            raise ValueError("candidates must not be empty")
        input_candidate_count = len(candidates)
        normalized = tuple(
            _candidate_from_value(value, index)
            for index, value in enumerate(candidates[: int(self.config.pool_k)])
        )
        seen_doc_ids: dict[str, int] = {}
        for index, candidate in enumerate(normalized):
            previous = seen_doc_ids.get(candidate.doc_id)
            if previous is not None:
                raise ValueError(
                    f"candidate {index} doc_id {candidate.doc_id!r} duplicates candidate {previous}"
                )
            seen_doc_ids[candidate.doc_id] = index
        if workdir is None:
            with TemporaryDirectory(prefix="evidencelink-select-") as temporary:
                return self._select_in_workdir(
                    question=clean_question,
                    candidates=normalized,
                    query_id=str(query_id),
                    workdir=Path(temporary),
                    gold_doc_ids=gold_doc_ids,
                    gold_titles=gold_titles,
                    gold_answers=gold_answers,
                    input_candidate_count=input_candidate_count,
                    expose_artifacts=False,
                )
        return self._select_in_workdir(
            question=clean_question,
            candidates=normalized,
            query_id=str(query_id),
            workdir=Path(workdir),
            gold_doc_ids=gold_doc_ids,
            gold_titles=gold_titles,
            gold_answers=gold_answers,
            input_candidate_count=input_candidate_count,
            expose_artifacts=True,
        )

    def _select_in_workdir(
        self,
        *,
        question: str,
        candidates: tuple[CandidateEvidence, ...],
        query_id: str,
        workdir: Path,
        gold_doc_ids: Sequence[str],
        gold_titles: Sequence[str],
        gold_answers: Sequence[str],
        input_candidate_count: int,
        expose_artifacts: bool,
    ) -> EvidenceSelection:
        workdir.mkdir(parents=True, exist_ok=True)
        questions_path = workdir / "questions.jsonl"
        candidate_pool_path = workdir / "candidate_pool.jsonl"
        evidence_needs_path = workdir / "evidence_needs.jsonl"
        binding_cache_path = workdir / "binding_cache.json"
        selection_path = workdir / "evidence_selection.json"

        write_jsonl(
            [
                {
                    "query_id": query_id,
                    "question": question,
                    "gold_doc_ids": [str(item) for item in gold_doc_ids],
                    "gold_titles": [str(item) for item in gold_titles],
                    "gold_answers": [str(item) for item in gold_answers],
                }
            ],
            questions_path,
        )
        write_jsonl(
            [
                _pool_row(
                    query_id=query_id,
                    question=question,
                    candidates=candidates,
                    gold_doc_ids=gold_doc_ids,
                    gold_titles=gold_titles,
                    gold_answers=gold_answers,
                    input_candidate_count=input_candidate_count,
                )
            ],
            candidate_pool_path,
        )
        need_rows = build_evidence_needs_bq(
            questions_path=questions_path,
            output_path=evidence_needs_path,
            mode=self.config.evidence_need_mode,
            llm_base_url=self.config.llm_base_url,
            llm_model=self.config.evidence_need_model,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_steps=self.config.max_steps,
        )
        build_support_cache(
            candidate_pool_path=candidate_pool_path,
            evidence_needs_path=evidence_needs_path,
            output_path=binding_cache_path,
            mode=self.config.binding_mode,
            binding_model=self.config.binding_model,
            max_candidates=self.config.binding_max_candidates,
            llm_base_url=self.config.llm_base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            fallback=True,
        )
        payload = compose_final_evidence_rq(
            dataset=self.config.dataset,
            candidate_pool_path=candidate_pool_path,
            evidence_needs_path=evidence_needs_path,
            binding_cache_path=binding_cache_path,
            output_path=selection_path,
            reader_budget_k=min(int(self.config.reader_budget_k), len(candidates)),
            stability_window_m=min(
                self.config.resolved_stability_window_m,
                max(0, min(int(self.config.reader_budget_k), len(candidates)) - 1),
            ),
            pool_k=len(candidates),
            binding_model=self.config.binding_model,
            embedding_name=self.config.embedding_name,
            embedding_base_url=self.config.embedding_base_url,
            embedding_batch_size=self.config.embedding_batch_size,
            embedding_api_key=self.config.api_key,
            embedding_timeout=self.config.timeout,
            binding_max_candidates=self.config.binding_max_candidates,
            min_coverage_gain=self.config.min_coverage_gain,
            min_swap_gain=self.config.min_swap_gain,
            allow_missing_requirements=True,
        )
        rows = list(payload.get("rows") or [])
        if not rows:
            raise RuntimeError("EvLink selection returned no rows")
        row = dict(rows[0])
        trace = dict(row.get("evidence_selection") or {})
        selected_positions = [int(pos) for pos in list(trace.get("final_positions") or [])]
        baseline_positions = [int(pos) for pos in list(trace.get("baseline_positions") or [])]
        requirements = list((need_rows[0] if need_rows else {}).get("requirements") or [])
        artifacts = (
            {
                "questions": str(questions_path),
                "candidate_pool": str(candidate_pool_path),
                "evidence_needs": str(evidence_needs_path),
                "binding_cache": str(binding_cache_path),
                "evidence_selection": str(selection_path),
            }
            if expose_artifacts
            else {}
        )
        return EvidenceSelection(
            query_id=query_id,
            question=question,
            evidence=tuple(candidates[pos] for pos in selected_positions if 0 <= pos < len(candidates)),
            baseline_evidence=tuple(candidates[pos] for pos in baseline_positions if 0 <= pos < len(candidates)),
            evidence_needs=tuple(dict(item) for item in requirements if isinstance(item, Mapping)),
            trace={
                **trace,
                "input_method": "external_retriever",
                "input_candidate_count": int(input_candidate_count),
                "used_candidate_count": int(len(candidates)),
                "truncated": bool(input_candidate_count > len(candidates)),
                "raw_selection_trace": dict(row.get("raw_selection_trace") or {}),
            },
            summary=dict(payload.get("summary") or {}),
            artifacts=artifacts,
        )


def select_evidence(
    *,
    question: str,
    candidates: Sequence[CandidateEvidence | Mapping[str, Any]],
    config: EvidenceSelectorConfig | None = None,
    query_id: str = "query-0",
    workdir: str | Path | None = None,
    gold_doc_ids: Sequence[str] = (),
    gold_titles: Sequence[str] = (),
    gold_answers: Sequence[str] = (),
) -> EvidenceSelection:
    """One-shot convenience wrapper for :class:`EvidenceSelector`."""

    return EvidenceSelector(config).select(
        question=question,
        candidates=candidates,
        query_id=query_id,
        workdir=workdir,
        gold_doc_ids=gold_doc_ids,
        gold_titles=gold_titles,
        gold_answers=gold_answers,
    )
