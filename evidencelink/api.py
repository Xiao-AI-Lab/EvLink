"""Paper-facing API for the public EvLink package.

This module keeps the public method boundary aligned with the paper notation:

    corpus + questions -> C_q -> B(q) -> support cache -> R_q

The lower-level modules remain available for tests and stage-specific tools,
but new integrations should prefer the functions in this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from evidencelink.artifacts import OpenIEFact, write_jsonl
from evidencelink.binding import build_binding_cache
from evidencelink.contract import (
    DEFAULT_BINDING_MAX_CANDIDATES,
    DEFAULT_MIN_COVERAGE_GAIN,
    DEFAULT_MIN_SWAP_GAIN,
    DEFAULT_POOL_K,
    DEFAULT_READER_BUDGET_K,
    DEFAULT_STABILITY_WINDOW_M,
    METHOD_CONTRACT,
    PAPER_FACING_METHOD_NAME,
)
from evidencelink.evidence_needs import build_evidence_need_rows
from evidencelink.index import build_evidence_link_index
from evidencelink.induction import build_candidate_pool_records, write_candidate_pool
from evidencelink.io_utils import write_json
from evidencelink.openie import build_openie_facts
from evidencelink.run_evidence_selection import run_evidence_selection


@dataclass(frozen=True)
class PaperPipelineArtifacts:
    """Standard EvLink artifact paths for one paper-facing run."""

    workdir: Path
    openie_facts: Path
    evidence_link_index: Path
    candidate_pool: Path
    evidence_needs: Path
    binding_cache: Path
    evidence_selection: Path

    @classmethod
    def from_workdir(cls, workdir: str | Path) -> "PaperPipelineArtifacts":
        root = Path(workdir)
        return cls(
            workdir=root,
            openie_facts=root / "openie_facts.jsonl",
            evidence_link_index=root / "evidence_link_index.json",
            candidate_pool=root / "candidate_pool.jsonl",
            evidence_needs=root / "evidence_needs.jsonl",
            binding_cache=root / "binding_cache.json",
            evidence_selection=root / "evidence_selection.json",
        )

    def to_mapping(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class PaperPipelineConfig:
    """Configuration for the paper-facing EvLink runner."""

    dataset: str = "custom"
    reader_budget_k: int = DEFAULT_READER_BUDGET_K
    stability_window_m: int = DEFAULT_STABILITY_WINDOW_M
    pool_k: int = DEFAULT_POOL_K
    dense_top_k: int = 20
    max_hops: int = 2
    max_queries: int = 0
    openie_mode: str = "simple"
    evidence_need_mode: str = "whole_question"
    binding_mode: str = "simple"
    openie_model: str = "gpt-4o-mini"
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
    allow_missing_requirements: bool = True
    force: bool = False
    dry_run: bool = False

    @classmethod
    def from_namespace(cls, args: Any) -> "PaperPipelineConfig":
        def value(name: str, default: Any) -> Any:
            return getattr(args, name, default)

        return cls(
            dataset=str(value("dataset", cls.dataset)),
            reader_budget_k=int(value("reader_budget_k", cls.reader_budget_k)),
            stability_window_m=int(value("stability_window_m", cls.stability_window_m)),
            pool_k=int(value("pool_k", cls.pool_k)),
            dense_top_k=int(value("dense_top_k", cls.dense_top_k)),
            max_hops=int(value("max_hops", cls.max_hops)),
            max_queries=int(value("max_queries", cls.max_queries)),
            openie_mode=str(value("openie_mode", cls.openie_mode)),
            evidence_need_mode=str(value("evidence_need_mode", cls.evidence_need_mode)),
            binding_mode=str(value("binding_mode", cls.binding_mode)),
            openie_model=str(value("openie_model", cls.openie_model)),
            evidence_need_model=str(value("evidence_need_model", cls.evidence_need_model)),
            binding_model=str(value("binding_model", cls.binding_model)),
            llm_base_url=str(value("llm_base_url", cls.llm_base_url)),
            api_key=str(value("api_key", cls.api_key)),
            timeout=float(value("timeout", cls.timeout)),
            max_steps=int(value("max_steps", cls.max_steps)),
            binding_max_candidates=int(value("binding_max_candidates", cls.binding_max_candidates)),
            embedding_name=str(value("embedding_name", cls.embedding_name)),
            embedding_base_url=str(value("embedding_base_url", cls.embedding_base_url)),
            embedding_batch_size=int(value("embedding_batch_size", cls.embedding_batch_size)),
            min_coverage_gain=float(value("min_coverage_gain", cls.min_coverage_gain)),
            min_swap_gain=float(value("min_swap_gain", cls.min_swap_gain)),
            allow_missing_requirements=bool(value("allow_missing_requirements", cls.allow_missing_requirements)),
            force=bool(value("force", cls.force)),
            dry_run=bool(value("dry_run", cls.dry_run)),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize configuration without exposing runtime credentials."""

        payload = asdict(self)
        payload["api_key"] = "<redacted>" if self.api_key else ""
        payload["api_key_configured"] = bool(self.api_key)
        return payload


def build_openie_artifact(
    *,
    corpus_path: str | Path,
    output_path: str | Path,
    mode: str = "simple",
    llm_base_url: str = "https://api.openai.com/v1",
    llm_model: str = "gpt-4o-mini",
    api_key: str = "",
    timeout: float = 120.0,
    fallback: bool = True,
) -> list[OpenIEFact]:
    """Build source-grounded OpenIE facts used as evidence-link witnesses."""

    return build_openie_facts(
        SimpleNamespace(
            corpus=Path(corpus_path),
            output=Path(output_path),
            mode=mode,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            api_key=api_key,
            timeout=timeout,
            fallback=fallback,
        )
    )


def build_candidate_pool_cq(
    *,
    questions_path: str | Path,
    corpus_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    dense_top_k: int = 20,
    max_hops: int = 2,
    pool_k: int = DEFAULT_POOL_K,
) -> list[Any]:
    """Build query-local candidate pools C_q and write them as JSONL."""

    records = build_candidate_pool_records(
        questions_path=questions_path,
        corpus_path=corpus_path,
        index_path=index_path,
        dense_top_k=dense_top_k,
        max_hops=max_hops,
        pool_k=pool_k,
    )
    write_candidate_pool(records, output_path)
    return records


def build_evidence_needs_bq(
    *,
    questions_path: str | Path,
    output_path: str | Path,
    mode: str = "whole_question",
    llm_base_url: str = "https://api.openai.com/v1",
    llm_model: str = "gpt-4o-mini",
    api_key: str = "",
    timeout: float = 120.0,
    max_steps: int = 5,
) -> list[dict[str, object]]:
    """Build evidence-need sets B(q) and write them as JSONL."""

    rows = build_evidence_need_rows(
        SimpleNamespace(
            questions=Path(questions_path),
            output=Path(output_path),
            mode=mode,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            api_key=api_key,
            timeout=timeout,
            max_steps=max_steps,
        )
    )
    write_jsonl(rows, output_path)
    return rows


def build_support_cache(
    *,
    candidate_pool_path: str | Path,
    evidence_needs_path: str | Path,
    output_path: str | Path,
    mode: str = "simple",
    binding_model: str = "simple-binding",
    max_candidates: int = DEFAULT_BINDING_MAX_CANDIDATES,
    llm_base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
    timeout: float = 120.0,
    fallback: bool = True,
) -> dict[str, list[str]]:
    """Build the support cache used by coverage-aware evidence selection."""

    return build_binding_cache(
        SimpleNamespace(
            candidate_pool=Path(candidate_pool_path),
            evidence_needs=Path(evidence_needs_path),
            output=Path(output_path),
            mode=mode,
            binding_model=binding_model,
            max_candidates=max_candidates,
            llm_base_url=llm_base_url,
            api_key=api_key,
            timeout=timeout,
            fallback=fallback,
        )
    )


def compose_final_evidence_rq(
    *,
    dataset: str,
    candidate_pool_path: str | Path,
    evidence_needs_path: str | Path,
    binding_cache_path: str | Path,
    output_path: str | Path,
    reader_budget_k: int = DEFAULT_READER_BUDGET_K,
    stability_window_m: int = DEFAULT_STABILITY_WINDOW_M,
    pool_k: int = DEFAULT_POOL_K,
    max_queries: int = 0,
    binding_model: str = "simple-binding",
    embedding_name: str = "deterministic-hash",
    embedding_base_url: str = "offline",
    embedding_batch_size: int = 8,
    embedding_api_key: str = "",
    embedding_timeout: float = 120.0,
    binding_max_candidates: int = DEFAULT_BINDING_MAX_CANDIDATES,
    min_coverage_gain: float = DEFAULT_MIN_COVERAGE_GAIN,
    min_swap_gain: float = DEFAULT_MIN_SWAP_GAIN,
    allow_missing_requirements: bool = True,
) -> dict[str, Any]:
    """Compose the final selected evidence set R_q from C_q, B(q), and support cache."""

    output = Path(output_path)
    payload = run_evidence_selection(
        SimpleNamespace(
            dataset=dataset,
            limit=0,
            max_queries=max_queries,
            pool_k=pool_k,
            reader_budget_k=reader_budget_k,
            stability_window_m=stability_window_m,
            pool_json=Path(candidate_pool_path),
            requirement_report=Path(evidence_needs_path),
            binding_cache_path=Path(binding_cache_path),
            output_json=output,
            output_root=output.parent,
            llm_binding_model=binding_model,
            embedding_name=embedding_name,
            embedding_base_url=embedding_base_url,
            embedding_batch_size=embedding_batch_size,
            embedding_api_key=embedding_api_key,
            embedding_timeout=embedding_timeout,
            binding_max_candidates=binding_max_candidates,
            llm_binding_title_match_mode="wiki_title",
            min_coverage_gain=min_coverage_gain,
            min_swap_gain=min_swap_gain,
            allow_missing_requirements=allow_missing_requirements,
        )
    )
    write_json(payload, output)
    return payload


def _step_payload(name: str, output: Path, status: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "output": str(output), **extra}


def _should_skip(path: Path, *, force: bool) -> bool:
    return path.exists() and not force


def run_paper_pipeline(
    *,
    corpus_path: str | Path,
    questions_path: str | Path,
    workdir: str | Path,
    config: PaperPipelineConfig | None = None,
) -> dict[str, Any]:
    """Run the paper-facing EvLink pipeline end to end."""

    cfg = config or PaperPipelineConfig()
    artifacts = PaperPipelineArtifacts.from_workdir(workdir)
    steps: list[dict[str, Any]] = []
    if cfg.dry_run:
        return {
            "method": PAPER_FACING_METHOD_NAME,
            "method_contract": dict(METHOD_CONTRACT),
            "config": cfg.to_mapping(),
            "artifacts": artifacts.to_mapping(),
            "steps": [
                _step_payload("build_openie_facts", artifacts.openie_facts, "planned"),
                _step_payload("build_evidence_link_index", artifacts.evidence_link_index, "planned"),
                _step_payload("build_candidate_pool_cq", artifacts.candidate_pool, "planned"),
                _step_payload("build_evidence_needs_bq", artifacts.evidence_needs, "planned"),
                _step_payload("build_support_cache", artifacts.binding_cache, "planned"),
                _step_payload("compose_final_evidence_rq", artifacts.evidence_selection, "planned"),
            ],
        }

    artifacts.workdir.mkdir(parents=True, exist_ok=True)

    if _should_skip(artifacts.openie_facts, force=cfg.force):
        steps.append(_step_payload("build_openie_facts", artifacts.openie_facts, "skipped"))
    else:
        facts = build_openie_artifact(
            corpus_path=corpus_path,
            output_path=artifacts.openie_facts,
            mode=cfg.openie_mode,
            llm_base_url=cfg.llm_base_url,
            llm_model=cfg.openie_model,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            fallback=True,
        )
        steps.append(_step_payload("build_openie_facts", artifacts.openie_facts, "completed", count=len(facts)))

    if _should_skip(artifacts.evidence_link_index, force=cfg.force):
        steps.append(_step_payload("build_evidence_link_index", artifacts.evidence_link_index, "skipped"))
    else:
        index = build_evidence_link_index(
            corpus_path=corpus_path,
            openie_path=artifacts.openie_facts,
            output_path=artifacts.evidence_link_index,
        )
        steps.append(
            _step_payload(
                "build_evidence_link_index",
                artifacts.evidence_link_index,
                "completed",
                documents=len(index.documents),
                links=len(index.links),
            )
        )

    if _should_skip(artifacts.candidate_pool, force=cfg.force):
        steps.append(_step_payload("build_candidate_pool_cq", artifacts.candidate_pool, "skipped"))
    else:
        records = build_candidate_pool_cq(
            questions_path=questions_path,
            corpus_path=corpus_path,
            index_path=artifacts.evidence_link_index,
            output_path=artifacts.candidate_pool,
            dense_top_k=cfg.dense_top_k,
            max_hops=cfg.max_hops,
            pool_k=cfg.pool_k,
        )
        steps.append(_step_payload("build_candidate_pool_cq", artifacts.candidate_pool, "completed", count=len(records)))

    if _should_skip(artifacts.evidence_needs, force=cfg.force):
        steps.append(_step_payload("build_evidence_needs_bq", artifacts.evidence_needs, "skipped"))
    else:
        rows = build_evidence_needs_bq(
            questions_path=questions_path,
            output_path=artifacts.evidence_needs,
            mode=cfg.evidence_need_mode,
            llm_base_url=cfg.llm_base_url,
            llm_model=cfg.evidence_need_model,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_steps=cfg.max_steps,
        )
        steps.append(_step_payload("build_evidence_needs_bq", artifacts.evidence_needs, "completed", count=len(rows)))

    if _should_skip(artifacts.binding_cache, force=cfg.force):
        steps.append(_step_payload("build_support_cache", artifacts.binding_cache, "skipped"))
    else:
        cache = build_support_cache(
            candidate_pool_path=artifacts.candidate_pool,
            evidence_needs_path=artifacts.evidence_needs,
            output_path=artifacts.binding_cache,
            mode=cfg.binding_mode,
            binding_model=cfg.binding_model,
            max_candidates=cfg.binding_max_candidates,
            llm_base_url=cfg.llm_base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            fallback=True,
        )
        steps.append(_step_payload("build_support_cache", artifacts.binding_cache, "completed", count=len(cache)))

    if _should_skip(artifacts.evidence_selection, force=cfg.force):
        steps.append(_step_payload("compose_final_evidence_rq", artifacts.evidence_selection, "skipped"))
        selection_payload: Mapping[str, Any] = {}
    else:
        selection_payload = compose_final_evidence_rq(
            dataset=cfg.dataset,
            candidate_pool_path=artifacts.candidate_pool,
            evidence_needs_path=artifacts.evidence_needs,
            binding_cache_path=artifacts.binding_cache,
            output_path=artifacts.evidence_selection,
            reader_budget_k=cfg.reader_budget_k,
            stability_window_m=cfg.stability_window_m,
            pool_k=cfg.pool_k,
            max_queries=cfg.max_queries,
            binding_model=cfg.binding_model,
            embedding_name=cfg.embedding_name,
            embedding_base_url=cfg.embedding_base_url,
            embedding_batch_size=cfg.embedding_batch_size,
            embedding_api_key=cfg.api_key,
            embedding_timeout=cfg.timeout,
            binding_max_candidates=cfg.binding_max_candidates,
            min_coverage_gain=cfg.min_coverage_gain,
            min_swap_gain=cfg.min_swap_gain,
            allow_missing_requirements=cfg.allow_missing_requirements,
        )
        steps.append(
            _step_payload(
                "compose_final_evidence_rq",
                artifacts.evidence_selection,
                "completed",
                count=len(selection_payload.get("rows") or []),
            )
        )

    return {
        "method": PAPER_FACING_METHOD_NAME,
        "method_contract": dict(METHOD_CONTRACT),
        "config": cfg.to_mapping(),
        "artifacts": artifacts.to_mapping(),
        "selection": str(artifacts.evidence_selection),
        "selection_summary": dict(selection_payload.get("summary") or {}),
        "steps": steps,
    }
