"""Coverage utility provider for EvidenceLink evidence selection."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evidencelink.evidence_need_utils import EvidenceNeed, select_coverage_aware_positions  # noqa: E402
from evidencelink.utils.hash import compute_mdhash_id  # noqa: E402

from evidencelink.contract import (  # noqa: E402
    DEFAULT_BINDING_MAX_CANDIDATES,
    DEFAULT_MIN_COVERAGE_GAIN,
    DEFAULT_MIN_SWAP_GAIN,
    admission_window,
)
from evidencelink.types import EvidenceQueryState  # noqa: E402


def load_binding_cache(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): [str(item) for item in value]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def make_cache_only_binding_extractor(
    *,
    cache: Mapping[str, Sequence[str]],
    model: str,
    stats: dict[str, int],
) -> Callable[[str, str], list[str]]:
    think_re = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def extract(subquery: str, doc_text: str) -> list[str]:
        prompt_payload = {
            # Keep this cache key stable so existing released binding caches remain usable.
            "version": "coverage_llm_binding_v1",
            "model": str(model),
            "subquery": str(subquery),
            "passage": str(doc_text[:1500]),
        }
        cache_key = compute_mdhash_id(json.dumps(prompt_payload, sort_keys=True, ensure_ascii=False))
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        if cache_key not in cache:
            stats["misses"] = int(stats.get("misses", 0)) + 1
            return []
        stats["hits"] = int(stats.get("hits", 0)) + 1
        return [think_re.sub("", str(item)).strip() for item in cache[cache_key] if str(item).strip()]

    return extract


def build_text_embeddings(embedding_client: Any, texts: Sequence[str], keys: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    items: list[tuple[str, str]] = []
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for idx, text in enumerate(texts):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        key = str(keys[idx]) if keys is not None and idx < len(keys) else clean_text
        items.append((key, clean_text))
        if clean_text not in seen_texts:
            seen_texts.add(clean_text)
            unique_texts.append(clean_text)
    passage_query_embeddings = embedding_client.embed(unique_texts) if unique_texts else {}
    embeddings: dict[str, np.ndarray] = {}
    for key, text in items:
        vector = passage_query_embeddings.get(text)
        if vector is not None:
            embeddings[str(key)] = np.asarray(vector, dtype=float)
    return embeddings


def build_requirement_embeddings(embedding_client: Any, requirements: Sequence[EvidenceNeed]) -> dict[str, np.ndarray]:
    filtered = [req for req in requirements if str(req.subquery or "").strip()]
    return build_text_embeddings(
        embedding_client,
        [req.subquery for req in filtered],
        keys=[req.unit_id for req in filtered],
    )


class CoverageUtilityProvider:
    """Adapter over evidence-need binding and noisy-OR coverage selection."""

    def __init__(
        self,
        *,
        embedding_client: Any,
        binding_cache: Mapping[str, Sequence[str]],
        binding_model: str,
        binding_top_m: int = DEFAULT_BINDING_MAX_CANDIDATES,
        min_objective_gain: float = DEFAULT_MIN_COVERAGE_GAIN,
        min_swap_gain: float = DEFAULT_MIN_SWAP_GAIN,
        llm_binding_title_match_mode: str = "wiki_title",
    ) -> None:
        self.embedding_client = embedding_client
        self.binding_cache = dict(binding_cache)
        self.binding_model = str(binding_model)
        self.binding_top_m = int(binding_top_m)
        self.min_objective_gain = float(min_objective_gain)
        self.min_swap_gain = float(min_swap_gain)
        self.llm_binding_title_match_mode = str(llm_binding_title_match_mode)
        self.binding_stats = {"attempts": 0, "hits": 0, "misses": 0}
        self._llm_extract_fn = make_cache_only_binding_extractor(
            cache=self.binding_cache,
            model=self.binding_model,
            stats=self.binding_stats,
        )

    def select_positions(
        self,
        state: EvidenceQueryState,
        requirements: Sequence[EvidenceNeed],
    ) -> tuple[list[int], dict[str, Any]]:
        requirement_embeddings = build_requirement_embeddings(self.embedding_client, requirements)
        pool_doc_embeddings = build_text_embeddings(self.embedding_client, state.pool_docs)
        passage_embeddings = []
        local_pool_doc_ids = []
        for pos, doc_text in enumerate(state.pool_docs):
            vector = pool_doc_embeddings.get(str(doc_text or ""))
            if vector is None:
                local_pool_doc_ids.append(None)
                passage_embeddings.append(np.array([], dtype=float))
                continue
            local_pool_doc_ids.append(pos)
            passage_embeddings.append(np.asarray(vector, dtype=float))

        def embed_bound_texts(texts: Sequence[str]) -> dict[str, np.ndarray]:
            return build_text_embeddings(self.embedding_client, texts)

        selected_positions, selection_trace = select_coverage_aware_positions(
            query=state.question,
            requirements=list(requirements),
            requirement_embeddings=requirement_embeddings,
            pool_docs=list(state.pool_docs),
            pool_doc_ids=list(local_pool_doc_ids),
            pool_doc_scores=np.asarray(list(state.pool_doc_scores), dtype=float),
            pool_doc_titles=list(state.pool_titles),
            doc_idx_to_entities={},
            passage_embeddings=np.asarray(passage_embeddings, dtype=float),
            qa_top_k=int(state.reader_budget_k),
            binding_top_m=int(self.binding_top_m),
            embed_texts_fn=embed_bound_texts,
            safe_projection=True,
            safe_min_objective_gain=float(self.min_objective_gain),
            safe_min_swap_gain=float(self.min_swap_gain),
            safe_max_swaps=admission_window(state.reader_budget_k, state.stability_window_m),
            safe_stability_window_m=int(state.stability_window_m),
            safe_projection_mode="rank_cutoff",
            safe_retriever_margin_threshold=1.01,
            safe_retriever_rank_penalty=0.0,
            llm_extract_fn=self._llm_extract_fn,
            binding_mode="llm",
            llm_binding_title_match_mode=self.llm_binding_title_match_mode,
            evidence_link_metadata=dict(state.pool_trace or {}),
            selection_label_override="coverage_aware_evidence_selection",
        )
        return [int(pos) for pos in selected_positions], dict(selection_trace)

    def summary(self) -> dict[str, Any]:
        return {
            "provider": "coverage_aware_evidence_selection",
            "binding_model": self.binding_model,
            "binding_cache_entry_count": int(len(self.binding_cache)),
            "binding_top_m": int(self.binding_top_m),
            "min_objective_gain": float(self.min_objective_gain),
            "min_swap_gain": float(self.min_swap_gain),
            "llm_binding_title_match_mode": self.llm_binding_title_match_mode,
            **dict(self.binding_stats),
        }
