"""Public method contract for EvidenceLink.

The reported EvidenceLink protocol is:

    source-grounded evidence-link pool -> coverage-aware evidence selection
    -> top-5 reader input
"""

from __future__ import annotations

from typing import Mapping


METHOD_FAMILY = "source_grounded_evidence_link_retrieval"
METHOD_VERSION = "v4_composition"
METHOD_NAME = "evidencelink"
PAPER_FACING_METHOD_NAME = "EvidenceLink"
SELECTION_COMPONENT_NAME = "Coverage-Aware Evidence Selection"
SELECTION_NAME = "coverage_aware_evidence_selection"
COVERAGE_SELECTION_NAME = "noisy_or_marginal_coverage"
MAIN_PROTOCOL_NAME = "evidencelink_pool_to_coverage_aware_selection"
MAIN_ENTRYPOINT = "evidencelink/run_evidence_selection.py"
MAIN_UPSTREAM_RETRIEVER_NAME = "source_grounded_evidence_link_pool"
MAIN_POOL_PROVENANCE_KEY = "retrieval.input_method"

DEFAULT_READER_BUDGET_K = 5
DEFAULT_STABILITY_WINDOW_M = DEFAULT_READER_BUDGET_K - 1
DEFAULT_POOL_K = 100
DEFAULT_MIN_COVERAGE_GAIN = 0.0
DEFAULT_MIN_SWAP_GAIN = 0.000001
DEFAULT_BINDING_MAX_CANDIDATES = 5
DEFAULT_ORDERING_POLICY = "fixed_budget_coverage_selection_with_rank_stability"
DEFAULT_RETENTION_PROXY = "traversal_rank_stability"
DEFAULT_ADMISSION_OBJECTIVE = "noisy_or_marginal_coverage"


METHOD_CONTRACT: Mapping[str, object] = {
    "method_family": METHOD_FAMILY,
    "method_version": METHOD_VERSION,
    "method_name": METHOD_NAME,
    "paper_facing_method_name": PAPER_FACING_METHOD_NAME,
    "main_protocol": MAIN_PROTOCOL_NAME,
    "main_entrypoint": MAIN_ENTRYPOINT,
    "main_upstream_retriever": MAIN_UPSTREAM_RETRIEVER_NAME,
    "main_pool_provenance_key": MAIN_POOL_PROVENANCE_KEY,
    "main_pool_contract": (
        "Table-1 EvidenceLink requires an external pool whose "
        "retrieval.input_method is source_grounded_evidence_link_pool."
    ),
    "uses_EvidenceLink_fact_witnessed_sto_pool_for_main_table": True,
    "uses_historical_experiment_adapter": False,
    "selection_component": SELECTION_COMPONENT_NAME,
    "selection_object": SELECTION_NAME,
    "method_object": "coverage_aware_fixed_budget_evidence_selection",
    "optimization_form": "fixed_budget_noisy_or_coverage_selection",
    "evidence_need_binding": "frozen_cache_binding_candidates",
    "retention_proxy": DEFAULT_RETENTION_PROXY,
    "admission_objective": DEFAULT_ADMISSION_OBJECTIVE,
    "coverage_selection": COVERAGE_SELECTION_NAME,
    "default_reader_budget_k": DEFAULT_READER_BUDGET_K,
    "default_stability_window_m": "reader_budget_k - 1",
    "default_admission_window": "derived",
    "default_pool_k": DEFAULT_POOL_K,
    "default_safe_min_objective_gain": DEFAULT_MIN_COVERAGE_GAIN,
    "default_safe_min_swap_gain": DEFAULT_MIN_SWAP_GAIN,
    "ordering_policy": DEFAULT_ORDERING_POLICY,
    "qa_context": "fixed_top5_passages",
    "train_free": True,
    "changes_candidate_generation": False,
    "changes_reader_budget": False,
    "changes_selection_objective": True,
    "uses_traversal_rank_stability": True,
    "uses_evidence_need_coverage": True,
    "uses_rank_stability_guard": True,
    "uses_lambda_regularization": False,
    "uses_weighted_score_fusion": False,
    "uses_dataset_routing": False,
    "uses_query_adaptive_gate": False,
    "uses_cross_signal_agreement_retention": False,
    "uses_learned_reranker": False,
    "uses_prompt_objective_patch": False,
}


def pool_upstream_retriever(pool_payload: Mapping[str, object]) -> str:
    """Return the upstream retriever recorded by an exported pool payload."""
    retrieval = pool_payload.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return ""
    return str(retrieval.get("input_method") or "")


def is_main_evidencelink_pool(pool_payload: Mapping[str, object]) -> bool:
    """Return whether a pool is the main EvidenceLink input expected by EvidenceLink."""
    return pool_upstream_retriever(pool_payload) == MAIN_UPSTREAM_RETRIEVER_NAME


def admission_window(reader_budget_k: int, stability_window_m: int) -> int:
    """Return the candidate admission window used by the stability guard."""
    validate_budgets(reader_budget_k=reader_budget_k, stability_window_m=stability_window_m)
    return int(reader_budget_k) - int(stability_window_m)


def validate_budgets(*, reader_budget_k: int, stability_window_m: int) -> None:
    """Validate reader cardinality and rank-stability budgets."""
    if int(reader_budget_k) <= 0:
        raise ValueError(f"reader_budget_k must be positive, got {reader_budget_k}")
    if int(stability_window_m) < 0:
        raise ValueError(f"stability_window_m must be non-negative, got {stability_window_m}")
    if int(stability_window_m) > int(reader_budget_k):
        raise ValueError(
            f"stability_window_m cannot exceed reader_budget_k: {stability_window_m} > {reader_budget_k}"
        )
