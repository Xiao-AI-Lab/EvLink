"""Coverage-aware evidence selection over an exported EvLink candidate pool."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evidencelink.contract import (
    DEFAULT_ADMISSION_OBJECTIVE,
    DEFAULT_ORDERING_POLICY,
    DEFAULT_RETENTION_PROXY,
    METHOD_NAME,
    PAPER_FACING_METHOD_NAME,
    admission_window,
    validate_budgets,
)
from evidencelink.types import (
    EvidenceQueryState,
    EvidenceSelectionResult,
    ordered_unique_positions,
)


def finalized_positions(selected_positions: Sequence[int], *, pool_size: int, reader_budget_k: int) -> list[int]:
    front = ordered_unique_positions(selected_positions, pool_size=pool_size, limit=reader_budget_k)
    seen = set(front)
    for pos in range(int(pool_size)):
        if len(front) >= int(reader_budget_k):
            break
        if pos not in seen:
            seen.add(pos)
            front.append(pos)
    return front[: int(reader_budget_k)]


def admission_steps(selection_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    safe_trace = selection_trace.get("safe_projection_trace")
    if isinstance(safe_trace, Mapping) and isinstance(safe_trace.get("admission_steps"), list):
        return [dict(row) for row in safe_trace.get("admission_steps") or [] if isinstance(row, Mapping)]
    if isinstance(safe_trace, Mapping) and isinstance(safe_trace.get("safe_swap_steps"), list):
        return [dict(row) for row in safe_trace.get("safe_swap_steps") or [] if isinstance(row, Mapping)]
    steps = selection_trace.get("selection_steps")
    return [dict(row) for row in steps or [] if isinstance(row, Mapping)]


def compose_evidence_selection(
    state: EvidenceQueryState,
    *,
    requirements: Sequence[Any],
    utility_provider: Any,
) -> EvidenceSelectionResult:
    validate_budgets(reader_budget_k=state.reader_budget_k, stability_window_m=state.stability_window_m)
    baseline_positions = list(range(min(int(state.reader_budget_k), len(state.pool_docs))))
    protected_baseline_positions = list(
        range(min(int(state.stability_window_m), len(baseline_positions)))
    )
    selected_positions, selection_trace = utility_provider.select_positions(state, requirements)
    final_positions = finalized_positions(
        selected_positions,
        pool_size=len(state.pool_docs),
        reader_budget_k=state.reader_budget_k,
    )
    admitted_positions = [pos for pos in final_positions if pos not in baseline_positions]
    steps = admission_steps(selection_trace)
    first_step = steps[0] if steps else {}
    safe_trace = selection_trace.get("safe_projection_trace") if isinstance(selection_trace, Mapping) else {}
    if not isinstance(safe_trace, Mapping):
        safe_trace = {}
    decision = "admit" if admitted_positions or final_positions != baseline_positions else "keep_baseline"
    rank_stability_held = (
        final_positions[: len(protected_baseline_positions)] == protected_baseline_positions
    )
    final_titles = [state.pool_titles[pos] for pos in final_positions if pos < len(state.pool_titles)]
    final_docs = [state.pool_docs[pos] for pos in final_positions if pos < len(state.pool_docs)]
    trace = {
        "method": METHOD_NAME,
        "paper_facing_method": PAPER_FACING_METHOD_NAME,
        "selection": "coverage_aware_fixed_budget_selection",
        "objective": DEFAULT_ADMISSION_OBJECTIVE,
        "retention_proxy": DEFAULT_RETENTION_PROXY,
        "reader_budget_k": int(state.reader_budget_k),
        "stability_window_m": int(state.stability_window_m),
        "admission_window": int(admission_window(state.reader_budget_k, state.stability_window_m)),
        "baseline_positions": list(baseline_positions),
        "baseline_titles": [state.pool_titles[pos] for pos in baseline_positions if pos < len(state.pool_titles)],
        "stable_seed_positions": list(protected_baseline_positions),
        "stable_seed_titles": [
            state.pool_titles[pos]
            for pos in protected_baseline_positions
            if pos < len(state.pool_titles)
        ],
        "protected_baseline_positions": list(protected_baseline_positions),
        "protected_baseline_titles": [
            state.pool_titles[pos]
            for pos in protected_baseline_positions
            if pos < len(state.pool_titles)
        ],
        "baseline_objective": selection_trace.get("baseline_objective", safe_trace.get("baseline_objective")),
        "best_candidate_position": first_step.get("in_position") if first_step else None,
        "best_candidate_title": first_step.get("in_title") if first_step else None,
        "best_candidate_objective": first_step.get("objective") if first_step else None,
        "best_candidate_gain": first_step.get("objective_gain") if first_step else None,
        "decision": decision,
        "rank_stability_held": bool(rank_stability_held),
        "final_positions": list(final_positions),
        "final_titles": list(final_titles),
        "admitted_positions": list(admitted_positions),
        "ordering_policy": DEFAULT_ORDERING_POLICY,
        "evidence_need_binding": "frozen_cache_binding_candidates",
        "selected_binding_id": selection_trace.get("selected_binding_id"),
        "safe_decision": safe_trace.get("safe_decision"),
        "admission_steps": list(steps),
    }
    for key in (
        "input_method",
        "input_candidate_count",
        "used_candidate_count",
        "truncated",
    ):
        if key in state.pool_trace:
            trace[key] = state.pool_trace[key]
    return EvidenceSelectionResult(
        final_positions=tuple(final_positions),
        final_docs=tuple(final_docs),
        final_titles=tuple(final_titles),
        baseline_positions=tuple(baseline_positions),
        stable_seed_positions=tuple(protected_baseline_positions),
        admitted_positions=tuple(admitted_positions),
        trace=trace,
        raw_selection_trace=dict(selection_trace),
    )
