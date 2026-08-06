from __future__ import annotations

from evidencelink.contract import (
    EXTERNAL_UPSTREAM_RETRIEVER_NAME,
    MAIN_UPSTREAM_RETRIEVER_NAME,
    admission_window,
    is_main_evidencelink_pool,
    pool_provenance_key,
    resolved_pool_upstream_retriever,
    validate_budgets,
)
from evidencelink.io_utils import title_all_covered, title_recall


def test_budget_helpers() -> None:
    validate_budgets(reader_budget_k=3, stability_window_m=2)
    assert admission_window(3, 2) == 1


def test_title_metrics_ignore_parenthetical_disambiguation() -> None:
    gold = ["Bank of China Tower (Hong Kong)", "One Raffles Place"]
    candidates = ["One Raffles Place", "Bank of China Tower"]
    assert title_recall(gold, candidates, k=2) == 1.0
    assert title_all_covered(gold, candidates, k=2) is True


def test_pool_provenance_prefers_explicit_main_protocol() -> None:
    payload = {
        "retrieval": {"input_method": MAIN_UPSTREAM_RETRIEVER_NAME},
        "records": [
            {"pool_trace": {"input_method": EXTERNAL_UPSTREAM_RETRIEVER_NAME}}
        ],
    }

    assert resolved_pool_upstream_retriever(payload) == MAIN_UPSTREAM_RETRIEVER_NAME
    assert pool_provenance_key(payload) == "retrieval.input_method"
    assert is_main_evidencelink_pool(payload) is True


def test_pool_provenance_infers_uniform_external_selector_records() -> None:
    payload = {
        "records": [
            {"pool_trace": {"input_method": EXTERNAL_UPSTREAM_RETRIEVER_NAME}},
            {"pool_trace": {"input_method": EXTERNAL_UPSTREAM_RETRIEVER_NAME}},
        ]
    }

    assert resolved_pool_upstream_retriever(payload) == EXTERNAL_UPSTREAM_RETRIEVER_NAME
    assert pool_provenance_key(payload) == "records[].pool_trace.input_method"
    assert is_main_evidencelink_pool(payload) is False
    assert resolved_pool_upstream_retriever({"records": [{"pool_trace": {}}]}) == ""
    assert resolved_pool_upstream_retriever(
        {
            "records": [
                {"pool_trace": {"input_method": MAIN_UPSTREAM_RETRIEVER_NAME}},
                {"pool_trace": {"input_method": EXTERNAL_UPSTREAM_RETRIEVER_NAME}},
            ]
        }
    ) == ""
