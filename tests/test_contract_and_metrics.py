from __future__ import annotations

from evidencelink.contract import admission_window, validate_budgets
from evidencelink.io_utils import title_all_covered, title_recall


def test_budget_helpers() -> None:
    validate_budgets(reader_budget_k=3, stability_window_m=2)
    assert admission_window(3, 2) == 1


def test_title_metrics_ignore_parenthetical_disambiguation() -> None:
    gold = ["Bank of China Tower (Hong Kong)", "One Raffles Place"]
    candidates = ["One Raffles Place", "Bank of China Tower"]
    assert title_recall(gold, candidates, k=2) == 1.0
    assert title_all_covered(gold, candidates, k=2) is True
