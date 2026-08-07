#!/usr/bin/env python3
"""Smoke-test an installed EvLink wheel without source-tree imports."""

from __future__ import annotations

import json

from evidencelink import (
    EvidenceSelector,
    EvidenceSelectorConfig,
    load_query_result_view_schema,
)
from evidencelink.integrations import candidates_from_hipporag


candidates = candidates_from_hipporag(
    {
        "docs": [
            "Acme Corporation\nAcme Corporation was founded by Alice Chen.",
            "Alice Chen\nAlice Chen was born in Singapore.",
        ],
        "scores": [0.98, 0.94],
    }
)
result = EvidenceSelector(
    EvidenceSelectorConfig(reader_budget_k=2, evidence_need_mode="anchor_list")
).select(
    question="Who founded Acme Corporation and where was the founder born?",
    candidates=candidates,
)
assert len(result.evidence) == 2
assert result.trace["input_method"] == "external_retriever"
assert load_query_result_view_schema()["title"] == "EvLink QueryResultView/v1"
print(json.dumps({"version": __import__("evidencelink").__version__, "evidence": 2}))
