"""Adapt a HippoRAG retrieval result without importing HippoRAG."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink import EvidenceSelector, EvidenceSelectorConfig
from evidencelink.integrations import candidates_from_hipporag


# In an installed HippoRAG application, replace this fixture with:
# retrieval_result = hipporag.retrieve([question], num_to_retrieve=20)[0]
retrieval_result = SimpleNamespace(
    docs=[
        "Acme Corporation\nAcme Corporation was founded by Alice Chen.",
        "Alice Chen\nAlice Chen was born in Singapore.",
        "Company history\nAcme opened a European office in 2018.",
    ],
    scores=[0.98, 0.94, 0.72],
    doc_metadata=[{"doc_id": "h0"}, {"doc_id": "h1"}, {"doc_id": "h2"}],
    graph_seeds=[("Acme Corporation", "founded_by", "Alice Chen")],
)

candidates = candidates_from_hipporag(retrieval_result)
result = EvidenceSelector(
    EvidenceSelectorConfig(reader_budget_k=2, evidence_need_mode="anchor_list")
).select(
    question="Who founded Acme Corporation and where was the founder born?",
    candidates=candidates,
)
print(json.dumps(result.to_mapping(), indent=2, ensure_ascii=False))
