"""Select a fixed-budget evidence set from an external retriever pool."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink import EvidenceSelector, EvidenceSelectorConfig


candidates = [
    {
        "doc_id": "d0",
        "title": "Acme Corporation",
        "text": "Acme Corporation was founded by Alice Chen in 2012.",
        "score": 0.98,
        "source": "example_dense_retriever",
    },
    {
        "doc_id": "d1",
        "title": "Alice Chen",
        "text": "Alice Chen was born in Singapore.",
        "score": 0.94,
        "source": "example_dense_retriever",
    },
    {
        "doc_id": "d2",
        "title": "Company history",
        "text": "Acme opened its first European office in 2018.",
        "score": 0.72,
        "source": "example_dense_retriever",
    },
]

selector = EvidenceSelector(
    EvidenceSelectorConfig(
        reader_budget_k=2,
        evidence_need_mode="anchor_list",
    )
)
result = selector.select(
    question="Who founded Acme Corporation and where was the founder born?",
    candidates=candidates,
)

print(json.dumps(result.to_mapping(), indent=2, ensure_ascii=False))
