"""Adapt a LightRAG QueryResult without importing LightRAG."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink import EvidenceSelector, EvidenceSelectorConfig
from evidencelink.integrations import candidates_from_lightrag


# In an installed LightRAG application, replace this fixture with a QueryResult
# returned by a query call configured to include structured raw_data.
query_result = SimpleNamespace(
    raw_data={
        "data": {
            "chunks": [
                {
                    "reference_id": "1",
                    "content": "Acme Corporation was founded by Alice Chen.",
                    "file_path": "/knowledge/acme.md",
                    "chunk_id": "chunk-acme",
                },
                {
                    "reference_id": "2",
                    "content": "Alice Chen was born in Singapore.",
                    "file_path": "/knowledge/alice.md",
                    "chunk_id": "chunk-alice",
                },
            ],
            "references": [
                {"reference_id": "1", "file_path": "/knowledge/acme.md"},
                {"reference_id": "2", "file_path": "/knowledge/alice.md"},
            ],
        }
    }
)

candidates = candidates_from_lightrag(query_result)
result = EvidenceSelector(
    EvidenceSelectorConfig(reader_budget_k=2, evidence_need_mode="anchor_list")
).select(
    question="Who founded Acme Corporation and where was the founder born?",
    candidates=candidates,
)
print(json.dumps(result.to_mapping(), indent=2, ensure_ascii=False))
