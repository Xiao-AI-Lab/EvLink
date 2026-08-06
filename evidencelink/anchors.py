"""Question anchor extraction for query-local EvidenceLink induction."""

from __future__ import annotations

import re
from typing import Any


_QUESTION_FILTER_WORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "the",
    "this",
    "that",
}


def extract_question_anchors(question: Any, *, max_anchors: int = 8) -> list[str]:
    text = str(question or "")
    candidates: list[str] = []
    candidates.extend(match.group(1).strip() for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text) if match.group(1))
    candidates.extend(match.group(0).strip() for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5}\b", text))
    candidates.extend(match.group(0).strip() for match in re.finditer(r"\b[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+){1,3}\b", text))

    anchors: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = re.sub(r"\s+", " ", str(candidate or "")).strip(" ?.,;:")
        key = clean.lower()
        if len(clean) < 2 or key in seen or key in _QUESTION_FILTER_WORDS:
            continue
        if all(token.lower() in _QUESTION_FILTER_WORDS for token in clean.split()):
            continue
        seen.add(key)
        anchors.append(clean)
        if len(anchors) >= int(max_anchors):
            break
    return anchors
