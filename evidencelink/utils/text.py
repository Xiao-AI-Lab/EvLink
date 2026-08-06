"""Text normalization helpers."""

from __future__ import annotations

import re
import string


def text_processing(text):
    """Normalize text with the same simple policy used in the original runs."""
    if isinstance(text, list):
        return [text_processing(item) for item in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def normalize_structure_text(text: str) -> str:
    normalized = text_processing(text)
    normalized = re.sub(r"\s+", " ", str(normalized)).strip()
    return normalized


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization used for lightweight reader eval."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))
