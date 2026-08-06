"""OpenAI-compatible embedding client used by standalone EvLink."""

from __future__ import annotations

import json
import hashlib
import urllib.request
from typing import Iterable, Sequence

import numpy as np


class OpenAIEmbeddingClient:
    """Small OpenAI-compatible embeddings client with an in-memory cache."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        batch_size: int = 16,
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.batch_size = int(batch_size)
        self.api_key = str(api_key or "")
        self.timeout = float(timeout)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        unique = []
        seen = set()
        for text in texts:
            clean = str(text or "")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            unique.append(clean)

        missing = [text for text in unique if text not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
            request = urllib.request.Request(
                self.base_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            data = sorted(body.get("data", []), key=lambda row: int(row.get("index", 0)))
            if len(data) != len(batch):
                raise RuntimeError(f"Embedding endpoint returned {len(data)} vectors for batch size {len(batch)}")
            for text, row in zip(batch, data):
                self._cache[text] = np.asarray(row["embedding"], dtype=float)
        return {text: self._cache[text] for text in unique if text in self._cache}


class DeterministicHashEmbeddingClient:
    """Tiny deterministic embedding client for tests and offline smoke checks."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        output = {}
        for text in texts:
            vec = np.zeros(self.dim, dtype=float)
            for token in str(text or "").lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                vec[int(digest[:8], 16) % self.dim] += 1.0
            output[str(text or "")] = vec
        return output
