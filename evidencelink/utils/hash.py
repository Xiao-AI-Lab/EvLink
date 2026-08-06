"""Hash helpers."""

from __future__ import annotations

from hashlib import md5


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Return a stable MD5 identifier with an optional prefix."""
    return prefix + md5(str(content).encode("utf-8")).hexdigest()
