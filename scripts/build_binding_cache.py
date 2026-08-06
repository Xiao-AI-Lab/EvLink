#!/usr/bin/env python3
"""CLI wrapper for :mod:`evidencelink.build_binding_cache`."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink.build_binding_cache import main


if __name__ == "__main__":
    raise SystemExit(main())
