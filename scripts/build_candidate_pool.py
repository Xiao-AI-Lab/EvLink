#!/usr/bin/env python3
"""CLI wrapper for :mod:`evidencelink.build_candidate_pool`."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink.build_candidate_pool import main


if __name__ == "__main__":
    raise SystemExit(main())
