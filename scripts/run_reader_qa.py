#!/usr/bin/env python3
"""CLI wrapper for :mod:`evidencelink.reader_qa`."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink.reader_qa import main


if __name__ == "__main__":
    raise SystemExit(main())
