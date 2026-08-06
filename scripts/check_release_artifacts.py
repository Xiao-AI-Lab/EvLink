#!/usr/bin/env python3
"""Validate that built distributions contain only the public release surface."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from zipfile import ZipFile


FORBIDDEN_PARTS = {
    ".private",
    "adapters",
    "evidenceflow",
    "evidence_transition_graphrag",
    "evidence_transition_graphragv4_composition",
    "evidence_transition_graphragv4_fact_witnessed_sto",
    "rebuttal_repro",
    "run_logs",
    "runs",
    "tests",
}


def members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"unsupported distribution: {path}")


def validate(path: Path) -> None:
    names = members(path)
    forbidden_suffixes = (
        ".sqlite",
        ".db",
        ".pkl",
        ".pickle",
        ".parquet",
        ".pt",
        ".pth",
        ".safetensors",
        ".bin",
        ".gguf",
        ".npz",
        ".zip",
        ".tar",
        ".zst",
    )
    violations = [
        name
        for name in names
        if FORBIDDEN_PARTS.intersection(Path(name).parts)
        or name.endswith(forbidden_suffixes)
        or ("datasets" in Path(name).parts and "raw" in Path(name).parts and name.endswith(".json"))
    ]
    if violations:
        preview = "\n".join(f"  - {name}" for name in violations[:20])
        raise SystemExit(f"forbidden release members in {path}:\n{preview}")
    if path.suffix == ".whl":
        roots = {name.split("/", 1)[0] for name in names}
        unexpected = {root for root in roots if root != "evidencelink" and not root.endswith(".dist-info")}
        if unexpected:
            raise SystemExit(f"unexpected wheel roots in {path}: {sorted(unexpected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    distributions = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
    if not distributions:
        raise SystemExit(f"no distributions found under {args.dist}")
    for distribution in distributions:
        validate(distribution)
        print(f"OK {distribution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
