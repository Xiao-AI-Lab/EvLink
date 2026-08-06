#!/usr/bin/env python3
"""Run the public EvLink pipeline from a versioned JSON config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink import PaperPipelineConfig, run_paper_pipeline


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        resolved = os.environ.get(name)
        if resolved is None:
            raise ValueError(f"required environment variable is not set: {name}")
        return resolved
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def _missing_environment_variables(value: Any) -> list[str]:
    """Return referenced environment variables that are not available."""

    missing: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str) and item.startswith("$"):
            name = item[1:]
            if name and name not in os.environ and name not in missing:
                missing.append(name)
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return missing


def load_reproduce_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("reproduction config must be a JSON object")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported reproduction config schema_version")
    inputs = payload.get("inputs")
    pipeline = payload.get("pipeline")
    if not isinstance(inputs, Mapping) or not isinstance(pipeline, Mapping):
        raise ValueError("reproduction config requires inputs and pipeline objects")
    missing = _missing_environment_variables(payload)
    if missing:
        raise ValueError(
            "required environment variables are not set: " + ", ".join(missing)
        )
    return dict(_expand_environment(payload))


def _project_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = load_reproduce_config(args.config)
    inputs = dict(payload["inputs"])
    config_values = dict(payload["pipeline"])
    unknown = set(config_values).difference(PaperPipelineConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown pipeline config fields: {', '.join(sorted(unknown))}")
    if args.dry_run:
        config_values["dry_run"] = True
    config = PaperPipelineConfig(**config_values)
    workdir = args.workdir or _project_path(payload.get("workdir", "runs/reproduce"))
    result = run_paper_pipeline(
        corpus_path=_project_path(inputs["corpus"]),
        questions_path=_project_path(inputs["questions"]),
        workdir=workdir,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
