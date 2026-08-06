from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_reproduce_config import load_reproduce_config


def test_offline_reproduction_config_runs_end_to_end(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_reproduce_config.py",
            "reproduce/configs/offline-smoke.json",
            "--workdir",
            str(tmp_path / "run"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["selection_summary"]["count"] == 1
    assert Path(payload["selection"]).exists()
    assert payload["config"]["embedding_name"] == "deterministic-hash"


def test_model_config_requires_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EVLINK_LLM_BASE_URL",
        "EVLINK_LLM_MODEL",
        "EVLINK_EMBEDDING_BASE_URL",
        "EVLINK_EMBEDDING_MODEL",
        "EVLINK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="EVLINK_LLM_BASE_URL"):
        load_reproduce_config("reproduce/configs/paper-qwen32-nv.json")


def test_model_config_expands_environment_without_persisting_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "EVLINK_LLM_BASE_URL": "https://llm.example/v1",
        "EVLINK_LLM_MODEL": "qwen3-32b",
        "EVLINK_EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "EVLINK_EMBEDDING_MODEL": "nv-embed",
        "EVLINK_API_KEY": "test-only-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    payload = load_reproduce_config("reproduce/configs/paper-qwen32-nv.json")

    assert payload["pipeline"]["llm_base_url"] == values["EVLINK_LLM_BASE_URL"]
    assert payload["pipeline"]["embedding_name"] == values["EVLINK_EMBEDDING_MODEL"]
    assert payload["pipeline"]["api_key"] == values["EVLINK_API_KEY"]


def test_pipeline_results_redact_api_keys(tmp_path: Path) -> None:
    from evidencelink import PaperPipelineConfig, run_paper_pipeline

    result = run_paper_pipeline(
        corpus_path="examples/corpus.jsonl",
        questions_path="examples/questions.jsonl",
        workdir=tmp_path / "dry-run",
        config=PaperPipelineConfig(api_key="secret-value", dry_run=True),
    )

    assert result["config"]["api_key"] == "<redacted>"
    assert result["config"]["api_key_configured"] is True
    assert "secret-value" not in json.dumps(result)
