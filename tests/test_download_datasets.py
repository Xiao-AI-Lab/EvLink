from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evidencelink.download_datasets import (
    DEFAULT_MANIFEST,
    dataset_names,
    download_datasets,
    load_manifest,
)


def test_packaged_dataset_manifest_lists_all_paper_datasets() -> None:
    manifest = load_manifest()

    assert set(manifest["datasets"]) == {
        "2wikimultihopqa",
        "hotpotqa",
        "musique",
        "nq_rear",
        "popqa",
    }
    assert DEFAULT_MANIFEST.exists()
    assert dataset_names("2wikimultihopqa,musique", manifest) == ["2wikimultihopqa", "musique"]


def test_downloader_verifies_local_file_url_and_reports_manual_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('[{"question": "Q?"}]\n', encoding="utf-8")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "automatic": {
                        "files": [
                            {
                                "filename": "automatic.json",
                                "url": source.as_uri(),
                                "sha256": checksum,
                            }
                        ]
                    },
                    "manual": {"files": [], "notes": "Supply this dataset manually."},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = download_datasets(
        datasets=["automatic", "manual"],
        output_root=tmp_path / "downloaded",
        manifest_path=manifest,
    )

    assert payload["downloaded_or_verified"][0]["status"] == "downloaded"
    assert payload["manual_sources"][0]["dataset"] == "manual"
    assert (tmp_path / "downloaded" / "automatic.json").read_bytes() == source.read_bytes()


def test_downloader_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="unknown datasets"):
        dataset_names("unknown", load_manifest())
