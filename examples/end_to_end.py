"""Run the complete deterministic EvidenceLink pipeline from source inputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidencelink import PaperPipelineConfig, run_paper_pipeline


with TemporaryDirectory(prefix="evidencelink-e2e-") as temporary:
    result = run_paper_pipeline(
        corpus_path=ROOT / "examples" / "corpus.jsonl",
        questions_path=ROOT / "examples" / "questions.jsonl",
        workdir=Path(temporary),
        config=PaperPipelineConfig(dataset="demo", force=True),
    )
    payload = {
        "selection_summary": result["selection_summary"],
        "artifact_names": sorted(
            Path(path).name
            for path in result.get("artifacts", {}).values()
            if isinstance(path, str)
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
