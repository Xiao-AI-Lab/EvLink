# Benchmark Data Preparation

Benchmark source files are not committed to EvidenceLink. `datasets/raw/` is a
local download/placement directory ignored by Git. The repository only carries
a deterministic toy fixture under `datasets/fixtures/` and a checksum-pinned
source manifest in `evidencelink/data/dataset_sources.json`.

## Automatic Sources

The automatic entries are frozen to a HippoRAG source revision. The downloader
checks file integrity automatically; users do not need to manage checksum
values.

List metadata or download the automatic sources with:

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py \
  --dataset 2wikimultihopqa,hotpotqa,musique
```

Existing files are accepted only when their SHA-256 matches. Use `--force` to
replace a mismatched local file.

## Manual Sources

NQ-ReAR and PopQA are marked `manual_source_required` because no approved,
stable, checksum-pinned mirror is encoded in the manifest. Supply the expected
HippoRAG-style 1K query/corpus files manually:

```text
datasets/raw/nq_rear.json
datasets/raw/nq_rear_corpus.json
datasets/raw/popqa.json
datasets/raw/popqa_corpus.json
```

## Conversion

Use `scripts/prepare_dataset.py` or `evidencelink-prepare-dataset` after the
source files are present. For one dataset:

```bash
python scripts/prepare_dataset.py \
  --dataset musique \
  --output-root runs/datasets/musique \
  --force
```

The generated `runs/datasets/*/corpus.jsonl` and
`runs/datasets/*/questions.jsonl` files are build artifacts and are not tracked.
Each output directory also contains `dataset_manifest.json` with source and row
counts.

## License Responsibility

The EvidenceLink MIT license covers EvidenceLink source code, not third-party
datasets. Users are responsible for reviewing and complying with each upstream
dataset's terms. The manifest links to the upstream projects and records known
license information without granting redistribution rights.
