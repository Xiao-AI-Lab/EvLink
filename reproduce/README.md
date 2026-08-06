# EvLink Reproduction Protocol

This directory defines the public, configuration-driven reproduction surface.
It covers offline validation, benchmark preparation, model-backed runs,
artifact schemas, and aggregate metric definitions. It does not publish or
depend on private experiment caches.

## Offline Contract Check

The offline configuration uses deterministic embeddings and local heuristic
OpenIE, evidence-need, and binding stages. It validates the complete artifact
flow but does not claim to reproduce model-backed paper scores.

```bash
python scripts/run_reproduce_config.py reproduce/configs/offline-smoke.json
```

The expected output files are listed in [ARTIFACT_SCHEMA.md](ARTIFACT_SCHEMA.md).
The run should report one selected row and write under
`runs/reproduce/offline-smoke/`.

## Benchmark Preparation

Managed downloads are available for 2WikiMultiHopQA, HotpotQA, and MuSiQue.
NQ-ReAR and PopQA require manually supplied source files because no approved
stable mirror is encoded in the manifest.

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py \
  --dataset 2wikimultihopqa,hotpotqa,musique

for dataset in 2wikimultihopqa hotpotqa musique; do
  python scripts/prepare_dataset.py \
    --dataset "$dataset" \
    --output-root "runs/datasets/$dataset" \
    --force
done
```

See [../datasets/README.md](../datasets/README.md) for file names,
manual-source requirements, and license responsibility.

## Model-Backed Protocol Template

`configs/paper-qwen32-nv.json` records a Qwen 32B-compatible LLM plus NV
embedding protocol without storing a provider, credential, or private host.
Set the runtime values in the environment:

```bash
export EVLINK_LLM_BASE_URL="https://provider.example/v1"
export EVLINK_LLM_MODEL="provider-qwen-32b-model-id"
export EVLINK_EMBEDDING_BASE_URL="https://embedding.example/v1"
export EVLINK_EMBEDDING_MODEL="provider-nv-embedding-model-id"
export EVLINK_API_KEY="..."

python scripts/run_reproduce_config.py \
  reproduce/configs/paper-qwen32-nv.json \
  --dry-run
```

Remove `--dry-run` only after the prepared inputs and endpoints are verified.
This template fixes the public protocol shape, not a provider-specific model
revision. Record exact model revisions, endpoint software, prompts, and package
commit alongside any reported result.

## Public/Private Boundary

Public releases include source, configs, deterministic fixtures, source
manifests, aggregate metric definitions, and schema documentation. They exclude
model weights, embedding stores, graph pickles, SQLite caches, per-query model
outputs, host aliases, private endpoints, API keys, and internal rebuttal
artifacts.

The same source commit can produce different model-backed results when model
revisions, endpoint implementations, or cached generations differ. An exact
paper-result claim therefore requires all items in the run-record checklist:

1. EvLink Git commit and `evidencelink` package version.
2. Prepared dataset manifest and source revision.
3. LLM and embedding model identifiers and revisions.
4. Endpoint implementation and generation parameters.
5. Pipeline config and artifact schema version.
6. Aggregate metrics computed under [METRICS.md](METRICS.md).
