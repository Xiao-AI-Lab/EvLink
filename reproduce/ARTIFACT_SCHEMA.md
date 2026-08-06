# Reproduction Artifact Schema

One end-to-end run writes these artifacts in dependency order:

| File | Format | Role |
| --- | --- | --- |
| `openie_facts.jsonl` | JSONL | Source-grounded fact witnesses extracted from the corpus. |
| `evidence_link_index.json` | JSON | Passage nodes, evidence links, witnesses, and index summary. |
| `candidate_pool.jsonl` | JSONL | Query-local candidate pools `C_q`. |
| `evidence_needs.jsonl` | JSONL | Evidence-need sets `B(q)`. |
| `binding_cache.json` | JSON | Support strings keyed by need/candidate/model/prompt identity. |
| `evidence_selection.json` | JSON | Final fixed-budget evidence sets `R_q`, traces, and aggregates. |
| `reader_qa.json` | JSON | Optional reader predictions and EM/F1 aggregates. |

## Required Identity Fields

- Every input question has a stable string `query_id`.
- Every passage and candidate has a stable string `doc_id`.
- Candidate order is represented by `rank` and by its position in the pool.
- External candidate pools record `pool_trace.input_method` as
  `external_retriever`; their `source` and `metadata` remain upstream
  provenance.
- EvLink-built graph candidates may contain `path` and `edge_evidence`;
  adapters do not synthesize these fields from third-party graph metadata.

## Selection Output

The `evidence_selection.json` root contains `rows`, `summary`,
`evidence_selection_query_traces`, `evidence_selection_config`, and a method
contract. Each row contains the question identity, gold fields when available,
top-K evidence documents/titles/IDs, retrieval metrics, and the selection trace.

The schema is inspectable JSON rather than a pickle. Consumers should tolerate
additional fields but must not reinterpret `candidate_pool` as the final reader
evidence set. Full field descriptions are in [../ARTIFACTS.md](../ARTIFACTS.md).
