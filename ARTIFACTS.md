# EvLink Artifacts

EvLink stages communicate through explicit JSON or JSONL artifacts. This
keeps evidence-link construction, query-local induction, and coverage-aware
selection inspectable as separate steps.

## Input Artifacts

### `corpus.jsonl`

One passage per line.

Required fields:

- `doc_id`: stable passage identifier;
- `title`: passage title;
- `text`: passage body.

Optional fields:

- `metadata`: dataset-specific metadata.

### `questions.jsonl`

One question per line.

Required fields:

- `query_id`: stable question identifier;
- `question`: question text.

Optional fields:

- `gold_doc_ids`;
- `gold_titles`;
- `gold_answers`;
- `metadata`.

## Method Artifacts

### `openie_facts.jsonl`

Source-grounded OpenIE facts. Facts are grounding material for evidence links;
they are not retrieval states.

Main fields:

- `doc_id`;
- `fact_id`;
- `subject`;
- `relation`;
- `object`;
- `source_span`;
- `confidence`;
- `raw_text`.

### `evidence_link_index.json`

The source-grounded evidence-link index. Passages are nodes. Links connect
passages through relation-grounded or endpoint-aligned witnesses.

Main fields:

- `documents`;
- `links`;
- `summary`.

Each link includes:

- `source_doc_id`;
- `target_doc_id`;
- `link_type`;
- `relation`;
- `endpoint`;
- `witnesses`.

### `candidate_pool.jsonl`

One query-local candidate pool `C_q` per line.

Main fields:

- `query_id`;
- `question`;
- `anchors`;
- `seed_doc_ids`;
- `local_subgraph`;
- `candidate_pool`;
- `pool_docs`;
- `pool_titles`;
- `pool_doc_ids`;
- `pool_doc_scores`.

Each candidate includes:

- `rank`;
- `doc_id`;
- `title`;
- `text`;
- `source`;
- `score`;
- `path`;
- `edge_evidence`.
- `metadata`.

For external retrievers, `metadata` retains upstream reference/file/collection
information. Adapters do not convert upstream graph seeds into EvLink
`edge_evidence`.

### `evidence_needs.jsonl`

One evidence-need set `B(q)` per line.

Main fields:

- `query_id`;
- `question`;
- `requirements`;
- `B_q`;
- `trace`.

Each evidence need includes:

- `unit_id`;
- `subquery`;
- `anchor_mentions`;
- `expected_answer_type`;
- `role`;
- `satisfiable_by`;
- `depends_on`.

### `binding_cache.json`

A stable implementation cache used by coverage-aware evidence selection. Keys
are derived from the evidence need, candidate passage, model name, and prompt
version. Values are lists of extracted support strings.

### `evidence_selection.json`

Coverage-aware evidence selection output.

Main fields:

- `rows`;
- `summary`;
- `evidence_selection_query_traces`;
- `evidence_selection_config`.

Each row contains the selected final evidence set `R_q`, top titles/documents,
retrieval metrics when gold titles are available, and evidence-selection trace metadata.

## Reader Artifact

### `reader_qa.json`

Optional reader output produced from `evidence_selection.json`.

Main fields:

- `rows`;
- `summary`;
- `reader`.

Reader evaluation depends on the provided gold answers and the configured
reader model.
