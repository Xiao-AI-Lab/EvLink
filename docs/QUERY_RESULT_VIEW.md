# QueryResultView/v1

`QueryResultView/v1` is the stable application boundary between the EvLink SDK
and EvLink Studio. It joins one exact `query_id` across the candidate pool,
selection output, and reader output. Studio must not reconstruct application
state from implementation-specific raw traces.

## Build

```bash
evidencelink-build-view \
  --query-id QUERY_ID \
  --candidate-pool candidate_pool.jsonl \
  --selection evidence_selection.json \
  --reader reader_qa.json \
  --output query_result_view.json
```

The Python API is `evidencelink.build_query_result_view_from_files`. Joins are
strict: every source must contain exactly one matching `query_id`.

## Contract

The view contains:

- `answer`: reader text, required passage citations, and nullable claims;
- `needs`: evidence needs with coverage estimates and status;
- `passages`: ordered pool passages with retrieval and selection roles;
- `support_matrix`: normalized `need_passage_support/v1` cells;
- `evidence_graph`: question, answer, need, passage nodes and typed edges;
- `retrieval_trace`: seed roles, ordered discovery events, paths, links, and witnesses;
- `selection_trace`: baseline, final, protected, admitted, and decision fields;
- `provenance`: method, dataset, pool protocol, reader, and source schema versions.

The packaged JSON Schema is
`evidencelink/view/schemas/query_result_view_v1.schema.json`. The Python builder
also enforces referential integrity for citations and support cells.

## Capability Boundary

Self-index output exposes source-grounded link witnesses and faithful
step-through events. External candidate pools report both capabilities as
unavailable unless the adapter supplies EvLink-native events. Third-party
graph metadata is retained as provenance but is never relabeled as an EvLink
witness.

Coverage is a selector estimate, not factual verification or a calibrated
refusal signal. Claims remain `null` in v1; claim segmentation and
claim-to-passage alignment are outside this milestone.
