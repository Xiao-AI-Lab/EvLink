# Aggregate Metric Definitions

EvLink reports retrieval and optional reader metrics separately.

## Retrieval Metrics

Titles are lowercased, parenthetical disambiguation is removed, non-alphanumeric
characters become spaces, and repeated whitespace is collapsed. Duplicate gold
titles retain multiplicity.

- `title_recall_top5`: for each query, the number of matched normalized gold
  title occurrences in the final reader evidence divided by the number of gold
  title occurrences, then averaged across rows.
- `evidence_selection_title_all_gold_top5`: fraction of rows for which the
  final reader evidence covers every normalized gold title with its required
  multiplicity. This is All@5 on `R_q`, not recall of the larger candidate pool.
- `baseline_title_recall_top5` and `baseline_title_all_gold_top5`: the same
  metrics on the upstream top-K before coverage-aware selection when emitted by
  the evaluation path.
- `changed_count`: number of rows whose final evidence ordering/content differs
  from the upstream baseline under the selection trace.

Rows without gold titles cannot support title-recall evaluation. The current
runner emits zero-valued title metrics for such rows; do not aggregate those
rows into a benchmark score.

## Reader Metrics

- `em`: maximum exact match against normalized gold answer aliases.
- `f1`: maximum token F1 against normalized gold answer aliases.

Normalization lowercases text, removes punctuation and English articles, and
collapses whitespace. Reader metrics depend on the exact reader model, prompt,
endpoint implementation, and final evidence passages; report those alongside
the score.
