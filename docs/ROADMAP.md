# EvLink Release and Studio Roadmap

This roadmap separates the frozen research SDK from the application layer. The
v0.1 release remains a compact, reproducible EvLink core. EvLink Studio starts
in v0.2 as an additive consumer of versioned public artifacts.

## Product Thesis

EvLink Studio should make one query inspectable end to end:

```text
documents -> EvLink index -> ask -> cited answer
          -> evidence needs -> selected passages -> evidence graph
          -> baseline-versus-EvLink comparison
```

Ask is the primary entry point. The Evidence Ledger is the core explanatory
surface. A query-local passage graph is the visual differentiator. The external
retriever API provides integration value.

Unlike a corpus-centric entity graph, the default EvLink graph should explain
why a specific answer received a specific evidence set. Passages remain the
retrieval nodes, evidence-link edges retain source witnesses, and selection
roles are derived from persisted artifacts.

## Execution Sequence

| Milestone | Primary deliverable | Suggested effort | Exit decision |
| --- | --- | --- | --- |
| 0 | Frozen v0.1 SDK and release candidate | 0.5-1 day | Tag only after every release gate passes |
| 1 | `QueryResultView/v1`, retrieval events, support matrix | 3-5 days | Frontend starts only after one fixture validates end to end |
| 2 | Ask and Evidence Ledger MVP | 1.5-2 weeks | A fresh user can ask and inspect cited passages |
| 3 | Query-local evidence graph and witness inspector | 1-1.5 weeks | Graph roles and steps match persisted artifacts |
| 4 | Compare report, export, Docker demo, recording | 3-5 days | Portfolio release gate passes |
| 5 | External integrations and product validation | follow-up | Prioritize from observed user behavior |

Milestones 0-4 form the portfolio critical path and should fit in roughly four
to six focused weeks. Milestone 5 must not delay the first Studio release.
Contract, backend, and frontend work may overlap only after the relevant view
schema and fixtures are frozen.

## Milestone 0: Freeze v0.1 Core

### Scope

Make two bounded, backward-compatible fixes before creating the v0.1.0 tag.
Do not add application APIs, a server, or a frontend in this milestone.

1. Make external-pool provenance self-describing.
   - Preserve the current row-level `pool_trace.input_method`.
   - Set root-level `pool_protocol.upstream_retriever` to
     `external_retriever` when the selector-created pool has no root
     `retrieval.input_method`.
   - Keep `source_grounded_evidence_link_pool` as the main paper protocol.
   - Add regression coverage for both self-index and external-pool paths.

2. Clarify rank-stability terminology without breaking the artifact schema.
   - Document `stable_seed_positions` as protected positions in the baseline
     reader set, not dense or anchor retrieval seeds.
   - Add `protected_baseline_positions` and
     `protected_baseline_titles` as additive aliases in new output.
   - Retain the existing fields through v0.x for compatibility.

### Release Gate

- `python -m pytest tests -q` passes in the public checkout.
- Offline examples pass.
- Python 3.10-3.13 GitHub Actions pass.
- Wheel and sdist build successfully.
- `twine check` and `scripts/check_release_artifacts.py` pass.
- An isolated wheel install and CLI smoke test pass.
- The worktree is clean and local `main` matches `origin/main`.
- The tag points at the reviewed commit and the repository remains private
  until public-release approval.

### Frozen v0.1 Boundary

The v0.1 contract includes the Python SDK, stage CLIs, deterministic examples,
dataset preparation, paper-facing artifacts, metric definitions, and external
candidate adapters. It excludes the Studio server, frontend, workspaces, task
database, PDF parsing, and application-specific view models.

## Milestone 1: Versioned Application Contract

Create an additive projection layer under `evidencelink/view/`. Studio must
consume this public view model instead of parsing internal traces directly.

### `QueryResultView/v1`

```json
{
  "artifact_schema_version": "query_result_view/v1",
  "query_id": "...",
  "question": "...",
  "answer": {
    "text": "...",
    "citations": [{"passage_id": "p2", "marker": "[2]"}],
    "claims": null
  },
  "needs": [],
  "passages": [],
  "support_matrix": [],
  "evidence_graph": {"nodes": [], "edges": []},
  "retrieval_trace": {},
  "selection_trace": {},
  "provenance": {}
}
```

Claims remain optional in v1. Passage-level citations are required. Claim
segmentation and claim-to-passage alignment must not block the first Studio
release.

### Retrieval Trace

Persist explicit, query-scoped retrieval roles and events:

- `anchor_seed_doc_ids`;
- `dense_seed_doc_ids`;
- ordered discovery events with `step`, `doc_id`, `depth`, and
  `parent_doc_id`;
- whether a candidate came from BFS or dense fallback;
- link type and witnesses for every traversed hop.

The current candidate artifact already preserves `seed_doc_ids`, candidate
order, paths, and the final-hop witness. This milestone makes the semantics
explicit enough for faithful step-through playback.

### Support Matrix

Export a stable need-by-passage projection with support score, coverage delta,
selected status, and supporting bindings. The viewer must not reconstruct this
matrix from implementation-specific fields in the raw selection trace.

### Acceptance Criteria

- One deterministic fixture produces a schema-validated view document.
- The same `query_id` joins question, pool, selection, reader, and view output.
- Self-index output exposes witnesses; external-pool output explicitly reports
  witness capability as unavailable.
- Existing v0.1 artifacts and APIs continue to pass their tests.

## Milestone 2: Ask and Evidence Ledger MVP

Build the thinnest complete application loop.

### User Flow

1. Open a preloaded demo collection or add text/Markdown documents.
2. Build an EvLink index and show task progress.
3. Ask a multi-document question.
4. Read an answer with passage-level citation markers.
5. Inspect selected passage cards and the Evidence Ledger.

### Evidence Ledger

For each evidence need, display:

- coverage estimate and status;
- passages contributing support;
- marginal coverage gain;
- whether the need is covered, partial, or unsupported;
- the selection reason for each final passage.

Coverage is an estimate, not a factual verification or calibrated refusal
policy. Product copy must preserve that distinction.

### Initial Stack

- FastAPI service;
- React and TypeScript frontend;
- SQLite for local workspace and task state;
- filesystem-backed EvLink artifacts;
- Docker Compose for one-command startup.

The first MVP may ship with a preloaded demo before general PDF ingestion. Do
not build authentication, teams, billing, or multi-tenant storage.

## Milestone 3: Query-Local Evidence Graph

Use React Flow with a layered layout. Sigma-style large corpus visualization is
not the first target.

### Node Types

- question;
- evidence need;
- passage.

Answer-claim nodes are optional after passage-level citation is stable.

### Retrieval and Selection Roles

- `Seed`: document ID appears in dense or anchor retrieval seeds;
- `Bridge`: non-seed passage reached through an evidence-link path;
- `Retained`: passage appears in both baseline and final sets;
- `Admitted`: passage appears in final but not baseline;
- `Displaced`: passage appears in baseline but not final;
- `Protected`: baseline position protected by rank stability;
- `Selected`: passage appears in the final reader set.

A passage can have multiple roles. Role badges must not be collapsed into one
ambiguous color.

### Edge Types

- evidence-need dependency;
- relation-grounded evidence link;
- endpoint-alignment fallback;
- need-to-passage support;
- passage-level citation.

Clicking an evidence-link edge opens the source passage, relation/OpenIE fact,
endpoint, source span, link type, and witnesses.

### First Interaction Set

- synchronized answer citation, passage card, ledger row, and graph highlight;
- witness inspector;
- manual retrieval step-through;
- filters for seeds, bridge passages, selected passages, and link types.

Animated replay is optional. Correct step semantics are required.

## Milestone 4: Compare and Share

### Compare v1

Start with a table and passage diff:

- baseline Top-K versus EvLink final Top-K;
- retained, admitted, and displaced passages;
- coverage before and after selection;
- changed-from-baseline and selection decision;
- retrieval metrics only when gold labels are available.

Graph overlays and multiple external systems come later. The external adapter
path must be presented as coverage selection without fabricated source-grounded
witnesses.

### Shareable Output

- export `QueryResultView/v1` JSON;
- export a static HTML query report;
- record a 30-second Ask -> Ledger -> Compare -> Witness demo;
- add the demo and one measured multi-hop case to the repository README.

An online hosted demo is optional for the first portfolio release. A reliable
local Docker demo and recorded walkthrough are required.

## Milestone 5: Integration and Product Validation

- expose the versioned view through REST endpoints;
- accept ordered external candidate pools through the existing selector API;
- preserve retriever names and provenance end to end;
- add HippoRAG and LightRAG integration demonstrations;
- measure whether users spend time in Ask, Ledger, Compare, or Witness views;
- decide whether the long-term product is a standalone workbench or an
  Evidence Selection API with an attached debugger.

## Repository Layout Target

```text
evidencelink/           core SDK
evidencelink/view/      versioned application projections
server/                 FastAPI application boundary
apps/studio/            React frontend
examples/               SDK and integration examples
reproduce/              paper reproduction protocols
```

The core package remains usable without installing or running Studio.

## Explicit Non-Goals for the First Studio Release

- a general-purpose corpus entity-graph browser;
- claim-level factual verification;
- a calibrated refusal policy;
- collaborative editing or multi-tenant SaaS;
- support for every document parser and vector store;
- a replacement for the user's existing RAG system.

## Portfolio Release Gate

The first portfolio-ready Studio release is complete when a fresh user can:

1. run one documented startup command;
2. open a preloaded multi-hop collection;
3. ask a question and receive a cited answer;
4. inspect needs, coverage, selected passages, and witnesses;
5. compare baseline and EvLink evidence;
6. export a stable report;
7. reproduce the demo shown in the README.

The resume story should be grounded in measured benchmark results, public
tests, reproducible deployment, and the observable application workflow.
