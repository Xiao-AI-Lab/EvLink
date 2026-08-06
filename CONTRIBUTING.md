# Contributing to EvLink

EvLink accepts focused bug fixes, documentation improvements, retriever
adapters, and reproducibility improvements that preserve the public artifact
contract. Discuss broad API or method changes in an issue before implementation.

## Development Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
```

Run the offline examples before submitting a change:

```bash
python examples/end_to_end.py
python examples/external_retriever.py
python examples/integrations/hipporag.py
python examples/integrations/lightrag.py
```

## Change Requirements

- Keep Python 3.10 compatibility.
- Add focused tests for changed behavior and failure modes.
- Preserve candidate ordering, identifiers, scores, and provenance labels.
- Do not present external-retriever candidates as EvLink source-grounded
  graph links.
- Keep default tests and examples offline and deterministic.
- Do not commit model weights, benchmark source files, embeddings, graph
  pickles, caches, private endpoints, credentials, or per-query experiment
  outputs.
- Update `ARTIFACTS.md` or `reproduce/README.md` when an artifact schema or
  evaluation protocol changes.

## Pull Requests

Describe the behavioral change, the affected artifact/API contract, and the
commands used for verification. Keep pull requests focused and avoid unrelated
formatting or generated-artifact churn.
