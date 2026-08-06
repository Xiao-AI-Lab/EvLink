# EvLink v0.1 Release Scope

EvLink v0.1 provides a compact research SDK and reproducibility surface.

## Included

- the `evidencelink` Python package;
- deterministic offline examples and tests;
- dataset download and preprocessing tools;
- paper protocol, configuration, and aggregate metric definitions;
- dependency-free HippoRAG and LightRAG result adapters.

## Excluded

- H100 embedding stores, graph pickles, and model weights;
- SQLite LLM and reader caches;
- per-query private experiment outputs;
- server aliases, private endpoints, and host-specific rebuttal operations;
- vendored LightRAG, HippoRAG, or other baseline implementations.

## Distribution Boundary

The wheel contains only `evidencelink/` and distribution metadata. The source
distribution adds public examples, scripts, deterministic fixtures, protocol
docs, and governance files. Neither distribution contains tests, raw benchmark
files, model weights, caches, databases, pickles, or per-query experiment
outputs.
