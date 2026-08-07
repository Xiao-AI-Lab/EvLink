"""Build the stable ``QueryResultView/v1`` application projection."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencelink.artifacts import read_json_or_jsonl


QUERY_RESULT_VIEW_SCHEMA_VERSION = "query_result_view/v1"


def _records(payload: Any, *, root_key: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = payload.get(root_key or "records") or []
    else:
        values = []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def _query_id(row: Mapping[str, Any]) -> str:
    value = row.get("query_id", row.get("query_idx", row.get("query_index", "")))
    return str(value if value is not None else "")


def _one_by_query_id(rows: Sequence[Mapping[str, Any]], query_id: str, *, artifact: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if _query_id(row) == str(query_id)]
    if len(matches) != 1:
        raise ValueError(
            f"{artifact} must contain exactly one row for query_id={query_id!r}, found {len(matches)}"
        )
    return matches[0]


def _candidate_rows(pool_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in list(pool_row.get("candidate_pool") or [])
        if isinstance(row, Mapping)
    ]
    if candidates:
        return candidates
    docs = list(pool_row.get("pool_docs") or [])
    titles = list(pool_row.get("pool_titles") or [])
    doc_ids = list(pool_row.get("pool_doc_ids") or [])
    scores = list(pool_row.get("pool_doc_scores") or [])
    return [
        {
            "rank": pos + 1,
            "doc_id": doc_ids[pos] if pos < len(doc_ids) else pos,
            "title": titles[pos] if pos < len(titles) else str(doc).split("\n", 1)[0],
            "text": str(doc).split("\n", 1)[1] if "\n" in str(doc) else str(doc),
            "score": scores[pos] if pos < len(scores) else 0.0,
            "source": "unknown",
        }
        for pos, doc in enumerate(docs)
    ]


def _coverage_status(value: float) -> str:
    if value <= 0.0:
        return "unsupported"
    if value >= 1.0 - 1e-9:
        return "covered"
    return "partial"


def _selection_roles(
    *,
    position: int,
    doc_id: str,
    pool_row: Mapping[str, Any],
    selection_trace: Mapping[str, Any],
    event: Mapping[str, Any] | None,
) -> list[str]:
    baseline = {int(value) for value in list(selection_trace.get("baseline_positions") or [])}
    final = {int(value) for value in list(selection_trace.get("final_positions") or [])}
    protected = {
        int(value)
        for value in list(
            selection_trace.get("protected_baseline_positions")
            or selection_trace.get("stable_seed_positions")
            or []
        )
    }
    pool_trace = dict(pool_row.get("pool_trace") or {})
    anchor_seeds = {
        str(value)
        for value in list(
            pool_row.get("anchor_seed_doc_ids")
            or pool_trace.get("anchor_seed_doc_ids")
            or []
        )
    }
    dense_seeds = {
        str(value)
        for value in list(
            pool_row.get("dense_seed_doc_ids")
            or pool_trace.get("dense_seed_doc_ids")
            or []
        )
    }
    roles: list[str] = []
    if doc_id in anchor_seeds:
        roles.append("anchor_seed")
    if doc_id in dense_seeds:
        roles.append("dense_seed")
    if event and str(event.get("discovery_method") or "") == "bfs" and not (doc_id in anchor_seeds or doc_id in dense_seeds):
        roles.append("bridge")
    if position in baseline and position in final:
        roles.append("retained")
    if position in final and position not in baseline:
        roles.append("admitted")
    if position in baseline and position not in final:
        roles.append("displaced")
    if position in protected:
        roles.append("protected")
    if position in final:
        roles.append("selected")
    return roles


def _graph(
    *,
    query_id: str,
    question: str,
    answer: Mapping[str, Any],
    needs: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    support_matrix: Sequence[Mapping[str, Any]],
    discovery_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {"id": f"question:{query_id}", "type": "question", "label": question},
    ]
    if str(answer.get("text") or ""):
        nodes.append({"id": f"answer:{query_id}", "type": "answer", "label": str(answer["text"])})
    nodes.extend(
        {
            "id": f"need:{need['need_id']}",
            "type": "evidence_need",
            "label": str(need.get("subquery") or need["need_id"]),
            "status": str(need.get("coverage_status") or "unsupported"),
        }
        for need in needs
    )
    nodes.extend(
        {
            "id": f"passage:{passage['passage_id']}",
            "type": "passage",
            "label": str(passage.get("title") or passage["passage_id"]),
            "roles": list(passage.get("roles") or []),
        }
        for passage in passages
    )

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    def add_edge(source: str, target: str, edge_type: str, *, data: Mapping[str, Any] | None = None) -> None:
        payload = dict(data or {})
        identity = (source, target, edge_type, json.dumps(payload, sort_keys=True, ensure_ascii=False))
        if identity in seen_edges:
            return
        seen_edges.add(identity)
        edges.append(
            {
                "id": f"edge:{len(edges) + 1}",
                "source": source,
                "target": target,
                "type": edge_type,
                "data": payload,
            }
        )

    for need in needs:
        need_id = str(need["need_id"])
        dependencies = list(need.get("depends_on") or [])
        if not dependencies:
            add_edge(f"question:{query_id}", f"need:{need_id}", "evidence_need")
        for dependency in dependencies:
            add_edge(f"need:{dependency}", f"need:{need_id}", "need_dependency")
    for row in support_matrix:
        if float(row.get("support_score", 0.0) or 0.0) <= 0.0:
            continue
        add_edge(
            f"need:{row['need_id']}",
            f"passage:{row['passage_id']}",
            "need_support",
            data={
                "support_score": float(row.get("support_score", 0.0) or 0.0),
                "final_coverage_delta": float(row.get("final_coverage_delta", 0.0) or 0.0),
                "supporting_bindings": list(row.get("supporting_bindings") or []),
            },
        )
    for event in discovery_events:
        raw_links = list(event.get("incoming_links") or [])
        if not raw_links and isinstance(event.get("incoming_link"), Mapping):
            raw_links = [event["incoming_link"]]
        for link in raw_links:
            if not isinstance(link, Mapping):
                continue
            add_edge(
                f"passage:{link.get('source_doc_id')}",
                f"passage:{link.get('target_doc_id')}",
                str(link.get("link_type") or "evidence_link"),
                data={
                    "relation": str(link.get("relation") or ""),
                    "endpoint": str(link.get("endpoint") or ""),
                    "witnesses": [dict(item) for item in list(link.get("witnesses") or []) if isinstance(item, Mapping)],
                },
            )
    for citation in list(answer.get("citations") or []):
        if not isinstance(citation, Mapping):
            continue
        add_edge(
            f"answer:{query_id}",
            f"passage:{citation.get('passage_id')}",
            "citation",
            data={"marker": str(citation.get("marker") or "")},
        )
    return {"nodes": nodes, "edges": edges}


def build_query_result_view(
    *,
    query_id: str,
    pool_payload: Any,
    selection_payload: Mapping[str, Any],
    reader_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Join one query across pool, selection, and reader artifacts."""
    clean_query_id = str(query_id).strip()
    if not clean_query_id:
        raise ValueError("query_id must not be empty")
    pool_row = _one_by_query_id(_records(pool_payload), clean_query_id, artifact="candidate pool")
    selection_row = _one_by_query_id(
        _records(selection_payload, root_key="rows"),
        clean_query_id,
        artifact="evidence selection",
    )
    reader_row = _one_by_query_id(
        _records(reader_payload, root_key="rows"),
        clean_query_id,
        artifact="reader",
    )
    question = str(pool_row.get("question") or selection_row.get("question") or reader_row.get("question") or "")
    if not question:
        raise ValueError(f"missing question for query_id={clean_query_id!r}")

    trace = dict(selection_row.get("evidence_selection") or {})
    raw_trace = dict(selection_row.get("raw_selection_trace") or {})
    local_subgraph = dict(pool_row.get("local_subgraph") or (pool_row.get("pool_trace") or {}).get("local_subgraph") or {})
    discovery_events = [
        dict(event)
        for event in list(local_subgraph.get("discovery_events") or [])
        if isinstance(event, Mapping)
    ]
    event_by_doc_id = {str(event.get("doc_id")): event for event in discovery_events}
    candidates = _candidate_rows(pool_row)
    passages: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        passage_id = str(candidate.get("doc_id", position))
        event = event_by_doc_id.get(passage_id)
        passages.append(
            {
                "passage_id": passage_id,
                "position": int(position),
                "rank": int(candidate.get("rank", position + 1) or position + 1),
                "title": str(candidate.get("title") or ""),
                "text": str(candidate.get("text") or ""),
                "source": str(candidate.get("source") or "unknown"),
                "score": float(candidate.get("score", 0.0) or 0.0),
                "path": [str(value) for value in list(candidate.get("path") or [])],
                "edge_evidence": [dict(item) for item in list(candidate.get("edge_evidence") or []) if isinstance(item, Mapping)],
                "roles": _selection_roles(
                    position=position,
                    doc_id=passage_id,
                    pool_row=pool_row,
                    selection_trace=trace,
                    event=event,
                ),
                "metadata": dict(candidate.get("metadata") or {}),
            }
        )

    support_matrix = [
        dict(row)
        for row in list(raw_trace.get("support_matrix") or [])
        if isinstance(row, Mapping)
    ]
    coverage_by_need = {
        str(key): float(value)
        for key, value in dict(raw_trace.get("coverage_by_requirement") or {}).items()
    }
    needs = []
    for requirement in list(raw_trace.get("requirements") or []):
        if not isinstance(requirement, Mapping):
            continue
        need_id = str(requirement.get("unit_id") or requirement.get("id") or "")
        coverage = float(coverage_by_need.get(need_id, 0.0))
        supporting_passage_ids = [
            str(row.get("passage_id"))
            for row in support_matrix
            if str(row.get("need_id")) == need_id and float(row.get("support_score", 0.0) or 0.0) > 0.0
        ]
        needs.append(
            {
                "need_id": need_id,
                "subquery": str(requirement.get("subquery") or ""),
                "depends_on": [str(value) for value in list(requirement.get("depends_on") or [])],
                "expected_answer_type": str(requirement.get("expected_answer_type") or "unknown"),
                "anchor_mentions": [str(value) for value in list(requirement.get("anchor_mentions") or [])],
                "role": str(requirement.get("role") or "support"),
                "satisfiable_by": str(requirement.get("satisfiable_by") or "unknown"),
                "coverage_estimate": round(coverage, 6),
                "coverage_status": _coverage_status(coverage),
                "supporting_passage_ids": supporting_passage_ids,
            }
        )

    answer = {
        "text": str(reader_row.get("prediction") or ""),
        "citations": [dict(value) for value in list(reader_row.get("citations") or []) if isinstance(value, Mapping)],
        "claims": None,
    }
    pool_trace = dict(pool_row.get("pool_trace") or {})
    input_method = str(
        pool_trace.get("input_method")
        or (selection_payload.get("pool_protocol") or {}).get("upstream_retriever")
        or "unknown"
    )
    witness_available = input_method == "source_grounded_evidence_link_pool" and bool(discovery_events)
    retrieval_trace = {
        "schema_version": "retrieval_trace/v1",
        "input_method": input_method,
        "anchor_seed_doc_ids": [
            str(value)
            for value in list(
                pool_row.get("anchor_seed_doc_ids")
                or pool_trace.get("anchor_seed_doc_ids")
                or []
            )
        ],
        "dense_seed_doc_ids": [
            str(value)
            for value in list(
                pool_row.get("dense_seed_doc_ids")
                or pool_trace.get("dense_seed_doc_ids")
                or []
            )
        ],
        "discovery_events": discovery_events,
        "capabilities": {
            "witnesses": "available" if witness_available else "unavailable",
            "faithful_step_through": bool(discovery_events),
        },
    }
    selection_projection = {
        "schema_version": "selection_trace/v1",
        "baseline_positions": list(trace.get("baseline_positions") or []),
        "final_positions": list(trace.get("final_positions") or []),
        "protected_baseline_positions": list(
            trace.get("protected_baseline_positions")
            or trace.get("stable_seed_positions")
            or []
        ),
        "admitted_positions": list(trace.get("admitted_positions") or []),
        "decision": str(trace.get("decision") or ""),
        "rank_stability_held": bool(trace.get("rank_stability_held")),
        "baseline_objective": raw_trace.get("baseline_objective"),
        "final_objective": raw_trace.get("objective"),
        "admission_steps": list(trace.get("admission_steps") or []),
    }
    view = {
        "artifact_schema_version": QUERY_RESULT_VIEW_SCHEMA_VERSION,
        "query_id": clean_query_id,
        "question": question,
        "answer": answer,
        "needs": needs,
        "passages": passages,
        "support_matrix": support_matrix,
        "evidence_graph": _graph(
            query_id=clean_query_id,
            question=question,
            answer=answer,
            needs=needs,
            passages=passages,
            support_matrix=support_matrix,
            discovery_events=discovery_events,
        ),
        "retrieval_trace": retrieval_trace,
        "selection_trace": selection_projection,
        "provenance": {
            "method": str(selection_payload.get("paper_facing_method") or selection_payload.get("method") or "EvLink"),
            "dataset": str(selection_payload.get("dataset") or ""),
            "pool_protocol": dict(selection_payload.get("pool_protocol") or {}),
            "reader": dict(reader_payload.get("reader") or {}),
            "source_artifact_versions": {
                "support_matrix": str(raw_trace.get("support_matrix_schema_version") or ""),
            },
        },
    }
    validate_query_result_view(view)
    return view


def build_query_result_view_from_files(
    *,
    query_id: str,
    candidate_pool_path: str | Path,
    selection_path: str | Path,
    reader_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    view = build_query_result_view(
        query_id=query_id,
        pool_payload=read_json_or_jsonl(candidate_pool_path),
        selection_payload=dict(read_json_or_jsonl(selection_path) or {}),
        reader_payload=dict(read_json_or_jsonl(reader_path) or {}),
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(view, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return view


def load_query_result_view_schema() -> dict[str, Any]:
    schema_path = files("evidencelink.view.schemas").joinpath("query_result_view_v1.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_query_result_view(view: Mapping[str, Any]) -> None:
    required = {
        "artifact_schema_version",
        "query_id",
        "question",
        "answer",
        "needs",
        "passages",
        "support_matrix",
        "evidence_graph",
        "retrieval_trace",
        "selection_trace",
        "provenance",
    }
    missing = sorted(required.difference(view))
    if missing:
        raise ValueError(f"QueryResultView/v1 is missing fields: {missing}")
    if str(view.get("artifact_schema_version")) != QUERY_RESULT_VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported QueryResultView schema version")
    if not str(view.get("query_id") or "").strip():
        raise ValueError("QueryResultView query_id must not be empty")
    answer = view.get("answer")
    if not isinstance(answer, Mapping) or not isinstance(answer.get("citations"), list):
        raise ValueError("QueryResultView answer.citations must be a list")
    if str(answer.get("text") or "").strip() and not answer.get("citations"):
        raise ValueError("QueryResultView answers require at least one passage citation")
    passage_ids = {str(row.get("passage_id")) for row in list(view.get("passages") or []) if isinstance(row, Mapping)}
    need_ids = {str(row.get("need_id")) for row in list(view.get("needs") or []) if isinstance(row, Mapping)}
    dangling = [
        str(citation.get("passage_id"))
        for citation in list(answer.get("citations") or [])
        if isinstance(citation, Mapping) and str(citation.get("passage_id")) not in passage_ids
    ]
    if dangling:
        raise ValueError(f"QueryResultView contains dangling citations: {dangling}")
    dangling_matrix = [
        (str(row.get("need_id")), str(row.get("passage_id")))
        for row in list(view.get("support_matrix") or [])
        if isinstance(row, Mapping)
        and (
            str(row.get("need_id")) not in need_ids
            or str(row.get("passage_id")) not in passage_ids
        )
    ]
    if dangling_matrix:
        raise ValueError(f"QueryResultView contains dangling support cells: {dangling_matrix}")
    events = list((view.get("retrieval_trace") or {}).get("discovery_events") or [])
    steps = [int(event.get("step", 0)) for event in events if isinstance(event, Mapping)]
    if steps and steps != list(range(1, len(steps) + 1)):
        raise ValueError("QueryResultView discovery event steps must be contiguous and ordered")
