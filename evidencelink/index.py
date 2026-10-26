"""Source-grounded evidence-link index construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencelink.artifacts import Document, OpenIEFact, load_documents, load_openie_facts


def normalize_endpoint(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


MAX_ENDPOINT_ALIGNED_DOC_COUNT = 50

_LINK_TYPE_TRAVERSAL_PRIORITY = {"relation_grounded": 0, "endpoint_aligned": 1}


@dataclass(frozen=True)
class EvidenceLink:
    source_doc_id: str
    target_doc_id: str
    link_type: str
    relation: str = ""
    endpoint: str = ""
    witnesses: tuple[Mapping[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["witnesses"] = [dict(item) for item in self.witnesses]
        return payload


@dataclass
class EvidenceLinkIndex:
    documents: dict[str, Document] = field(default_factory=dict)
    links: list[EvidenceLink] = field(default_factory=list)
    _neighbors_by_source: dict[str, list[EvidenceLink]] = field(default_factory=dict, init=False, repr=False)
    _witnesses_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        neighbors: dict[str, list[EvidenceLink]] = defaultdict(list)
        witnesses: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for link in self.links:
            neighbors[link.source_doc_id].append(link)
            witnesses[(link.source_doc_id, link.target_doc_id)].extend(dict(item) for item in link.witnesses)
        for doc_links in neighbors.values():
            doc_links.sort(key=lambda link: _LINK_TYPE_TRAVERSAL_PRIORITY.get(link.link_type, 99))
        self._neighbors_by_source = dict(neighbors)
        self._witnesses_by_edge = dict(witnesses)

    @classmethod
    def build(cls, documents: Sequence[Document], facts: Sequence[OpenIEFact]) -> "EvidenceLinkIndex":
        docs = {str(doc.doc_id): doc for doc in documents}
        title_to_doc_ids: dict[str, list[str]] = defaultdict(list)
        scope_by_doc_id = {
            str(doc.doc_id): str((doc.metadata or {}).get("scope_id") or "").strip()
            for doc in documents
        }
        for doc in documents:
            title_to_doc_ids[normalize_endpoint(doc.title)].append(str(doc.doc_id))

        def compatible_scopes(source_doc_id: str, target_doc_id: str) -> bool:
            source_scope = scope_by_doc_id.get(str(source_doc_id), "")
            target_scope = scope_by_doc_id.get(str(target_doc_id), "")
            return not source_scope or not target_scope or source_scope == target_scope

        links_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        doc_endpoints: dict[str, set[str]] = defaultdict(set)
        for fact in facts:
            if fact.doc_id not in docs:
                continue
            endpoints = [normalize_endpoint(fact.subject), normalize_endpoint(fact.object)]
            for endpoint in endpoints:
                if endpoint:
                    doc_endpoints[fact.doc_id].add(endpoint)
            witness = fact.to_mapping()
            for endpoint in endpoints:
                for target_doc_id in title_to_doc_ids.get(endpoint, []):
                    if target_doc_id == fact.doc_id or not compatible_scopes(fact.doc_id, target_doc_id):
                        continue
                    key = (fact.doc_id, target_doc_id, "relation_grounded", endpoint)
                    links_by_key[key].append(witness)

        endpoint_to_doc_ids: dict[str, list[str]] = defaultdict(list)
        for doc_id, endpoints in doc_endpoints.items():
            for endpoint in endpoints:
                endpoint_to_doc_ids[endpoint].append(doc_id)
        for endpoint, endpoint_doc_ids in endpoint_to_doc_ids.items():
            if len(endpoint_doc_ids) > MAX_ENDPOINT_ALIGNED_DOC_COUNT:
                continue
            for src in endpoint_doc_ids:
                for dst in endpoint_doc_ids:
                    if src == dst or not compatible_scopes(src, dst):
                        continue
                    key = (src, dst, "endpoint_aligned", endpoint)
                    links_by_key[key].append({"endpoint": endpoint, "source": "endpoint_alignment"})

        links = [
            EvidenceLink(
                source_doc_id=src,
                target_doc_id=dst,
                link_type=link_type,
                relation=str((witnesses[0] or {}).get("relation") or ""),
                endpoint=endpoint,
                witnesses=tuple(witnesses),
            )
            for (src, dst, link_type, endpoint), witnesses in sorted(links_by_key.items())
        ]
        return cls(documents=docs, links=links)

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceLinkIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        documents = {
            str(row.get("doc_id")): Document.from_mapping(row)
            for row in list(payload.get("documents") or [])
            if isinstance(row, Mapping)
        }
        links = [
            EvidenceLink(
                source_doc_id=str(row.get("source_doc_id")),
                target_doc_id=str(row.get("target_doc_id")),
                link_type=str(row.get("link_type")),
                relation=str(row.get("relation") or ""),
                endpoint=str(row.get("endpoint") or ""),
                witnesses=tuple(dict(item) for item in list(row.get("witnesses") or []) if isinstance(item, Mapping)),
            )
            for row in list(payload.get("links") or [])
            if isinstance(row, Mapping)
        ]
        return cls(documents=documents, links=links)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "evidence_link_index",
            "documents": [doc.to_mapping() for doc in self.documents.values()],
            "links": [link.to_mapping() for link in self.links],
            "summary": {
                "document_count": len(self.documents),
                "link_count": len(self.links),
                "relation_grounded_link_count": sum(link.link_type == "relation_grounded" for link in self.links),
                "endpoint_aligned_link_count": sum(link.link_type == "endpoint_aligned" for link in self.links),
            },
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def neighbors(self, doc_id: str) -> list[EvidenceLink]:
        return list(self._neighbors_by_source.get(str(doc_id), ()))

    def edge_witnesses(self, source_doc_id: str, target_doc_id: str) -> list[dict[str, Any]]:
        return [
            dict(witness)
            for witness in self._witnesses_by_edge.get((str(source_doc_id), str(target_doc_id)), ())
        ]


def build_evidence_link_index(
    *,
    corpus_path: str | Path,
    openie_path: str | Path,
    output_path: str | Path,
) -> EvidenceLinkIndex:
    index = EvidenceLinkIndex.build(load_documents(corpus_path), load_openie_facts(openie_path))
    index.save(output_path)
    return index


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Build a source-grounded EvidenceLink index from corpus and OpenIE facts.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--openie", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    index = build_evidence_link_index(corpus_path=args.corpus, openie_path=args.openie, output_path=args.output)
    print(json.dumps({"output": str(args.output), "documents": len(index.documents), "links": len(index.links)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
