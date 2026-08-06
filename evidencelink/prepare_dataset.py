"""Convert supported benchmark datasets into EvidenceLink JSONL inputs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidencelink.artifacts import write_jsonl
from evidencelink.datasets import DatasetSpec, dataset_spec, iter_supported_datasets
from evidencelink.io_utils import write_json


DEFAULT_SOURCE_ROOT = Path("datasets/raw")


def read_source_json(path: str | Path) -> Any:
    """Read one HippoRAG-style benchmark JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_rows(payload: Any, path: Path) -> list[Mapping[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"benchmark file must contain a list: {path}")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"benchmark row {index} is not an object: {path}")
        rows.append(row)
    return rows


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = _clean_string(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _parse_answer_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if cleaned[:1] in {"[", "("} and cleaned[-1:] in {"]", ")"}:
            try:
                parsed = ast.literal_eval(cleaned)
            except (SyntaxError, ValueError):
                return [cleaned]
            return _parse_answer_values(parsed)
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output: list[str] = []
        for item in value:
            output.extend(_parse_answer_values(item))
        return output
    return [_clean_string(value)]


def _corpus_title(row: Mapping[str, Any], fallback_id: int) -> str:
    title = _clean_string(row.get("title"))
    if title:
        return title
    text = _clean_string(row.get("text") or row.get("passage"))
    if "\n" in text:
        first, _ = text.split("\n", 1)
        title = first.strip()
    return title or f"doc_{fallback_id}"


def _corpus_text(row: Mapping[str, Any]) -> str:
    text = _clean_string(row.get("text") or row.get("passage"))
    title = _clean_string(row.get("title"))
    if not title and "\n" in text:
        _, text = text.split("\n", 1)
    return text.strip()


def _passage(title: Any, text: Any) -> str:
    return f"{_clean_string(title)}\n{_clean_string(text)}".strip()


def convert_corpus_rows(
    corpus_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str | DatasetSpec,
) -> list[dict[str, Any]]:
    """Convert source corpus rows into the EvidenceLink corpus schema."""

    spec = dataset if isinstance(dataset, DatasetSpec) else dataset_spec(dataset)
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(corpus_rows):
        source_idx = row.get("idx")
        metadata: dict[str, Any] = {
            "dataset": spec.name,
            "display_dataset": spec.display_name,
            "source_doc_index": index,
        }
        if source_idx is not None:
            metadata["source_idx"] = source_idx
        converted.append(
            {
                "doc_id": str(index),
                "title": _corpus_title(row, index),
                "text": _corpus_text(row),
                "metadata": metadata,
            }
        )
    return converted


def _title_to_doc_ids(corpus_records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in corpus_records:
        title = _clean_string(row.get("title"))
        if title:
            result.setdefault(title, []).append(str(row.get("doc_id")))
    return result


def _passage_to_doc_id(corpus_records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in corpus_records:
        passage = _passage(row.get("title"), row.get("text"))
        normalized = _normalize_text(passage)
        if normalized:
            result.setdefault(normalized, str(row.get("doc_id")))
    return result


def _context_title(item: Any) -> str:
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
        return _clean_string(item[0])
    return ""


def _context_passage(item: Any) -> str:
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
        return ""
    title = _clean_string(item[0])
    sentences = item[1]
    if isinstance(sentences, Sequence) and not isinstance(sentences, (str, bytes)):
        body = " ".join(_clean_string(sentence) for sentence in sentences)
    else:
        body = _clean_string(sentences)
    return _passage(title, body)


def _dict_passage(item: Mapping[str, Any]) -> str:
    return _passage(item.get("title"), item.get("text", item.get("paragraph_text", "")))


def _doc_ids_for_source_passage(
    *,
    title: str,
    passage: str,
    title_to_doc_ids: Mapping[str, Sequence[str]],
    passage_to_doc_id: Mapping[str, str],
) -> list[str]:
    doc_id = passage_to_doc_id.get(_normalize_text(passage))
    if doc_id is not None:
        return [doc_id]
    candidates = title_to_doc_ids.get(title, ())
    if len(candidates) == 1:
        return [str(candidates[0])]
    return []


def _gold_from_supporting_facts(
    row: Mapping[str, Any],
    *,
    title_to_doc_ids: Mapping[str, Sequence[str]],
    passage_to_doc_id: Mapping[str, str],
) -> list[str]:
    context = list(row.get("context", []) or [])
    local_by_title: dict[str, list[Any]] = {}
    for item in context:
        title = _context_title(item)
        if title:
            local_by_title.setdefault(title, []).append(item)

    gold_doc_ids: list[str] = []
    for fact in row.get("supporting_facts", []) or []:
        if not isinstance(fact, Sequence) or isinstance(fact, (str, bytes)) or not fact:
            continue
        title = _clean_string(fact[0])
        local_items = local_by_title.get(title, [])
        if not local_items:
            candidates = title_to_doc_ids.get(title, ())
            if len(candidates) == 1:
                gold_doc_ids.append(str(candidates[0]))
            continue
        for item in local_items:
            gold_doc_ids.extend(
                _doc_ids_for_source_passage(
                    title=title,
                    passage=_context_passage(item),
                    title_to_doc_ids=title_to_doc_ids,
                    passage_to_doc_id=passage_to_doc_id,
                )
            )
    return _dedupe_strings(gold_doc_ids)


def _gold_from_marked_items(
    items: Sequence[Any],
    *,
    title_to_doc_ids: Mapping[str, Sequence[str]],
    passage_to_doc_id: Mapping[str, str],
) -> list[str]:
    gold_doc_ids: list[str] = []
    for item in items:
        if not isinstance(item, Mapping) or not item.get("is_supporting"):
            continue
        title = _clean_string(item.get("title"))
        gold_doc_ids.extend(
            _doc_ids_for_source_passage(
                title=title,
                passage=_dict_passage(item),
                title_to_doc_ids=title_to_doc_ids,
                passage_to_doc_id=passage_to_doc_id,
            )
        )
    return _dedupe_strings(gold_doc_ids)


def _source_query_id(row: Mapping[str, Any], fallback_id: int) -> str:
    for key in ("_id", "id", "query_id", "query_idx", "query_index"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback_id)


def _answers_for_row(spec: DatasetSpec, row: Mapping[str, Any]) -> list[str]:
    if spec.name in {"hotpotqa", "2wikimultihopqa"}:
        return _dedupe_strings([row.get("answer")])
    if spec.name == "musique":
        return _dedupe_strings([row.get("answer"), *_parse_answer_values(row.get("answer_aliases"))])
    if spec.name == "nq_rear":
        return _dedupe_strings(
            [
                *_parse_answer_values(row.get("reference")),
                *_parse_answer_values(row.get("answer_aliases")),
            ]
        )
    if spec.name == "popqa":
        answers: list[str] = []
        for field in ("obj", "possible_answers", "o_wiki_title", "o_aliases", "answer_aliases"):
            answers.extend(_parse_answer_values(row.get(field)))
        return _dedupe_strings(answers)
    raise ValueError(f"unsupported dataset: {spec.name}")


def _gold_doc_ids_for_row(
    spec: DatasetSpec,
    row: Mapping[str, Any],
    *,
    title_to_doc_ids: Mapping[str, Sequence[str]],
    passage_to_doc_id: Mapping[str, str],
) -> list[str]:
    if spec.name in {"hotpotqa", "2wikimultihopqa"}:
        return _gold_from_supporting_facts(
            row,
            title_to_doc_ids=title_to_doc_ids,
            passage_to_doc_id=passage_to_doc_id,
        )
    if spec.name == "musique":
        return _gold_from_marked_items(
            list(row.get("paragraphs", []) or []),
            title_to_doc_ids=title_to_doc_ids,
            passage_to_doc_id=passage_to_doc_id,
        )
    if spec.name == "nq_rear":
        return _gold_from_marked_items(
            list(row.get("contexts", []) or []),
            title_to_doc_ids=title_to_doc_ids,
            passage_to_doc_id=passage_to_doc_id,
        )
    if spec.name == "popqa":
        return _gold_from_marked_items(
            list(row.get("paragraphs", []) or []),
            title_to_doc_ids=title_to_doc_ids,
            passage_to_doc_id=passage_to_doc_id,
        )
    raise ValueError(f"unsupported dataset: {spec.name}")


def convert_question_rows(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str | DatasetSpec,
    corpus_records: Sequence[Mapping[str, Any]],
    max_queries: int = 0,
) -> list[dict[str, Any]]:
    """Convert source question rows into the EvidenceLink question schema."""

    spec = dataset if isinstance(dataset, DatasetSpec) else dataset_spec(dataset)
    selected_rows = list(dataset_rows[: int(max_queries)] if int(max_queries) > 0 else dataset_rows)
    title_to_doc_ids = _title_to_doc_ids(corpus_records)
    passage_to_doc_id = _passage_to_doc_id(corpus_records)
    title_by_doc_id = {str(row.get("doc_id")): _clean_string(row.get("title")) for row in corpus_records}

    converted: list[dict[str, Any]] = []
    for query_index, row in enumerate(selected_rows):
        query_id = _source_query_id(row, query_index)
        gold_doc_ids = _gold_doc_ids_for_row(
            spec,
            row,
            title_to_doc_ids=title_to_doc_ids,
            passage_to_doc_id=passage_to_doc_id,
        )
        gold_titles = _dedupe_strings(title_by_doc_id.get(doc_id) for doc_id in gold_doc_ids)
        converted.append(
            {
                "query_id": query_id,
                "question": _clean_string(row.get("question")),
                "gold_doc_ids": gold_doc_ids,
                "gold_titles": gold_titles,
                "gold_answers": _answers_for_row(spec, row),
                "metadata": {
                    "dataset": spec.name,
                    "display_dataset": spec.display_name,
                    "task_family": spec.task_family,
                    "query_index": query_index,
                    "source_query_id": query_id,
                },
            }
        )
    return converted


def prepare_benchmark_dataset(
    *,
    dataset: str,
    source_root: str | Path,
    output_root: str | Path,
    max_queries: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare one supported benchmark as ``corpus.jsonl`` and ``questions.jsonl``."""

    spec = dataset_spec(dataset)
    source_root_path = Path(source_root)
    output_root_path = Path(output_root)
    dataset_path, corpus_path = spec.source_paths(source_root_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing dataset file for {spec.name}: {dataset_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"missing corpus file for {spec.name}: {corpus_path}")

    output_corpus = output_root_path / "corpus.jsonl"
    output_questions = output_root_path / "questions.jsonl"
    output_manifest = output_root_path / "dataset_manifest.json"
    existing = [path for path in (output_corpus, output_questions, output_manifest) if path.exists()]
    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing prepared files without force=True: {formatted}")

    raw_dataset_rows = _require_rows(read_source_json(dataset_path), dataset_path)
    raw_corpus_rows = _require_rows(read_source_json(corpus_path), corpus_path)
    corpus_records = convert_corpus_rows(raw_corpus_rows, dataset=spec)
    question_records = convert_question_rows(
        raw_dataset_rows,
        dataset=spec,
        corpus_records=corpus_records,
        max_queries=max_queries,
    )

    write_jsonl(corpus_records, output_corpus)
    write_jsonl(question_records, output_questions)

    manifest = {
        "dataset": spec.name,
        "dataset_spec": spec.to_mapping(),
        "source_root": str(source_root_path),
        "source_dataset": str(dataset_path),
        "source_corpus": str(corpus_path),
        "output_root": str(output_root_path),
        "corpus_path": str(output_corpus),
        "questions_path": str(output_questions),
        "corpus_count": len(corpus_records),
        "question_count": len(question_records),
        "max_queries": int(max_queries),
        "format": {
            "corpus": "EvidenceLink corpus.jsonl",
            "questions": "EvidenceLink questions.jsonl",
        },
    }
    write_json(manifest, output_manifest)
    return manifest


def _dataset_names_from_arg(value: str) -> list[str]:
    raw = str(value).strip().lower()
    if raw == "all":
        return [spec.name for spec in iter_supported_datasets()]
    return [dataset_spec(item.strip()).name for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset name, comma-separated names, or 'all'. Defaults to all five paper datasets.",
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Directory containing <dataset>.json and <dataset>_corpus.json. Defaults to datasets/raw.",
    )
    parser.add_argument("--output-root", help="Output directory for prepared EvidenceLink inputs.")
    parser.add_argument("--max-queries", type=int, default=0, help="Optional question limit; 0 keeps all rows.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing prepared files.")
    parser.add_argument("--list-datasets", action="store_true", help="Print supported datasets and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.list_datasets:
        print(json.dumps([spec.to_mapping() for spec in iter_supported_datasets()], ensure_ascii=False, indent=2))
        return 0

    if not args.output_root:
        raise SystemExit("--output-root is required unless --list-datasets is used")

    dataset_names = _dataset_names_from_arg(args.dataset)
    output_root = Path(args.output_root)
    manifests = []
    for name in dataset_names:
        destination = output_root / name if len(dataset_names) > 1 else output_root
        manifests.append(
            prepare_benchmark_dataset(
                dataset=name,
                source_root=args.source_root,
                output_root=destination,
                max_queries=args.max_queries,
                force=args.force,
            )
        )
    print(json.dumps(manifests[0] if len(manifests) == 1 else manifests, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
