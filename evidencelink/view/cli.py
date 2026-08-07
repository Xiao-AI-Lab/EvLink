"""Build one ``QueryResultView/v1`` JSON document from EvLink artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencelink.view import build_query_result_view_from_files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--reader", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    view = build_query_result_view_from_files(
        query_id=args.query_id,
        candidate_pool_path=args.candidate_pool,
        selection_path=args.selection,
        reader_path=args.reader,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output), "query_id": view["query_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
