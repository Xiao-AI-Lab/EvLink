"""Download checksum-pinned benchmark source files for EvLink."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence
import urllib.request


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data" / "dataset_sources.json"
DEFAULT_OUTPUT_ROOT = Path("datasets/raw")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("datasets"), Mapping):
        raise ValueError(f"invalid dataset source manifest: {path}")
    return dict(payload)


def dataset_names(value: str, manifest: Mapping[str, Any]) -> list[str]:
    available = dict(manifest.get("datasets") or {})
    requested = [item.strip() for item in str(value).split(",") if item.strip()]
    if requested == ["all"]:
        return list(available)
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")
    return requested


def download_file(
    record: Mapping[str, Any],
    *,
    output_root: str | Path,
    force: bool = False,
    timeout: float = 120.0,
) -> dict[str, Any]:
    filename = str(record.get("filename") or "").strip()
    url = str(record.get("url") or "").strip()
    expected = str(record.get("sha256") or "").strip().lower()
    if not filename or not url or len(expected) != 64:
        raise ValueError(f"incomplete downloadable file record: {record}")

    output = Path(output_root) / filename
    if output.exists() and not force:
        actual = sha256_file(output)
        if actual == expected:
            return {"filename": filename, "path": str(output), "status": "verified", "sha256": actual}
        raise FileExistsError(
            f"existing file checksum mismatch for {output}; use --force to replace it"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "EvLink-dataset-downloader/0.1"})
    with NamedTemporaryFile(prefix=f".{filename}.", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    actual = sha256_file(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"checksum mismatch for {filename}: expected {expected}, got {actual}"
        )
    temporary.replace(output)
    return {"filename": filename, "path": str(output), "status": "downloaded", "sha256": actual}


def download_datasets(
    *,
    datasets: Sequence[str],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    force: bool = False,
    timeout: float = 120.0,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    specs = dict(manifest.get("datasets") or {})
    results: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    for name in datasets:
        spec = dict(specs[name] or {})
        files = [dict(item) for item in list(spec.get("files") or []) if isinstance(item, Mapping)]
        if not files or any(not str(item.get("url") or "").strip() for item in files):
            manual.append({"dataset": name, "status": "manual_source_required", "notes": spec.get("notes", "")})
            continue
        for item in files:
            results.append(
                {
                    "dataset": name,
                    **download_file(item, output_root=output_root, force=force, timeout=timeout),
                }
            )
    return {
        "output_root": str(output_root),
        "downloaded_or_verified": results,
        "manual_sources": manual,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="Dataset name, comma-separated names, or all.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--list", action="store_true", help="Print source metadata without downloading.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    names = dataset_names(args.dataset, manifest)
    if args.list:
        print(json.dumps({name: manifest["datasets"][name] for name in names}, indent=2, sort_keys=True))
        return 0
    payload = download_datasets(
        datasets=names,
        output_root=args.output_root,
        manifest_path=args.manifest,
        force=args.force,
        timeout=args.timeout,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
