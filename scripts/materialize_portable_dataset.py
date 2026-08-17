#!/usr/bin/env python3
"""Export generated MS-SWIFT JSONL files and referenced media as a portable bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


MEDIA_FIELDS = (
    "images",
    "videos",
    "audios",
    "rejected_images",
    "rejected_videos",
    "rejected_audios",
)


@dataclass
class MediaEntry:
    source_path: Path
    bundle_path: Path
    size: int
    references: int = 0
    sha256: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def resolve_source_path(value: str, jsonl_path: Path, source_root: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [jsonl_path.parent / path, source_root / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Media path {value!r} referenced by {jsonl_path} does not exist"
    )


def source_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def safe_basename(path: Path) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in path.name
    )
    return cleaned[-120:] or "media.bin"


def allocate_bundle_path(source: Path) -> Path:
    digest = hashlib.sha256(source_key(source).encode("utf-8")).hexdigest()
    return Path("media") / "files" / digest[:2] / f"{digest[:20]}_{safe_basename(source)}"


def rewrite_nested_media(value: Any, rewrite_path: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return rewrite_path(value)
    if isinstance(value, list):
        return [rewrite_nested_media(item, rewrite_path) for item in value]
    raise TypeError(f"Unsupported media value: {type(value).__name__}")


def collect_and_rewrite(
    source_root: Path,
    jsonl_path: Path,
    rows: list[dict[str, Any]],
    media_entries: dict[str, MediaEntry],
) -> list[dict[str, Any]]:
    rewritten_rows: list[dict[str, Any]] = []
    for row in rows:
        rewritten = dict(row)

        def rewrite_one(value: str) -> str:
            source = resolve_source_path(value, jsonl_path, source_root)
            key = source_key(source)
            entry = media_entries.get(key)
            if entry is None:
                entry = MediaEntry(
                    source_path=source,
                    bundle_path=allocate_bundle_path(source),
                    size=source.stat().st_size,
                )
                media_entries[key] = entry
            entry.references += 1
            return entry.bundle_path.as_posix()

        for field in MEDIA_FIELDS:
            if field in rewritten:
                rewritten[field] = rewrite_nested_media(rewritten[field], rewrite_one)
        rewritten_rows.append(rewritten)
    return rewritten_rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_media(
    bundle_root: Path,
    entries: Iterable[MediaEntry],
    *,
    mode: str,
    checksum: bool,
    repair: bool,
) -> None:
    for index, entry in enumerate(entries, start=1):
        destination = bundle_root / entry.bundle_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == entry.size:
            pass
        elif destination.exists() and not repair:
            raise FileExistsError(
                f"Existing bundle file differs from source: {destination}. Use --repair to replace it."
            )
        elif mode == "copy":
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(entry.source_path, temporary)
            temporary.replace(destination)
        else:
            if destination.exists():
                destination.unlink()
            os.link(entry.source_path, destination)
        if checksum:
            entry.sha256 = file_sha256(destination)
        if index % 1000 == 0:
            print(f"materialized media: {index}")


def write_manifest(path: Path, entries: Iterable[MediaEntry]) -> None:
    rows = (
        {
            "source_path": str(entry.source_path),
            "bundle_path": entry.bundle_path.as_posix(),
            "size": entry.size,
            "references": entry.references,
            "sha256": entry.sha256,
        }
        for entry in entries
    )
    write_jsonl(path, rows)


def write_bundle_metadata(
    path: Path,
    *,
    source_root: Path,
    source_data: Path,
    jsonl_files: int,
    rows: int,
    media_entries: dict[str, MediaEntry],
) -> None:
    metadata = {
        "format": "portable-ms-swift-dataset-v1",
        "source_root": str(source_root),
        "source_data": str(source_data),
        "jsonl_files": jsonl_files,
        "rows_across_files": rows,
        "unique_media_files": len(media_entries),
        "media_bytes": sum(entry.size for entry in media_entries.values()),
        "canonical_data_directory": "data",
        "runtime_data_directory": "runtime_data",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(path: Path) -> None:
    content = """# Portable MS-SWIFT Dataset

This directory contains the generated JSONL files and only the media referenced by them.

## Layout

- `data/`: canonical JSONL files with bundle-relative media paths.
- `media/files/`: deduplicated images and sampled video frames.
- `metadata/`: bundle statistics and source-to-bundle file mapping.
- `scripts/resolve_paths.py`: creates runtime JSONL files with absolute paths.
- `runtime_data/`: generated after running the resolver on the destination machine.

## After migration

Run this command from any directory:

```powershell
python scripts/resolve_paths.py --bundle-root .
```

Then pass `runtime_data/train.jsonl` and `runtime_data/val.jsonl` to MS-SWIFT.
Do not train directly from `data/*.jsonl` unless the current working directory is the bundle root
and the installed MS-SWIFT version is known to resolve relative media paths accordingly.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("dataset/ms_swift/full"),
        help="Generated dataset directory containing train.jsonl and val.jsonl.",
    )
    parser.add_argument("--output", type=Path, default=Path("portable_dataset"))
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use copy for migration. Hardlink is intended only for fast local validation.",
    )
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Replace an existing bundle media file when its size differs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_data = args.source_data if args.source_data.is_absolute() else root / args.source_data
    bundle_root = args.output if args.output.is_absolute() else root / args.output
    source_data = source_data.resolve()
    bundle_root = bundle_root.resolve()

    jsonl_paths = sorted(source_data.rglob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found under {source_data}")
    media_entries: dict[str, MediaEntry] = {}
    rewritten_files: dict[Path, list[dict[str, Any]]] = {}
    row_count = 0

    for jsonl_path in jsonl_paths:
        rows = read_jsonl(jsonl_path)
        rewritten_files[jsonl_path.relative_to(source_data)] = collect_and_rewrite(
            root, jsonl_path, rows, media_entries
        )
        row_count += len(rows)

    media_bytes = sum(entry.size for entry in media_entries.values())
    print(
        json.dumps(
            {
                "jsonl_files": len(jsonl_paths),
                "rows_across_files": row_count,
                "unique_media_files": len(media_entries),
                "media_gib": round(media_bytes / 1024**3, 2),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    bundle_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(bundle_root).free
    missing_bytes = sum(
        entry.size
        for entry in media_entries.values()
        if not (bundle_root / entry.bundle_path).is_file()
    )
    if args.mode == "copy" and free_bytes < missing_bytes:
        raise OSError(
            f"Insufficient free space: need {missing_bytes / 1024**3:.2f} GiB, "
            f"have {free_bytes / 1024**3:.2f} GiB"
        )

    sorted_entries = sorted(media_entries.values(), key=lambda item: item.bundle_path.as_posix())
    materialize_media(
        bundle_root,
        sorted_entries,
        mode=args.mode,
        checksum=args.checksum,
        repair=args.repair,
    )
    for relative_path, rows in rewritten_files.items():
        write_jsonl(bundle_root / "data" / relative_path, rows)
    write_manifest(bundle_root / "metadata" / "media_manifest.jsonl", sorted_entries)
    write_bundle_metadata(
        bundle_root / "metadata" / "bundle.json",
        source_root=root,
        source_data=source_data,
        jsonl_files=len(jsonl_paths),
        rows=row_count,
        media_entries=media_entries,
    )

    resolver_source = Path(__file__).with_name("resolve_portable_dataset.py")
    resolver_destination = bundle_root / "scripts" / "resolve_paths.py"
    resolver_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolver_source, resolver_destination)
    write_readme(bundle_root / "README.md")
    print(f"portable bundle: {bundle_root}")


if __name__ == "__main__":
    main()
