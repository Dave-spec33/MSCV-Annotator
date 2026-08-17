#!/usr/bin/env python3
"""Resolve bundle-relative media paths to absolute paths after migration."""

from __future__ import annotations

import argparse
import json
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rewrite_nested(value: Any, rewrite_path: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return rewrite_path(value)
    if isinstance(value, list):
        return [rewrite_nested(item, rewrite_path) for item in value]
    raise TypeError(f"Unsupported media value: {type(value).__name__}")


def resolve_rows(rows: list[dict[str, Any]], bundle_root: Path) -> list[dict[str, Any]]:
    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        resolved = dict(row)

        def resolve_one(value: str) -> str:
            path = Path(value)
            target = path if path.is_absolute() else bundle_root / path
            target = target.resolve()
            if not target.is_file():
                raise FileNotFoundError(f"Bundle media file does not exist: {target}")
            return str(target)

        for field in MEDIA_FIELDS:
            if field in resolved:
                resolved[field] = rewrite_nested(resolved[field], resolve_one)
        resolved_rows.append(resolved)
    return resolved_rows


def parse_args() -> argparse.Namespace:
    default_bundle = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=default_bundle)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime_data"))
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    data_root = args.data_dir if args.data_dir.is_absolute() else bundle_root / args.data_dir
    output_root = (
        args.output_dir if args.output_dir.is_absolute() else bundle_root / args.output_dir
    )
    jsonl_paths = sorted(data_root.rglob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found under {data_root}")

    row_count = 0
    for jsonl_path in jsonl_paths:
        rows = read_jsonl(jsonl_path)
        resolved = resolve_rows(rows, bundle_root)
        row_count += len(resolved)
        if not args.check_only:
            write_jsonl(output_root / jsonl_path.relative_to(data_root), resolved)
    print(
        json.dumps(
            {
                "bundle_root": str(bundle_root),
                "jsonl_files": len(jsonl_paths),
                "rows_across_files": row_count,
                "output": None if args.check_only else str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
