#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


MEDIA_FIELDS = ("images", "videos", "audios")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {path}:{line_no}: {e}") from e


def flatten_media(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten_media(item)
    else:
        raise TypeError(f"Unsupported media value type: {type(value)}")


def verify_image(path: Path):
    with Image.open(path) as img:
        img.verify()


def check_row(row: dict, path: Path, line_no: int, verify_images: bool):
    errors = []

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("missing or invalid messages")
    else:
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"] and roles != ["user", "assistant"]:
            errors.append(f"unexpected message roles: {roles}")

        user_text = ""
        assistant_text = ""
        for message in messages:
            if message.get("role") == "user":
                user_text += str(message.get("content", ""))
            if message.get("role") == "assistant":
                assistant_text += str(message.get("content", ""))

        if not assistant_text.strip():
            errors.append("empty assistant response")

        image_count = len(row.get("images", []))
        video_count = len(row.get("videos", []))
        if user_text.count("<image>") != image_count:
            errors.append(
                f"image placeholder mismatch: placeholders={user_text.count('<image>')}, "
                f"images={image_count}"
            )
        if user_text.count("<video>") != video_count:
            errors.append(
                f"video placeholder mismatch: placeholders={user_text.count('<video>')}, "
                f"videos={video_count}"
            )

    for field in MEDIA_FIELDS:
        if field not in row:
            continue
        for value in flatten_media(row[field]):
            media_path = Path(value)
            if not media_path.is_file():
                errors.append(f"missing media file: {media_path}")
                continue
            if verify_images and media_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                try:
                    verify_image(media_path)
                except Exception as e:
                    errors.append(f"corrupt image: {media_path}: {e}")

    if errors:
        raise ValueError(f"{path}:{line_no}: " + " | ".join(errors))


def check_jsonl(path: Path, verify_images: bool, max_errors: int):
    rows = 0
    media_counter = Counter()
    errors = []

    for line_no, row in read_jsonl(path):
        rows += 1
        for field in MEDIA_FIELDS:
            if field in row:
                media_counter[field] += 1
        try:
            check_row(row, path, line_no, verify_images)
        except Exception as e:
            errors.append(str(e))
            if len(errors) >= max_errors:
                break

    return {
        "file": str(path),
        "rows": rows,
        "media_rows": dict(media_counter),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("runtime_data"))
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    data_root = args.data_dir if args.data_dir.is_absolute() else bundle_root / args.data_dir

    train = data_root / "train.jsonl"
    val = data_root / "val.jsonl"

    if not train.is_file():
        raise FileNotFoundError(f"Missing train file: {train}")
    if not val.is_file():
        raise FileNotFoundError(f"Missing val file: {val}")

    reports = [
        check_jsonl(train, args.verify_images, args.max_errors),
        check_jsonl(val, args.verify_images, args.max_errors),
    ]

    total_errors = sum(len(report["errors"]) for report in reports)
    print(
        json.dumps(
            {
                "bundle_root": str(bundle_root),
                "data_root": str(data_root),
                "reports": reports,
                "total_errors": total_errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
