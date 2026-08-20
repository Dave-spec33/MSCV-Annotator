#!/usr/bin/env python3
"""Run Qwen3-VL-Plus inference through the DashScope OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON: {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object: {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(flatten(item))
        return result
    raise TypeError(f"Unsupported media value: {type(value).__name__}")


def resolve_media_path(
    value: str,
    dataset_path: Path,
    media_root: Path | None,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if media_root is not None:
        return (media_root / path).resolve()
    return (dataset_path.parent / path).resolve()


def file_to_data_url(
    value: str,
    dataset_path: Path,
    media_root: Path | None,
) -> str:
    path = resolve_media_path(value, dataset_path, media_root)
    if not path.is_file():
        raise FileNotFoundError(f"Media not found: {path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_user_content(
    row: dict[str, Any],
    dataset_path: Path,
    media_root: Path | None,
) -> list[dict[str, Any]]:
    prompt = get_message(row, "user")
    images = flatten(row.get("images", []))
    videos = row.get("videos", [])
    pieces = re.split(r"(<image>|<video>)", prompt)
    image_index = 0
    video_index = 0
    content: list[dict[str, Any]] = []

    for piece in pieces:
        if not piece:
            continue
        if piece == "<image>":
            if image_index >= len(images):
                raise ValueError("Prompt has more <image> tokens than images")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": file_to_data_url(
                            images[image_index],
                            dataset_path,
                            media_root,
                        )
                    },
                }
            )
            image_index += 1
        elif piece == "<video>":
            if video_index >= len(videos):
                raise ValueError("Prompt has more <video> tokens than videos")
            frames = flatten(videos[video_index])
            content.append(
                {
                    "type": "video",
                    "video": [
                        file_to_data_url(frame, dataset_path, media_root)
                        for frame in frames
                    ],
                }
            )
            video_index += 1
        elif piece.strip():
            content.append({"type": "text", "text": piece.strip()})

    if image_index != len(images) or video_index != len(videos):
        raise ValueError("Media count does not match prompt placeholders")
    return content


def build_messages(
    row: dict[str, Any],
    dataset_path: Path,
    media_root: Path | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_prompt = get_message(row, "system")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": build_user_content(row, dataset_path, media_root),
        }
    )
    return messages


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "portable_dataset" / "runtime_data" / "val.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "qwen3vl_plus_api.jsonl",
    )
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=5.0)
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def call_model(
    client: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    dataset_path: Path,
    media_root: Path | None,
) -> tuple[str, dict[str, int | None] | None]:
    messages = build_messages(row, dataset_path, media_root)
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"enable_thinking": args.enable_thinking},
            )
            text = completion.choices[0].message.content
            if not isinstance(text, str):
                raise TypeError("API returned a non-text response")
            usage = completion.usage
            usage_dict = None
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return text.strip(), usage_dict
        except Exception as error:
            last_error = error
            print(f"[WARN] API attempt {attempt}/{args.max_retries} failed: {error}")
            if attempt < args.max_retries:
                time.sleep(args.retry_wait_seconds * attempt)
    raise RuntimeError(f"API failed after {args.max_retries} attempts: {last_error}")


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")

    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    media_root = args.media_root.resolve() if args.media_root else None
    rows = read_jsonl(dataset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    completed = 0
    if args.resume and output_path.is_file():
        previous = read_jsonl(output_path)
        for index, row in enumerate(previous):
            if int(row.get("index", -1)) != index:
                raise ValueError(f"Cannot resume: output index mismatch at row {index}")
        completed = len(previous)
    elif output_path.exists():
        output_path.unlink()
    if completed > len(rows):
        raise ValueError("Existing output has more rows than the requested dataset")
    if completed == len(rows):
        print("All samples already completed.")
        return

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    print(f"Model: {args.model}")
    print(f"Samples: {len(rows)}; starting at: {completed}")
    for index in range(completed, len(rows)):
        response, usage = call_model(
            client,
            rows[index],
            args,
            dataset_path,
            media_root,
        )
        result: dict[str, Any] = {
            "index": index,
            "response": response,
            "reference": get_message(rows[index], "assistant"),
            "model": args.model,
            "usage": usage,
        }
        for key in ("dataset", "source", "eval_meta"):
            if key in rows[index]:
                result[key] = rows[index][key]
        append_jsonl(output_path, result)
        print(f"[{index + 1}/{len(rows)}] {response[:160].replace(chr(10), ' ')}")
        if args.request_interval > 0:
            time.sleep(args.request_interval)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
