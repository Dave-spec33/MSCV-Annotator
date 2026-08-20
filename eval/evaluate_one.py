#!/usr/bin/env python3
"""Evaluate one prediction JSONL with BERTScore and output-format checks."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

PREDICTION_KEYS = (
    "response",
    "predict",
    "prediction",
    "generated_text",
    "output",
    "answer",
)

FORBIDDEN_PATTERNS = (
    r"\bbbox\b",
    r"\bbounding boxes?\b",
    r"\bcoordinates?\b",
    r"\bmasks?\b",
    r"\btrack[_ ]?ids?\b",
    r"\bsegmentation mask\b",
    (
        r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?"
        r"(?:\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?)?\s*\)"
    ),
)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_message(row: dict[str, Any], role: str) -> str:
    for message in reversed(row.get("messages", [])):
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def extract_reference(row: dict[str, Any]) -> str:
    text = get_message(row, "assistant")
    if text:
        return text
    for key in ("reference", "label", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Cannot find reference text")


def extract_prediction(row: dict[str, Any], field: str) -> str:
    if field != "auto":
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Prediction field {field!r} not found")
        return value.strip()
    for key in PREDICTION_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = get_message(row, "assistant")
    if text:
        return text
    raise ValueError("Cannot find prediction text")


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    parts = [part.strip() for part in re.split(r"[.!?]+(?:\s+|$)", text) if part.strip()]
    return max(1, len(parts))


def check_format(text: str, max_words: int, max_sentences: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    text = text.strip()
    if not text:
        reasons.append("empty")
    words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", text)
    if len(words) > max_words:
        reasons.append("too_long")
    if count_sentences(text) > max_sentences:
        reasons.append("too_many_sentences")
    if re.search(r"[\u3400-\u9fff]", text):
        reasons.append("non_english")
    if text.startswith(("{", "[")):
        reasons.append("json_like")
    if any(re.search(pattern, text, flags=re.I) for pattern in FORBIDDEN_PATTERNS):
        reasons.append("forbidden_annotation_output")
    return not reasons, reasons


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=project_root / "portable_dataset" / "runtime_data" / "val.jsonl",
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--bert-model",
        default=os.getenv("BERT_MODEL_PATH", "roberta-large"),
        help="Local RoBERTa-large path or Hugging Face model ID.",
    )
    parser.add_argument("--bert-num-layers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prediction-field", default="auto")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from bert_score import BERTScorer

    reference_rows = read_jsonl(args.reference.resolve())
    prediction_rows = read_jsonl(args.prediction.resolve())
    if len(reference_rows) != len(prediction_rows):
        raise ValueError(
            f"Sample count mismatch: reference={len(reference_rows)}, "
            f"prediction={len(prediction_rows)}"
        )
    for index, row in enumerate(prediction_rows):
        if "index" in row and int(row["index"]) != index:
            raise ValueError(f"Prediction order mismatch at row {index}: {row['index']}")

    references = [extract_reference(row) for row in reference_rows]
    predictions = [
        extract_prediction(row, args.prediction_field) for row in prediction_rows
    ]
    scorer = BERTScorer(
        model_type=args.bert_model,
        num_layers=args.bert_num_layers,
        lang="en",
        idf=False,
        rescale_with_baseline=False,
        device=args.device,
    )
    precision, recall, f1 = scorer.score(
        predictions,
        references,
        batch_size=args.batch_size,
    )

    records: list[dict[str, Any]] = []
    success_count = 0
    for index, (reference, prediction) in enumerate(zip(references, predictions)):
        success, reasons = check_format(
            prediction,
            max_words=args.max_words,
            max_sentences=args.max_sentences,
        )
        success_count += int(success)
        records.append(
            {
                "index": index,
                "reference": reference,
                "prediction": prediction,
                "bertscore_p": float(precision[index].item()),
                "bertscore_r": float(recall[index].item()),
                "bertscore_f1": float(f1[index].item()),
                "format_success": success,
                "format_fail_reasons": reasons,
            }
        )

    model_name = args.model_name or args.prediction.stem
    summary = {
        "model": model_name,
        "num_samples": len(records),
        "bertscore_p": float(precision.mean().item()),
        "bertscore_r": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
        "format_success_rate": success_count / len(records) if records else 0.0,
        "format_success_count": success_count,
        "format_failure_count": len(records) - success_count,
        "metric_config": {
            "bert_model": args.bert_model,
            "bert_num_layers": args.bert_num_layers,
            "idf": False,
            "rescale_with_baseline": False,
            "max_words": args.max_words,
            "max_sentences": args.max_sentences,
        },
    }

    output_dir = args.output_dir.resolve()
    write_jsonl(output_dir / f"{model_name}_per_sample.jsonl", records)
    summary_path = output_dir / f"{model_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print(f"BERTScore-P  : {summary['bertscore_p']:.4f}")
    print(f"BERTScore-R  : {summary['bertscore_r']:.4f}")
    print(f"BERTScore-F1 : {summary['bertscore_f1']:.4f}")
    print(
        f"Format Success Rate: {summary['format_success_rate']:.4f} "
        f"({success_count}/{len(records)})"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
