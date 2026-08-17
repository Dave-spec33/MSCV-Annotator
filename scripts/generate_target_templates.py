#!/usr/bin/env python3
"""Generate GT-initialized target templates for SOREC, LaSOT, and TNL2K."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_directory(project_root: Path, name: str) -> Path:
    organized = project_root / "cv_datasets" / name
    legacy = project_root / name
    if organized.is_dir():
        return organized
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        f"Dataset directory not found: expected {organized} or {legacy}"
    )


@dataclass
class TemplateRecord:
    sample_id: str
    source: str
    sequence: str | None
    source_index: int | None
    source_image: str
    template_image: str
    bbox_xywh: list[float]
    crop_xyxy: list[int]
    original_size: list[int]
    context_scale: float
    output_size: int
    source_text: str | None = None


def natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def parse_bbox(line: str) -> tuple[float, float, float, float]:
    values = [float(v) for v in re.split(r"[,\s\t]+", line.strip()) if v]
    if len(values) == 4:
        x, y, width, height = values
    elif len(values) == 8:
        xs = values[0::2]
        ys = values[1::2]
        x, y = min(xs), min(ys)
        width, height = max(xs) - x, max(ys) - y
    else:
        raise ValueError(f"Expected 4 or 8 bbox values, got {len(values)}: {line!r}")
    if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox: {line!r}")
    return x, y, width, height


def square_crop_box(
    image_size: tuple[int, int], bbox: Sequence[float], context_scale: float
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2
    side = max(2, min(max(width, height) * context_scale, image_width, image_height))

    left = center_x - side / 2
    top = center_y - side / 2
    left = min(max(left, 0), image_width - side)
    top = min(max(top, 0), image_height - side)
    right = left + side
    bottom = top + side

    return (
        max(0, int(math.floor(left))),
        max(0, int(math.floor(top))),
        min(image_width, int(math.ceil(right))),
        min(image_height, int(math.ceil(bottom))),
    )


def render_template(
    source_image: Path,
    output_image: Path,
    bbox: Sequence[float],
    context_scale: float,
    output_size: int,
) -> tuple[list[int], list[int]]:
    with Image.open(source_image) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        x, y, width, height = bbox
        if x + width <= 0 or y + height <= 0 or x >= image.width or y >= image.height:
            raise ValueError(
                f"BBox {list(bbox)} does not intersect image {source_image} with size {image.size}"
            )
        crop_box = square_crop_box(image.size, bbox, context_scale)
        template = image.crop(crop_box)
        template = template.resize((output_size, output_size), Image.Resampling.LANCZOS)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        template.save(output_image, format="JPEG", quality=95, subsampling=0, optimize=True)
        return list(crop_box), list(image.size)


def first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        line = handle.readline()
    if not line.strip():
        raise ValueError(f"Empty file: {path}")
    return line.strip()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )


def relative_string(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def process_tracking_dataset(
    root: Path,
    output_root: Path,
    dataset_name: str,
    sequences_root: Path,
    frames_dir_name: str,
    context_scale: float,
    output_size: int,
) -> list[TemplateRecord]:
    records: list[TemplateRecord] = []
    for sequence_dir in sorted((p for p in sequences_root.iterdir() if p.is_dir()), key=natural_key):
        frames_dir = sequence_dir / frames_dir_name
        gt_path = sequence_dir / "groundtruth.txt"
        if not frames_dir.is_dir() or not gt_path.is_file():
            continue

        frames = image_files(frames_dir)
        if not frames:
            continue
        bbox = parse_bbox(first_line(gt_path))
        source_image = frames[0]
        sample_id = f"{dataset_name}_{sequence_dir.name}"
        output_image = output_root / dataset_name / f"{sequence_dir.name}.jpg"
        crop_box, original_size = render_template(
            source_image, output_image, bbox, context_scale, output_size
        )
        records.append(
            TemplateRecord(
                sample_id=sample_id,
                source=dataset_name,
                sequence=sequence_dir.name,
                source_index=None,
                source_image=relative_string(source_image, root),
                template_image=relative_string(output_image, root),
                bbox_xywh=list(bbox),
                crop_xyxy=crop_box,
                original_size=original_size,
                context_scale=context_scale,
                output_size=output_size,
                source_text=(first_line(sequence_dir / ("nlp.txt" if dataset_name == "lasot" else "language.txt"))),
            )
        )
    return records


def build_image_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name, path)
    return index


def select_sorec_rows(rows: list[dict], limit: int | None, seed: int) -> list[tuple[int, dict]]:
    indexed_rows = list(enumerate(rows))
    random.Random(seed).shuffle(indexed_rows)
    if limit is None or limit >= len(indexed_rows):
        return indexed_rows

    selected: list[tuple[int, dict]] = []
    used_images: set[str] = set()
    deferred: list[tuple[int, dict]] = []
    for item in indexed_rows:
        image_name = item[1]["image"]
        if image_name not in used_images:
            selected.append(item)
            used_images.add(image_name)
            if len(selected) == limit:
                return selected
        else:
            deferred.append(item)
    selected.extend(deferred[: limit - len(selected)])
    return selected


def process_sorec(
    root: Path,
    output_root: Path,
    annotations_path: Path,
    images_root: Path,
    context_scale: float,
    output_size: int,
    limit: int | None,
    seed: int,
) -> list[TemplateRecord]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    selected = select_sorec_rows(rows, limit, seed)
    image_index = build_image_index(images_root)

    records: list[TemplateRecord] = []
    missing: set[str] = set()
    for source_index, row in selected:
        image_name = row["image"]
        source_image = image_index.get(Path(image_name).name)
        if source_image is None:
            missing.add(image_name)
            continue
        bbox = tuple(float(v) for v in row["bbox"])
        digest = hashlib.sha1(
            f"{source_index}:{image_name}:{bbox}".encode("utf-8")
        ).hexdigest()[:10]
        sample_id = f"sorec_{source_index:06d}_{digest}"
        output_image = output_root / "sorec" / f"{sample_id}.jpg"
        crop_box, original_size = render_template(
            source_image, output_image, bbox, context_scale, output_size
        )
        records.append(
            TemplateRecord(
                sample_id=sample_id,
                source="sorec",
                sequence=None,
                source_index=source_index,
                source_image=relative_string(source_image, root),
                template_image=relative_string(output_image, root),
                bbox_xywh=list(bbox),
                crop_xyxy=crop_box,
                original_size=original_size,
                context_scale=context_scale,
                output_size=output_size,
                source_text=row["ref"].strip(),
            )
        )

    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} selected SOREC images under {images_root}. Examples: {preview}"
        )
    return records


def write_manifest(path: Path, records: Iterable[TemplateRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_sorec_extract_list(
    annotations_path: Path, output_path: Path, limit: int | None, seed: int
) -> int:
    with annotations_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    selected = select_sorec_rows(rows, limit, seed)
    image_names = sorted({row["image"] for _, row in selected})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for image_name in image_names:
            handle.write(f"Images\\{Path(image_name).name}\n")
    return len(image_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("dataset/assets/target_templates"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("sorec", "lasot", "tnl2k"),
        default=("sorec", "lasot", "tnl2k"),
    )
    parser.add_argument("--output-size", type=int, default=336)
    parser.add_argument("--sorec-split", default="trainL")
    parser.add_argument("--sorec-limit", type=int, default=6000)
    parser.add_argument("--write-sorec-extract-list", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output if args.output.is_absolute() else root / args.output
    sorec_root = (
        dataset_directory(root, "sorec")
        if "sorec" in args.datasets or args.write_sorec_extract_list is not None
        else None
    )

    if args.write_sorec_extract_list is not None:
        assert sorec_root is not None
        list_path = (
            args.write_sorec_extract_list
            if args.write_sorec_extract_list.is_absolute()
            else root / args.write_sorec_extract_list
        )
        count = write_sorec_extract_list(
            sorec_root / f"{args.sorec_split}.json",
            list_path,
            args.sorec_limit,
            args.seed,
        )
        print(json.dumps({"extract_list": str(list_path), "count": count}, indent=2))
        return

    all_records: list[TemplateRecord] = []

    if "lasot" in args.datasets:
        all_records.extend(
            process_tracking_dataset(
                root,
                output_root,
                "lasot",
                dataset_directory(root, "LaSOTTest"),
                "img",
                context_scale=2.0,
                output_size=args.output_size,
            )
        )
    if "tnl2k" in args.datasets:
        all_records.extend(
            process_tracking_dataset(
                root,
                output_root,
                "tnl2k",
                dataset_directory(root, "TNL2K") / "TNL2k_train_subset_p5",
                "imgs",
                context_scale=2.0,
                output_size=args.output_size,
            )
        )
    if "sorec" in args.datasets:
        assert sorec_root is not None
        all_records.extend(
            process_sorec(
                root,
                output_root,
                sorec_root / f"{args.sorec_split}.json",
                sorec_root / "SODA-D" / "Images" / "extracted",
                context_scale=3.0,
                output_size=args.output_size,
                limit=args.sorec_limit,
                seed=args.seed,
            )
        )

    manifest_path = root / "dataset" / "manifests" / "target_templates.jsonl"
    write_manifest(manifest_path, all_records)
    for source in sorted({record.source for record in all_records}):
        write_manifest(
            root / "dataset" / "manifests" / f"target_templates_{source}.jsonl",
            (record for record in all_records if record.source == source),
        )
    counts: dict[str, int] = {}
    for record in all_records:
        counts[record.source] = counts.get(record.source, 0) + 1
    print(json.dumps({"manifest": str(manifest_path), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
