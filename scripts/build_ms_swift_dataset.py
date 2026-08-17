#!/usr/bin/env python3
"""Build the mixed Qwen-VL SFT dataset in MS-SWIFT JSONL format.

The script supports a lightweight demo mode that emits only a few samples per
dataset together with an HTML preview. Full mode follows the quotas agreed for
the first training version.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SYSTEM_PROMPT = (
    "You are a professional computer vision semantic annotation assistant. "
    "Given visual inputs and task-specific instructions, generate concise, "
    "visually grounded English descriptions for the relevant target object or "
    "target set. Use only task-relevant evidence, such as appearance, visual "
    "attributes, spatial or relational context, actions, and motion, and include "
    "disambiguating cues when supported. Do not output coordinates, bounding "
    "boxes, masks, or track IDs."
)


USER_PROMPTS = {
    "d3": (
        "<image>Write one concise and flexible natural-language expression that "
        "describes a visually identifiable object or coherent set of objects in "
        "the image. The expression may use object category, visual attributes, "
        "relationships, or the absence of an attribute when relevant."
    ),
    "sorec": (
        "<image><image>The first image shows the complete scene, and the second "
        "image shows the designated small target. Write one sentence that refers "
        "to the target by describing its visual details, color, and relative "
        "relationship to surrounding objects so that it can be uniquely identified."
    ),
    "lasot": (
        "<image><video>The image shows the target initialized in the first frame, "
        "and the video frames show its appearance over time. Write one sentence "
        "describing the target's color, behavior, and surroundings to provide "
        "global semantic guidance for tracking."
    ),
    "tnl2k": (
        "<image><image>The first image shows the complete first-frame scene, and "
        "the second image shows the designated target. Write one sentence "
        "describing the target's category, visual attributes, properties, spatial "
        "position, and relative location to other objects in the first frame."
    ),
    "refer_kitti": (
        "<video>Write one concise referring expression for a visually coherent set "
        "of road users in the video. Use category, color, spatial position, motion "
        "state, relative speed, or an implicit relationship when relevant."
    ),
    "lamot": (
        "<video>Write one concise natural-language description for a visually "
        "identifiable target or target group in the video. Use appearance, "
        "position, or action cues that remain valid for most of the sequence."
    ),
    "mevis": (
        "<video>Write one concise motion expression that identifies one or more "
        "target objects through their actions or movement over time. Prefer motion "
        "cues over static appearance, and omit static attributes when motion alone "
        "is sufficient."
    ),
}


FULL_QUOTAS = {
    "d3": 9000,
    "sorec": 6000,
    "refer_kitti": 2400,
    "lamot": 300,
    "lasot": 100,
    "tnl2k": 100,
    "mevis": 1900,
}

DATASET_ORDER = tuple(FULL_QUOTAS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# This can be changed directly, or overridden with --lasot-scales.
DEFAULT_LASOT_SCALES = ("long",)
#DEFAULT_LASOT_SCALES = ("wide", "long", "full")
LASOT_SCALE_RANGES = {
    "wide": (0.10, 0.25),
    "long": (0.35, 0.65),
    "full": (0.75, 1.00),
}
LASOT_SCALE_ANCHORS = {
    "wide": 0.15,
    "long": 0.50,
    "full": 1.00,
}


@dataclass
class BuiltSample:
    dataset: str
    sample_id: str
    split: str
    row: dict[str, Any]
    metadata: dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def resolve_project_media(project_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() and relative.is_file():
        return relative
    candidates = (project_root / relative, project_root / "cv_datasets" / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Media path not found: {relative_path}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_for_group(group: str, seed: int, val_ratio: float = 0.1) -> str:
    return "val" if stable_fraction(group, seed) < val_ratio else "train"


def make_row(
    dataset: str,
    response: str,
    *,
    images: Sequence[Path] = (),
    video_frames: Sequence[Path] = (),
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPTS[dataset]},
            {"role": "assistant", "content": response.strip()},
        ]
    }
    if images:
        row["images"] = [str(path.resolve()) for path in images]
    if video_frames:
        row["videos"] = [[str(path.resolve()) for path in video_frames]]
    return row


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def list_images(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_key,
    )


def evenly_sample(values: Sequence[Any], count: int) -> list[Any]:
    if not values:
        raise ValueError("Cannot sample an empty sequence")
    if len(values) <= count:
        return list(values)
    positions = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[position] for position in positions]


def shuffled(items: Iterable[Any], seed: int) -> list[Any]:
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


class ZipFrameStore:
    def __init__(self, archive: Path, output_root: Path):
        self.archive = archive
        self.output_root = output_root
        self._zip: zipfile.ZipFile | None = None
        self._names: dict[str, str] | None = None

    def __enter__(self) -> "ZipFrameStore":
        self._zip = zipfile.ZipFile(self.archive)
        self._names = {name.replace("\\", "/"): name for name in self._zip.namelist()}
        return self

    def __exit__(self, *_: object) -> None:
        if self._zip is not None:
            self._zip.close()

    def materialize(self, member: str, relative_output: Path) -> Path:
        assert self._zip is not None and self._names is not None
        normalized = member.replace("\\", "/")
        archive_name = self._names.get(normalized)
        if archive_name is None:
            raise FileNotFoundError(f"Archive member not found: {member}")
        output = self.output_root / relative_output
        if not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            with self._zip.open(archive_name) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
        return output


class TarFrameStore:
    def __init__(self, archive: Path, output_root: Path):
        self.archive = archive
        self.output_root = output_root
        self._tar: tarfile.TarFile | None = None
        self._names: set[str] | None = None

    def __enter__(self) -> "TarFrameStore":
        self._tar = tarfile.open(self.archive, "r")
        self._names = {member.name for member in self._tar.getmembers() if member.isfile()}
        return self

    def __exit__(self, *_: object) -> None:
        if self._tar is not None:
            self._tar.close()

    def materialize(self, member: str, relative_output: Path) -> Path:
        assert self._tar is not None and self._names is not None
        if member not in self._names:
            raise FileNotFoundError(f"Archive member not found: {member}")
        output = self.output_root / relative_output
        if not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            source = self._tar.extractfile(member)
            if source is None:
                raise FileNotFoundError(f"Cannot read archive member: {member}")
            with source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
        return output


def build_d3(root: Path, limit: int, seed: int, _: Path) -> list[BuiltSample]:
    dataset_root = dataset_directory(root, "d3")
    annotation = load_json(dataset_root / "d3_json" / "d3_full_annotations.json")
    images = {item["id"]: item for item in annotation["images"]}
    categories = {item["id"]: item["name"] for item in annotation["categories"]}
    positive_pairs = sorted(
        {(item["image_id"], item["category_id"]) for item in annotation["annotations"]}
    )
    candidates = shuffled(positive_pairs, seed + 11)[:limit]
    samples: list[BuiltSample] = []
    for image_id, category_id in candidates:
        source_image = dataset_root / "d3_images" / images[image_id]["file_name"]
        if not source_image.is_file():
            continue
        sample_id = f"d3_{image_id}_{category_id}"
        samples.append(
            BuiltSample(
                dataset="d3",
                sample_id=sample_id,
                split=split_for_group(str(image_id), seed, val_ratio=0.05),
                row=make_row("d3", categories[category_id], images=[source_image]),
                metadata={"image_id": image_id, "category_id": category_id},
            )
        )
    return samples


def build_from_template_manifest(
    root: Path, dataset: str, limit: int, seed: int
) -> list[BuiltSample]:
    manifest = read_jsonl(
        root / "dataset" / "manifests" / f"target_templates_{dataset}.jsonl"
    )
    records = shuffled(manifest, seed + (23 if dataset == "sorec" else 29))[:limit]
    samples: list[BuiltSample] = []
    for record in records:
        source_image = resolve_project_media(root, record["source_image"])
        template_image = root / record["template_image"]
        group = record["sequence"] or record["source_image"]
        samples.append(
            BuiltSample(
                dataset=dataset,
                sample_id=record["sample_id"],
                split=split_for_group(str(group), seed),
                row=make_row(
                    dataset,
                    record["source_text"],
                    images=[source_image, template_image],
                ),
                metadata={
                    "sequence": record["sequence"],
                    "source_index": record["source_index"],
                    "bbox_xywh": record["bbox_xywh"],
                },
            )
        )
    return samples


def evenly_spaced_floats(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [(start + end) / 2]
    return [start + index * (end - start) / (count - 1) for index in range(count)]


def lasot_frame_groups(
    frames: Sequence[Path], selected_scales: Sequence[str], variants_per_scale: int
) -> list[tuple[str, int, list[Path]]]:
    last = len(frames) - 1
    groups: list[tuple[str, int, list[Path]]] = []
    seen: set[tuple[str, ...]] = set()
    for scale_name in selected_scales:
        ratio_start, ratio_end = LASOT_SCALE_RANGES[scale_name]
        ratios = (
            [LASOT_SCALE_ANCHORS[scale_name]]
            if variants_per_scale == 1
            else evenly_spaced_floats(ratio_start, ratio_end, variants_per_scale)
        )
        for variant, endpoint_ratio in enumerate(ratios):
            endpoint = min(last, max(4, round(last * endpoint_ratio)))
            group = evenly_sample(frames[: endpoint + 1], 5)
            key = tuple(path.name for path in group)
            if key not in seen:
                groups.append((scale_name, variant, group))
                seen.add(key)
    return groups


def build_lasot(
    root: Path,
    limit: int,
    seed: int,
    _: Path,
    selected_scales: Sequence[str] = DEFAULT_LASOT_SCALES,
) -> list[BuiltSample]:
    manifest = read_jsonl(root / "dataset" / "manifests" / "target_templates_lasot.jsonl")
    if not selected_scales:
        raise ValueError("At least one LaSOT temporal scale must be selected")
    variants_per_scale = max(
        1,
        (limit + len(manifest) * len(selected_scales) - 1)
        // (len(manifest) * len(selected_scales)),
    )
    candidates: list[tuple[dict[str, Any], list[Path], str, int]] = []
    for record in manifest:
        sequence_dir = dataset_directory(root, "LaSOTTest") / record["sequence"]
        frames = list_images(sequence_dir / "img")
        for scale_name, variant, frame_group in lasot_frame_groups(
            frames, selected_scales, variants_per_scale
        ):
            candidates.append((record, frame_group, scale_name, variant))
    candidates = shuffled(candidates, seed + 31)[:limit]

    samples: list[BuiltSample] = []
    for record, frame_group, scale_name, variant in candidates:
        template = root / record["template_image"]
        sequence = record["sequence"]
        samples.append(
            BuiltSample(
                dataset="lasot",
                sample_id=f"lasot_{sequence}_{scale_name}_v{variant}",
                split=split_for_group(sequence, seed),
                row=make_row(
                    "lasot",
                    record["source_text"],
                    images=[template],
                    video_frames=frame_group,
                ),
                metadata={
                    "sequence": sequence,
                    "temporal_scale": scale_name,
                    "scale_variant": variant,
                    "sampled_frames": [path.name for path in frame_group],
                },
            )
        )
    return samples


def round_robin_refer_candidates(
    expression_files: Sequence[Path], limit: int, seed: int
) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    for path in expression_files:
        data = load_json(path)
        raw = data.get("raw_sentence") or data["sentence"]
        groups.setdefault(raw.strip().lower(), []).append(path)
    rng = random.Random(seed + 37)
    group_items = list(groups.values())
    rng.shuffle(group_items)
    for items in group_items:
        rng.shuffle(items)

    selected: list[Path] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for items in group_items:
            if depth < len(items):
                selected.append(items[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def build_refer_kitti(
    root: Path, limit: int, seed: int, materialized_root: Path
) -> list[BuiltSample]:
    dataset_root = dataset_directory(root, "refer-kitti-v2")
    expression_root = dataset_root / "expression"
    expression_files = sorted(expression_root.glob("*/*.json"))
    # Some expressions contain no active target frame. Iterate through the full
    # diversity-preserving order so skipped records can be backfilled.
    selected = round_robin_refer_candidates(expression_files, len(expression_files), seed)
    archive = dataset_root / "kitti" / "data_tracking_image_2.zip"

    samples: list[BuiltSample] = []
    with ZipFrameStore(archive, materialized_root) as store:
        for path in selected:
            if len(samples) >= limit:
                break
            data = load_json(path)
            sequence = path.parent.name
            active_frames = sorted(
                int(frame_id)
                for frame_id, track_ids in data["label"].items()
                if track_ids
            )
            if not active_frames:
                continue
            sampled_ids = evenly_sample(active_frames, 6)
            frame_paths = [
                store.materialize(
                    f"training/image_02/{sequence}/{frame_id:06d}.png",
                    Path("refer_kitti") / sequence / f"{frame_id:06d}.png",
                )
                for frame_id in sampled_ids
            ]
            sample_id = f"refer_kitti_{sequence}_{path.stem}"
            samples.append(
                BuiltSample(
                    dataset="refer_kitti",
                    sample_id=sample_id,
                    split=split_for_group(sequence, seed),
                    row=make_row(
                        "refer_kitti", data["sentence"], video_frames=frame_paths
                    ),
                    metadata={
                        "sequence": sequence,
                        "expression_file": str(path.relative_to(root)),
                        "raw_sentence": data.get("raw_sentence"),
                        "sampled_frames": sampled_ids,
                    },
                )
            )
    return samples


def lamot_sequence_name(annotation_path: Path) -> str:
    return annotation_path.stem.rsplit("_", 1)[0]


def locate_mot17_sequence(root: Path, sequence: str) -> Path | None:
    dataset_root = dataset_directory(root, "LaMOT-main")
    for split in ("train", "test"):
        candidate = dataset_root / "MOT17" / split / sequence / "img1"
        if candidate.is_dir():
            return candidate
    return None


def lamot_frame_variants(
    active_ids: Sequence[int], count: int, seed: int
) -> list[list[int]]:
    if len(active_ids) <= 6:
        return [list(active_ids)]
    rng = random.Random(seed)
    variants = [evenly_sample(active_ids, 6)]
    spans = (24, 48, 96, 192)
    attempts = 0
    while len(variants) < count and attempts < count * 10:
        span = spans[attempts % len(spans)]
        start = rng.randrange(0, max(1, len(active_ids) - min(span, len(active_ids)) + 1))
        window = active_ids[start : start + span]
        candidate = evenly_sample(window, 6)
        if candidate not in variants:
            variants.append(candidate)
        attempts += 1
    return variants


def build_lamot(root: Path, limit: int, seed: int, _: Path) -> list[BuiltSample]:
    candidates: list[tuple[Path, dict[str, Any], list[int], str]] = []
    annotation_root = dataset_directory(root, "LaMOT-main") / "annotations_v1"
    annotation_files = sorted(annotation_root.glob("*/MOT17/*.json"))
    variants_per_annotation = max(1, (limit + len(annotation_files) - 1) // len(annotation_files))
    for annotation_path in annotation_files:
        sequence = lamot_sequence_name(annotation_path)
        frames_dir = locate_mot17_sequence(root, sequence)
        if frames_dir is None:
            continue
        data = load_json(annotation_path)
        active_ids = sorted(
            int(frame_id)
            for frame_id, track_ids in data["targets"].items()
            if track_ids
        )
        for sampled_ids in lamot_frame_variants(
            active_ids,
            variants_per_annotation,
            seed + int(hashlib.sha1(annotation_path.name.encode()).hexdigest()[:8], 16),
        ):
            candidates.append((annotation_path, data, sampled_ids, sequence))
    candidates = shuffled(candidates, seed + 41)[:limit]

    samples: list[BuiltSample] = []
    for index, (annotation_path, data, sampled_ids, sequence) in enumerate(candidates):
        frames_dir = locate_mot17_sequence(root, sequence)
        assert frames_dir is not None
        frame_paths = [frames_dir / f"{frame_id:06d}.jpg" for frame_id in sampled_ids]
        if not all(path.is_file() for path in frame_paths):
            continue
        official_split = "val" if "val" in annotation_path.parts else "train"
        samples.append(
            BuiltSample(
                dataset="lamot",
                sample_id=f"lamot_{annotation_path.stem}_{index:04d}",
                split=official_split,
                row=make_row("lamot", data["language"], video_frames=frame_paths),
                metadata={
                    "sequence": sequence,
                    "annotation_file": str(annotation_path.relative_to(root)),
                    "sampled_frames": sampled_ids,
                },
            )
        )
    return samples


def build_mevis(
    root: Path, limit: int, seed: int, materialized_root: Path
) -> list[BuiltSample]:
    dataset_root = dataset_directory(root, "MeVIS-valid")
    videos = load_json(dataset_root / "meta_expressions_v2_release.json")[
        "videos"
    ]
    candidates = [
        (video_id, expression_id, expression["exp"])
        for video_id, video in videos.items()
        for expression_id, expression in video["expressions"].items()
    ]
    selected = shuffled(candidates, seed + 43)[:limit]
    archive = dataset_root / "JPEGImages.tar"

    samples: list[BuiltSample] = []
    with TarFrameStore(archive, materialized_root) as store:
        for video_id, expression_id, expression in selected:
            frame_names = videos[video_id]["frames"]
            sampled_names = evenly_sample(frame_names, 6)
            frame_paths = [
                store.materialize(
                    f"JPEGImages/{video_id}/{frame_name}.jpg",
                    Path("mevis") / video_id / f"{frame_name}.jpg",
                )
                for frame_name in sampled_names
            ]
            samples.append(
                BuiltSample(
                    dataset="mevis",
                    sample_id=f"mevis_{video_id}_{expression_id}",
                    split=split_for_group(video_id, seed),
                    row=make_row("mevis", expression, video_frames=frame_paths),
                    metadata={
                        "video_id": video_id,
                        "expression_id": expression_id,
                        "sampled_frames": sampled_names,
                    },
                )
            )
    return samples


BUILDERS = {
    "d3": build_d3,
    "sorec": lambda root, limit, seed, output: build_from_template_manifest(
        root, "sorec", limit, seed
    ),
    "lasot": build_lasot,
    "tnl2k": lambda root, limit, seed, output: build_from_template_manifest(
        root, "tnl2k", limit, seed
    ),
    "refer_kitti": build_refer_kitti,
    "lamot": build_lamot,
    "mevis": build_mevis,
}


def validate_sample(sample: BuiltSample) -> None:
    messages = sample.row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Invalid message roles in {sample.sample_id}")
    if not messages[-1]["content"].strip():
        raise ValueError(f"Empty response in {sample.sample_id}")
    for field in ("images", "videos"):
        values = sample.row.get(field, [])
        paths = values[0] if field == "videos" and values else values
        for value in paths:
            if not Path(value).is_file():
                raise FileNotFoundError(f"Missing media in {sample.sample_id}: {value}")
    user_content = messages[1]["content"]
    if user_content.count("<image>") != len(sample.row.get("images", [])):
        raise ValueError(f"Image placeholder mismatch in {sample.sample_id}")
    if user_content.count("<video>") != len(sample.row.get("videos", [])):
        raise ValueError(f"Video placeholder mismatch in {sample.sample_id}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(output_root: Path, samples: Sequence[BuiltSample], mode: str) -> None:
    for dataset in DATASET_ORDER:
        dataset_samples = [sample for sample in samples if sample.dataset == dataset]
        if not dataset_samples:
            continue
        if mode == "demo":
            write_jsonl(
                output_root / "demo" / f"{dataset}.jsonl",
                (sample.row for sample in dataset_samples),
            )
            write_jsonl(
                output_root / "demo" / f"{dataset}_meta.jsonl",
                (
                    {"sample_id": sample.sample_id, **sample.metadata}
                    for sample in dataset_samples
                ),
            )
        else:
            for split in ("train", "val"):
                split_samples = [sample for sample in dataset_samples if sample.split == split]
                write_jsonl(
                    output_root / "full" / split / f"{dataset}.jsonl",
                    (sample.row for sample in split_samples),
                )
                write_jsonl(
                    output_root / "full" / split / f"{dataset}_meta.jsonl",
                    (
                        {"sample_id": sample.sample_id, **sample.metadata}
                        for sample in split_samples
                    ),
                )

    if mode == "demo":
        write_jsonl(output_root / "demo" / "merged.jsonl", (sample.row for sample in samples))
    else:
        for split in ("train", "val"):
            split_samples = [sample for sample in samples if sample.split == split]
            write_jsonl(
                output_root / "full" / f"{split}.jsonl",
                (sample.row for sample in split_samples),
            )


def media_html(row: dict[str, Any]) -> str:
    blocks: list[str] = []
    for image_path in row.get("images", []):
        uri = Path(image_path).resolve().as_uri()
        blocks.append(f'<img src="{html.escape(uri)}" alt="input image">')
    for video in row.get("videos", []):
        frames = "".join(
            f'<img src="{html.escape(Path(frame).resolve().as_uri())}" alt="video frame">'
            for frame in video
        )
        blocks.append(f'<div class="video">{frames}</div>')
    return "".join(blocks)


def write_demo_preview(path: Path, samples: Sequence[BuiltSample]) -> None:
    cards = []
    for sample in samples:
        messages = sample.row["messages"]
        cards.append(
            f"""
            <article>
              <h2>{html.escape(sample.dataset)} · {html.escape(sample.sample_id)}</h2>
              <div class="media">{media_html(sample.row)}</div>
              <h3>User</h3><p>{html.escape(messages[1]['content'])}</p>
              <h3>Assistant target</h3><p>{html.escape(messages[2]['content'])}</p>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MS-SWIFT dataset demo</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f4f5f7;color:#202124}}
main{{max-width:1400px;margin:auto}} article{{background:white;border:1px solid #d8dce2;
padding:16px;margin:0 0 16px;border-radius:6px}} h2{{font-size:18px;margin:0 0 12px}}
h3{{font-size:13px;margin:14px 0 4px;color:#475467}} p{{margin:0;line-height:1.45}}
.media,.video{{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}}
.media>img,.video img{{width:220px;height:160px;object-fit:contain;background:#111}}
.video{{padding:8px;border:1px solid #d8dce2}}
</style></head><body><main><h1>MS-SWIFT Dataset Demo</h1>{''.join(cards)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("dataset/ms_swift"))
    parser.add_argument("--mode", choices=("demo", "full"), default="demo")
    parser.add_argument("--demo-per-dataset", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lasot-scales",
        nargs="+",
        choices=tuple(LASOT_SCALE_RANGES),
        default=DEFAULT_LASOT_SCALES,
        help="LaSOT temporal ranges to sample: wide, long, and/or full.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output if args.output.is_absolute() else root / args.output
    materialized_root = output_root / "assets" / "materialized_frames"
    samples: list[BuiltSample] = []

    for dataset in args.datasets:
        limit = args.demo_per_dataset if args.mode == "demo" else FULL_QUOTAS[dataset]
        if dataset == "lasot":
            built = build_lasot(
                root,
                limit,
                args.seed,
                materialized_root,
                selected_scales=args.lasot_scales,
            )
        else:
            built = BUILDERS[dataset](root, limit, args.seed, materialized_root)
        for sample in built:
            if args.mode == "demo":
                sample.split = "demo"
            validate_sample(sample)
        samples.extend(built)
        print(f"{dataset}: {len(built)} samples")

    random.Random(args.seed + 101).shuffle(samples)
    write_outputs(output_root, samples, args.mode)
    if args.mode == "demo":
        preview = output_root / "demo" / "preview.html"
        write_demo_preview(preview, samples)
        print(f"preview: {preview}")
    print(f"total: {len(samples)} samples")


if __name__ == "__main__":
    main()
