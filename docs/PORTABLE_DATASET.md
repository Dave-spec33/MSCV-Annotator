# Portable Dataset Workflow

> This bundle contains derived dataset media and annotations. It is intended
> only for private migration between authorized machines and is excluded from
> the public Git repository.

The generated MS-SWIFT JSONL files initially contain absolute paths into the
source datasets. Use the portable export step after building the full dataset so
that the source D3, SOREC, KITTI, LaMOT, LaSOT, TNL2K, and MeVIS directories do
not need to be migrated.

## 1. Inspect required space

```powershell
python scripts/materialize_portable_dataset.py --dry-run
```

This scans all JSONL files under `dataset/ms_swift/full`, deduplicates media
references, and reports the number and total size of files that will be copied.

## 2. Export

```powershell
python scripts/materialize_portable_dataset.py --output portable_dataset
```

Use the default `copy` mode for a real migration. For a quick local validation,
`--mode hardlink` avoids duplicating file contents but the result should not be
treated as an independent backup.

The resulting structure is:

```text
portable_dataset/
|-- README.md
|-- data/                       # Canonical JSONL with relative media paths
|   |-- train.jsonl
|   |-- val.jsonl
|   |-- train/
|   `-- val/
|-- media/files/                # Deduplicated images and video frames
|-- metadata/
|   |-- bundle.json
|   `-- media_manifest.jsonl
|-- scripts/resolve_paths.py
`-- runtime_data/               # Created on the destination machine
```

## 3. Use on another machine

After copying only `portable_dataset`, run:

```powershell
python portable_dataset/scripts/resolve_paths.py --bundle-root portable_dataset
```

This creates `portable_dataset/runtime_data` and writes absolute media paths for
the current environment. Train with `runtime_data/train.jsonl` and validate with
`runtime_data/val.jsonl`.

## Integrity check

Add `--checksum` during export to record a SHA-256 digest for every media file.
Use `resolve_paths.py --check-only` after migration to verify that every media
reference can be resolved without writing runtime JSONL files.
