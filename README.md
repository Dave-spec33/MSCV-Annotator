# MSCV-Annotator

MSCV-Annotator builds a multi-source visual-language supervision dataset for
LoRA/QLoRA fine-tuning of Qwen-VL models. The model learns to produce concise,
natural-language semantic guidance for visual targets and scenes, including
target descriptions, scene context, and tracking or motion cues.

This repository publishes dataset construction and validation code, the
Qwen3-VL-8B QLoRA training command used on a single RTX 4090, and reproducible
comparison and text-quality evaluation scripts. It does **not** redistribute
source datasets, generated JSONL files, target crops, sampled frames, or
portable dataset bundles.

## Dataset composition

The default full build contains 19,800 samples.

| Source | Samples | Visual input | Supervision focus |
| --- | ---: | --- | --- |
| D3 | 9,000 | One image | Flexible described-object expression |
| SOREC | 6,000 | Full image + GT-derived target template | Small-object referring expression |
| Refer-KITTI-V2 | 2,400 | Six sampled frames | Referring multi-object tracking |
| LaMOT | 300 | Up to six sampled frames | Language-guided multi-object tracking |
| LaSOT | 100 | GT-derived first-frame template + five sampled frames | Long-term single-object tracking guidance |
| TNL2K | 100 | First frame + GT-derived target template | First-frame natural-language target description |
| MeViS | 1,900 | Six sampled frames | Motion-focused referring expression |

The default random seed is `42`. LaSOT supports `wide`, `long`, and `full`
temporal sampling ranges; the current default is `long`. See
[docs/DATASETS.md](docs/DATASETS.md) for upstream sources and the expected local
layout.

## Requirements

Dataset construction requires Python 3.10 or newer and Pillow:

```bash
python -m pip install -r requirements.txt
```

Training additionally requires a CUDA-compatible PyTorch environment,
MS-SWIFT, bitsandbytes, and a local Qwen3-VL model. Record the exact package,
CUDA, base-model revision, and GPU versions when reporting results.

## Build the dataset

Download each source dataset from its official provider and arrange it under
`cv_datasets/` as documented in [docs/DATASETS.md](docs/DATASETS.md). No source
dataset is downloaded automatically.

Generate the GT-initialized target templates used by SOREC, LaSOT, and TNL2K:

```bash
python scripts/generate_target_templates.py --root .
```

Build a small demo first. It creates three samples per source and an HTML
preview under the ignored `dataset/ms_swift/demo/` directory:

```bash
python scripts/build_ms_swift_dataset.py --mode demo --demo-per-dataset 3
```

Build the full 19,800-sample dataset:

```bash
python scripts/build_ms_swift_dataset.py --mode full --lasot-scales long
```

Alternative LaSOT ranges can be selected independently or combined:

```bash
python scripts/build_ms_swift_dataset.py --mode full --lasot-scales wide long full
```

## Create a local portable bundle

This optional step copies only media referenced by the generated JSONL files.
The resulting bundle is intended for private machine-to-machine migration and
is excluded from Git.

```bash
python scripts/materialize_portable_dataset.py --dry-run
python scripts/materialize_portable_dataset.py --output portable_dataset
python portable_dataset/scripts/resolve_paths.py --bundle-root portable_dataset
python scripts/check_portable_ms_swift_dataset.py \
  --bundle-root portable_dataset \
  --verify-images
```

More details are available in
[docs/PORTABLE_DATASET.md](docs/PORTABLE_DATASET.md).

## Train Qwen3-VL-8B with QLoRA

The provided script targets one RTX 4090 and keeps the vision encoder and
aligner frozen. Paths can be overridden without editing the script:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/portable_dataset \
OUTPUT_DIR=/path/to/output/qwen3vl8b_cv_semantic_qlora \
bash scripts/train_qwen3vl8b_qlora_4090.sh
```

The dataset and training script use ground-truth boxes only to construct target
templates for selected first-frame inputs. Coordinates, masks, and track IDs are
not exposed in the user prompt or expected model response.

## Released weights

The released training weights are hosted externally and are not stored in this
Git repository:

- [Download from Baidu Netdisk](https://pan.baidu.com/s/1MWE8Tmw1PhZX9S65qvm0Uw?pwd=uqvs)
- Extraction code: `uqvs`

## Evaluation

All models were evaluated on the same 1,488-sample validation split.

| Model | BERTScore-P | BERTScore-R | BERTScore-F1 | Format success |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-Plus | 0.9700 | 0.9779 | 0.9740 | 0.9892 (1472/1488) |
| Qwen3-VL-8B-Instruct | 0.9660 | 0.9779 | 0.9719 | 0.9940 (1479/1488) |
| Qwen3-VL-8B-Instruct + QLoRA | **0.9857** | **0.9842** | **0.9850** | **0.9993 (1487/1488)** |

See [eval/README.md](eval/README.md) for inference commands, API setup, metric
definitions, evaluation commands, and interpretation limits.

## Data and licensing

Users must obtain every source dataset directly from its provider and comply
with its terms. The construction scripts do not change or replace the licenses
of D3, SOREC/SODA-D, Refer-KITTI-V2/KITTI, LaMOT and its source datasets,
LaSOT, TNL2K, or MeViS.

No license is currently granted for this repository's own source code. A code
license should be selected before treating the repository as open-source.
