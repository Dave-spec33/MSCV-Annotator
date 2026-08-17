# Source datasets

MSCV-Annotator does not redistribute source images, videos, annotations,
generated target templates, or generated MS-SWIFT records. Download datasets
from their official providers and review their terms before use.

## Upstream sources

| Dataset | Original task and role in this project | Official source |
| --- | --- | --- |
| D3 | Described Object Detection; flexible image-level object or object-set expressions | [D3 repository](https://github.com/shikras/d-cube), [paper](https://arxiv.org/abs/2307.12813) |
| SOREC | Referring Expression Comprehension for Small Objects; small-target descriptions generated from SODA-D images | [SOREC repository](https://github.com/mmaiLab/sorec), [SODA project](https://shaunyuan22.github.io/SODA/) |
| Refer-KITTI-V2 | Referring multi-object tracking in road scenes | [TempRMOT repository](https://github.com/zyn213/TempRMOT), [paper](https://arxiv.org/abs/2406.05039), [KITTI tracking data](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) |
| LaMOT | Language-guided multi-object tracking | [LaMOT repository](https://github.com/Nathan-Li123/LaMOT), [paper](https://arxiv.org/abs/2406.08324) |
| LaSOT | Long-term single-object tracking with sequence-level language specifications | [LaSOT project](https://vision.cs.stonybrook.edu/~lasot/), [paper](https://arxiv.org/abs/2009.03465) |
| TNL2K | Single-object tracking with first-frame natural-language annotations | [TNL2K toolkit and download links](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit), [paper](https://arxiv.org/abs/2103.16621) |
| MeViS | Referring video object segmentation with motion expressions | [MeViS repository](https://github.com/henghuiding/MeViS), [paper](https://arxiv.org/abs/2308.08544) |

LaMOT annotations refer to media from MOT17, TAO, VisDrone2019, and SportsMOT.
The current builder uses the locally available MOT17 subset. Refer to the LaMOT
repository for all upstream download and citation requirements.

## Expected local layout

The scripts support the following project-relative layout:

```text
cv_datasets/
|-- d3/
|   |-- d3_images/
|   `-- d3_json/d3_full_annotations.json
|-- sorec/
|   |-- trainL.json
|   `-- SODA-D/Images/extracted/
|-- refer-kitti-v2/
|   |-- expression/
|   `-- kitti/data_tracking_image_2.zip
|-- LaMOT-main/
|   |-- annotations_v1/
|   `-- MOT17/
|-- LaSOTTest/
|   `-- <sequence>/
|       |-- img/
|       |-- groundtruth.txt
|       `-- language.txt
|-- TNL2K/
|   `-- TNL2k_train_subset_p5/
|       `-- <sequence>/
|           |-- imgs/
|           |-- groundtruth.txt
|           `-- language.txt
`-- MeVIS-valid/
    |-- JPEGImages.tar
    `-- meta_expressions_v2_release.json
```

For backward compatibility, each dataset may also be placed directly under the
project root, but `cv_datasets/` is the documented layout.

## Deterministic construction

The full build uses seed `42` and fixed source quotas. Train/validation splits
are assigned at sequence or image-group level to reduce visual leakage across
splits. Video records contain sampled frame lists rather than source GT.

The exact generated sample set still depends on the upstream dataset version
and locally available files. A partial source copy can therefore produce a
different selection or fewer samples than a complete official release. Record
the source versions and verify the printed per-source counts when reproducing a
training run.

## Ground-truth handling

SOREC, LaSOT, and TNL2K use a source bounding box to create a square target
template with surrounding context. The box is used only during offline dataset
construction. The resulting target template is a visual input, but numerical
coordinates and box overlays are not included in the user prompt, template, or
assistant response. Other generated metadata files may contain construction
diagnostics and are therefore kept local under the ignored `dataset/` tree.
