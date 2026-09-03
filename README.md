# FathomNet 2026 RT-DETR Pipeline

[![tests](https://github.com/Leo-0502/kaggle-competition/actions/workflows/tests.yml/badge.svg)](https://github.com/Leo-0502/kaggle-competition/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible object-detection pipeline for the **FathomNet 2026** competition. The project converts the released COCO annotations to YOLO format, trains an Ultralytics RT-DETR model, optionally creates pseudo-labels, generates a submission, and ensembles multiple submissions with Weighted Boxes Fusion (WBF).

The best score recorded during development was **0.0716**. Scores depend on the data version, split, model weights, and competition evaluation server; the repository does not bundle datasets or checkpoints.

## Pipeline

```text
COCO annotations + remote images
              |
              v
  deterministic train/val split
              |
              v
       RT-DETR training  <--- optional pseudo-label set
              |
              v
       TTA inference ----> submission validation
              |
              v
 optional multi-model WBF
```

Key choices used for the reported experiments:

- RT-DETR-L with 1024-pixel input size
- deterministic 85/15 train/validation split
- optional teacher-generated pseudo-labels at confidence 0.25
- TTA inference at confidence 0.15
- WBF with IoU threshold 0.5 and a larger weight for the stronger model

Pseudo-labeling is self-training, not knowledge distillation: this implementation produces hard detection labels and retrains a student on them. Before using test images for pseudo-labeling, confirm that the current competition rules allow it.

## Repository layout

```text
.
|-- prepare_data.py          # COCO -> YOLO conversion, download, train/val split
|-- train.py                 # RT-DETR training CLI
|-- pseudo_label.py          # optional teacher pseudo-label generation
|-- infer.py                 # TTA inference and submission generation
|-- ensemble.py              # WBF across two or more submissions
|-- validate_submission.py   # schema and bounding-box checks
|-- fathomnet_utils.py       # shared conversion and mapping helpers
|-- tests/                   # dependency-light unit tests
`-- .virtual_documents/      # original Kaggle notebook source (legacy reference)
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/Leo-0502/kaggle-competition.git
cd kaggle-competition
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Kaggle, the annotation paths are normally:

```text
/kaggle/input/competitions/fathomnet-2026/train_dataset.json
/kaggle/input/competitions/fathomnet-2026/test_dataset.json
```

## 1. Prepare the dataset

```bash
python prepare_data.py \
  --annotations /kaggle/input/competitions/fathomnet-2026/train_dataset.json \
  --output /kaggle/working/datasets/fathomnet \
  --val-fraction 0.15 \
  --seed 42
```

This downloads annotated images, writes YOLO labels, and creates `fathomnet.yaml`, `category_mapping.json`, and `download_failures.json`. Failures are reported instead of silently ignored, and re-running resumes already downloaded images.

## 2. Train

```bash
python train.py \
  --data /kaggle/working/datasets/fathomnet/fathomnet.yaml \
  --model rtdetr-l.pt \
  --epochs 12 \
  --imgsz 1024 \
  --batch 8 \
  --device 0
```

Adjust `--batch` for available GPU memory. For multiple GPUs, Ultralytics accepts a value such as `--device 0,1`.

## 3. Optional pseudo-labels

`selection.json` is optional. When supplied, it must be a JSON list whose entries contain `image_id`; without it, all test images are considered.

```bash
python pseudo_label.py \
  --weights /kaggle/input/my-model/best.pt \
  --test-json /kaggle/input/competitions/fathomnet-2026/test_dataset.json \
  --selection-json /kaggle/input/my-selection/selection.json \
  --output /kaggle/working/datasets/pseudo \
  --conf 0.25
```

Review pseudo-label precision visually before adding the resulting image directory to an Ultralytics dataset YAML. A weak teacher can amplify its own errors.

## 4. Inference and validation

```bash
python infer.py \
  --weights /kaggle/input/my-model/best.pt \
  --train-json /kaggle/input/competitions/fathomnet-2026/train_dataset.json \
  --test-json /kaggle/input/competitions/fathomnet-2026/test_dataset.json \
  --output /kaggle/working/submission.csv \
  --imgsz 1024 \
  --conf 0.15 \
  --tta

python validate_submission.py /kaggle/working/submission.csv \
  --test-json /kaggle/input/competitions/fathomnet-2026/test_dataset.json
```

Inference converts contiguous YOLO class indices back to original COCO category IDs. It clips predictions to image bounds and fails at the end if any image could not be processed.

## 5. Ensemble

```bash
python ensemble.py model_a.csv model_b.csv \
  --test-json /kaggle/input/competitions/fathomnet-2026/test_dataset.json \
  --weights 1 2 \
  --iou-threshold 0.5 \
  --output submission_wbf.csv
```

Use validation or leaderboard evidence to choose weights. Giving a poor model nonzero weight can reduce performance.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The unit tests cover category mapping, bounding-box conversion, deterministic splitting, and submission validation. Full training and inference tests require competition data and a CUDA-capable environment.

## Reproducibility notes

- Dataset files, pseudo-label selections, and model checkpoints are intentionally excluded from Git.
- The original Kaggle notebook export remains under `.virtual_documents/` only as historical reference; use the CLI scripts for new runs.
- Record package versions, Kaggle dataset versions, random seed, and checkpoint hash alongside each leaderboard result.

## License

Released under the [MIT License](LICENSE). Competition data and pretrained weights remain subject to their own terms.
