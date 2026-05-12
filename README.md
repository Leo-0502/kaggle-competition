# 🌊 FathomNet CLEF 2026 Solution: 0.0716+ Score Strategy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-RTDETR-blueviolet.svg)](https://github.com/ultralytics/ultralytics)

> **Competition:** [FathomNet CLEF 2026 @ LifeCLEF & CVPR-FGVC](https://www.life-clef.org/2026/fathomnet)
> **Final Score:** 0.0716 (Top XX% Strategy)
> **Location:** Shanghai, China (2026-05-12)

This repository contains the complete source code for our submission to the FathomNet CLEF 2026 challenge. Our solution leverages **RT-DETR** for real-time underwater object detection, combined with advanced ensemble techniques including **Pseudo-Labeling** and **Weighted Boxes Fusion (WBF)** to achieve high accuracy in fine-grained marine species classification.

## 🚀 Core Methodology

Our pipeline is designed to handle the challenges of underwater imagery, such as low visibility and class imbalance.

### 1. Data Preprocessing & Augmentation
- **FathomNet Dataset:** Converted from COCO JSON format to YOLOv8/RT-DETR format.
- **Resolution:** Trained at **1024x1024** to capture fine details of small marine organisms.
- **Cleaning:** Automated removal of corrupted or unlabeled images.

### 2. Teacher-Student Distillation (Semi-Supervised Learning)
To overcome the limited labeled data, we implemented a self-training loop:
- **Teacher Model:** A pre-trained RT-DETR model (or LLaVA-guided picks) used to infer on the unlabeled test set.
- **Pseudo-Labeling:** High-confidence predictions (`conf > 0.25`) from the teacher were used as "hard" labels.
- **Student Training:** The student model (RT-DETR-l) was trained on the merged dataset of official training data + pseudo-labeled test data.

### 3. Inference & Ensemble
- **TTA (Test Time Augmentation):** Enabled during inference to boost mAP.
- **WBF (Weighted Boxes Fusion):** Fused predictions from two models (`0.0680` and `0.0716` checkpoints) with different weights to suppress false positives and refine bounding boxes.

## 📂 Repository Structure

```text
fathomnet-clef-2026/
├── fathomnet.yaml          # Dataset configuration for Ultralytics
├── train_pseudo.py         # Script for data fusion & training
├── infer_tta.py            # Inference script with TTA
├── wbf_ensemble.py         # Weighted Boxes Fusion script
└── datasets/               # (Symlink or processed data)
    └── fathomnet/
        ├── images/
        └── labels/
