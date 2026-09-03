"""Shared, dependency-light helpers for the FathomNet pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SUBMISSION_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "score",
]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def category_maps(coco: dict[str, Any]) -> tuple[dict[int, int], dict[int, str]]:
    """Return COCO-to-contiguous IDs and contiguous class names."""
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    coco_to_yolo = {int(cat["id"]): index for index, cat in enumerate(categories)}
    names = {index: str(cat["name"]) for index, cat in enumerate(categories)}
    return coco_to_yolo, names


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, ...]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x, y, box_width, box_height = map(float, bbox)
    return (
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )


def deterministic_split(
    image_ids: list[int], val_fraction: float, seed: int
) -> tuple[set[int], set[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    ids = sorted(set(image_ids))
    random.Random(seed).shuffle(ids)
    val_count = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else 0
    return set(ids[val_count:]), set(ids[:val_count])


def clip_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        min(max(float(x1), 0.0), width),
        min(max(float(y1), 0.0), height),
        min(max(float(x2), 0.0), width),
        min(max(float(y2), 0.0), height),
    ]

