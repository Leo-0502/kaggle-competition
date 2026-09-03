"""Fuse two or more competition submissions with Weighted Boxes Fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

from fathomnet_utils import SUBMISSION_COLUMNS, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submissions", nargs="+", type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--skip-box-threshold", type=float, default=0.0001)
    parser.add_argument("--output", type=Path, default=Path("submission_wbf.csv"))
    args = parser.parse_args()
    if args.weights and len(args.weights) != len(args.submissions):
        parser.error("--weights must contain one value per submission")

    frames = [pd.read_csv(path) for path in args.submissions]
    image_info = {int(item["id"]): item for item in load_json(args.test_json)["images"]}
    image_ids = sorted(set().union(*(set(frame["image_id"].astype(int)) for frame in frames)))
    rows = []
    for image_id in image_ids:
        width, height = image_info[image_id]["width"], image_info[image_id]["height"]
        boxes_list, scores_list, labels_list = [], [], []
        for frame in frames:
            group = frame[frame["image_id"] == image_id]
            boxes_list.append([
                [max(0, row.bbox_x / width), max(0, row.bbox_y / height),
                 min(1, (row.bbox_x + row.bbox_width) / width), min(1, (row.bbox_y + row.bbox_height) / height)]
                for row in group.itertuples()
            ])
            scores_list.append(group["score"].astype(float).tolist())
            labels_list.append(group["category_id"].astype(int).tolist())
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=args.weights,
            iou_thr=args.iou_threshold, skip_box_thr=args.skip_box_threshold,
        )
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            rows.append({"annotation_id": len(rows) + 1, "image_id": image_id, "category_id": int(label),
                         "bbox_x": x1 * width, "bbox_y": y1 * height,
                         "bbox_width": (x2 - x1) * width, "bbox_height": (y2 - y1) * height,
                         "score": min(float(score), 1.0)})
    pd.DataFrame(rows, columns=SUBMISSION_COLUMNS).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

