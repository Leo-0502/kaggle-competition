"""Run RT-DETR inference and create a competition-format submission."""

from __future__ import annotations

import argparse
import logging
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import RTDETR

from fathomnet_utils import SUBMISSION_COLUMNS, category_maps, clip_xyxy, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_data, test_data = load_json(args.train_json), load_json(args.test_json)
    coco_to_yolo, _ = category_maps(train_data)
    yolo_to_coco = {value: key for key, value in coco_to_yolo.items()}
    model = RTDETR(str(args.weights))
    predictions, failures = [], []

    with tempfile.TemporaryDirectory() as temporary_dir:
        image_path = Path(temporary_dir) / "image"
        for image in tqdm(test_data["images"], desc="Inference"):
            try:
                urllib.request.urlretrieve(image["coco_url"], image_path)
                kwargs = dict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, augment=args.tta, verbose=False)
                if args.device is not None:
                    kwargs["device"] = args.device
                result = model.predict(**kwargs)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = clip_xyxy(box.xyxy[0].tolist(), image["width"], image["height"])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append({
                        "annotation_id": len(predictions) + 1,
                        "image_id": int(image["id"]),
                        "category_id": yolo_to_coco[int(box.cls[0].item())],
                        "bbox_x": x1, "bbox_y": y1,
                        "bbox_width": x2 - x1, "bbox_height": y2 - y1,
                        "score": min(max(float(box.conf[0].item()), 0.0), 1.0),
                    })
            except Exception as error:
                failures.append((image["id"], str(error)))
                logging.warning("Skipped image %s: %s", image["id"], error)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions, columns=SUBMISSION_COLUMNS).to_csv(args.output, index=False)
    logging.info("Wrote %d predictions to %s; %d images failed", len(predictions), args.output, len(failures))
    if failures:
        raise RuntimeError("Inference was incomplete; inspect warnings before submitting")


if __name__ == "__main__":
    main()

