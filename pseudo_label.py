"""Generate high-confidence YOLO pseudo-labels from selected test images."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from ultralytics import RTDETR

from fathomnet_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, help="JSON list containing image_id entries")
    parser.add_argument("--output", type=Path, default=Path("datasets/pseudo"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    images = {int(item["id"]): item for item in load_json(args.test_json)["images"]}
    selected_ids = set(images)
    if args.selection_json:
        selected_ids = {int(item["image_id"]) for item in load_json(args.selection_json)}
        unknown = selected_ids - set(images)
        if unknown:
            raise ValueError(f"Selection contains unknown image IDs: {sorted(unknown)[:5]}")

    model = RTDETR(str(args.weights))
    image_dir, label_dir = args.output / "images" / "train", args.output / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    kept, failures = 0, []
    with tempfile.TemporaryDirectory() as temporary_dir:
        download = Path(temporary_dir) / "image"
        for image_id in tqdm(sorted(selected_ids), desc="Pseudo-labeling"):
            item = images[image_id]
            stem = Path(item["file_name"]).stem
            destination = image_dir / f"{stem}.jpg"
            try:
                urllib.request.urlretrieve(item["coco_url"], download)
                with Image.open(download) as source:
                    source.convert("RGB").save(destination, "JPEG", quality=90)
                boxes = model.predict(source=str(destination), conf=args.conf, imgsz=args.imgsz, verbose=False)[0].boxes
                if len(boxes) == 0:
                    destination.unlink(missing_ok=True)
                    continue
                lines = []
                for box in boxes:
                    x, y, width, height = box.xywhn[0].tolist()
                    lines.append(f"{int(box.cls[0].item())} {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
                (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                kept += 1
            except Exception as error:
                destination.unlink(missing_ok=True)
                failures.append({"image_id": image_id, "error": str(error)})
                logging.warning("Skipped image %s: %s", image_id, error)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    logging.info("Kept %d pseudo-labeled images; %d failed", kept, len(failures))
    if failures:
        raise RuntimeError("Pseudo-label generation was incomplete")


if __name__ == "__main__":
    main()

