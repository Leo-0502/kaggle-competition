"""Convert the competition COCO annotations into a reproducible YOLO dataset."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from fathomnet_utils import category_maps, coco_bbox_to_yolo, deterministic_split, load_json

LOGGER = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/fathomnet"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def download_image(url: str, destination: Path, max_size: int, quality: int, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            urllib.request.urlretrieve(url, temporary)
            with Image.open(temporary) as image:
                image = image.convert("RGB")
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                image.save(destination, "JPEG", quality=quality)
            temporary.unlink(missing_ok=True)
            return
        except Exception as error:  # network and invalid-image errors are retried together
            last_error = error
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    coco = load_json(args.annotations)
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)

    eligible_ids = [image_id for image_id in image_by_id if annotations[image_id]]
    train_ids, val_ids = deterministic_split(eligible_ids, args.val_fraction, args.seed)
    coco_to_yolo, names = category_maps(coco)

    failures: list[dict[str, object]] = []
    for image_id in tqdm(eligible_ids, desc="Preparing images"):
        split = "val" if image_id in val_ids else "train"
        image = image_by_id[image_id]
        stem = Path(image["file_name"]).stem
        image_path = args.output / "images" / split / f"{stem}.jpg"
        label_path = args.output / "labels" / split / f"{stem}.txt"
        try:
            if not image_path.exists():
                download_image(image["coco_url"], image_path, args.max_size, args.jpeg_quality, args.retries)
            lines = []
            for annotation in annotations[image_id]:
                values = coco_bbox_to_yolo(annotation["bbox"], image["width"], image["height"])
                if values[2] <= 0 or values[3] <= 0:
                    continue
                lines.append(f"{coco_to_yolo[int(annotation['category_id'])]} " + " ".join(f"{v:.6f}" for v in values))
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as error:
            image_path.unlink(missing_ok=True)
            failures.append({"image_id": image_id, "error": str(error)})
            LOGGER.warning("Skipped image %s: %s", image_id, error)

    config = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "names": names}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fathomnet.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.output / "category_mapping.json").write_text(
        json.dumps({"coco_to_yolo": coco_to_yolo, "yolo_to_coco": {v: k for k, v in coco_to_yolo.items()}}, indent=2),
        encoding="utf-8",
    )
    (args.output / "download_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    LOGGER.info("Prepared %d train and %d validation images (%d failures)", len(train_ids), len(val_ids), len(failures))


if __name__ == "__main__":
    main()
