"""Validate schema, IDs, scores, and box bounds before uploading a submission."""

import argparse
from pathlib import Path

import pandas as pd

from fathomnet_utils import SUBMISSION_COLUMNS, load_json


def validate(frame: pd.DataFrame, test_data: dict) -> list[str]:
    errors: list[str] = []
    if list(frame.columns) != SUBMISSION_COLUMNS:
        errors.append(f"Columns must be exactly: {SUBMISSION_COLUMNS}")
        return errors
    if frame.isna().any().any():
        errors.append("Submission contains missing values")
    if frame["annotation_id"].duplicated().any():
        errors.append("annotation_id values must be unique")
    if not frame["score"].between(0, 1).all():
        errors.append("Scores must be in [0, 1]")
    if (frame[["bbox_width", "bbox_height"]] <= 0).any().any():
        errors.append("Box width and height must be positive")
    image_info = {int(item["id"]): item for item in test_data["images"]}
    if not set(frame["image_id"].astype(int)).issubset(image_info):
        errors.append("Submission contains unknown image IDs")
    else:
        for row in frame.itertuples():
            image = image_info[int(row.image_id)]
            if row.bbox_x < 0 or row.bbox_y < 0 or row.bbox_x + row.bbox_width > image["width"] + 1e-6 or row.bbox_y + row.bbox_height > image["height"] + 1e-6:
                errors.append(f"Box for annotation {row.annotation_id} is outside image bounds")
                break
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    args = parser.parse_args()
    errors = validate(pd.read_csv(args.submission), load_json(args.test_json))
    if errors:
        raise SystemExit("Invalid submission:\n- " + "\n- ".join(errors))
    print(f"Valid submission: {args.submission}")


if __name__ == "__main__":
    main()

