import pandas as pd
import pytest

from fathomnet_utils import SUBMISSION_COLUMNS, category_maps, coco_bbox_to_yolo, deterministic_split
from validate_submission import validate


def test_category_maps_sorts_non_contiguous_coco_ids():
    coco = {"categories": [{"id": 41, "name": "b"}, {"id": 2, "name": "a"}]}
    mapping, names = category_maps(coco)
    assert mapping == {2: 0, 41: 1}
    assert names == {0: "a", 1: "b"}


def test_bbox_conversion():
    assert coco_bbox_to_yolo([10, 20, 20, 20], 100, 200) == pytest.approx((0.2, 0.15, 0.2, 0.1))


def test_split_is_reproducible_and_disjoint():
    first = deterministic_split(list(range(20)), 0.2, 42)
    second = deterministic_split(list(range(20)), 0.2, 42)
    assert first == second
    assert first[0].isdisjoint(first[1])
    assert len(first[1]) == 4


def test_submission_validation_accepts_valid_frame():
    row = [1, 10, 2, 5.0, 6.0, 20.0, 30.0, 0.8]
    frame = pd.DataFrame([row], columns=SUBMISSION_COLUMNS)
    test_data = {"images": [{"id": 10, "width": 100, "height": 100}]}
    assert validate(frame, test_data) == []


def test_submission_validation_rejects_out_of_bounds_box():
    row = [1, 10, 2, 90.0, 6.0, 20.0, 30.0, 0.8]
    frame = pd.DataFrame([row], columns=SUBMISSION_COLUMNS)
    test_data = {"images": [{"id": 10, "width": 100, "height": 100}]}
    assert "outside image bounds" in validate(frame, test_data)[0]

