"""Train an Ultralytics RT-DETR model."""

import argparse

from ultralytics import RTDETR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/fathomnet/fathomnet.yaml")
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/rtdetr")
    parser.add_argument("--name", default="fathomnet_baseline")
    args = parser.parse_args()
    options = vars(args)
    model_path = options.pop("model")
    RTDETR(model_path).train(**options)


if __name__ == "__main__":
    main()
