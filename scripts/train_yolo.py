"""
Train a YOLO baseline from an exported YOLO dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as either:
# - python -m scripts.train_yolo
# - python scripts/train_yolo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.yolo import YoloTrainConfig, train_yolo_detector
from src.paths import EXPORTS_DIR, RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO baseline detector.")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=str(EXPORTS_DIR / "yolo" / "pedestrian_v1_exclude_unclear" / "dataset.yaml"),
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="yolov8n_pedestrian_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = YoloTrainConfig(
        data_yaml=args.data_yaml,
        model=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        project=str(RUNS_DIR / "yolo"),
        name=args.run_name,
    )

    print("Starting YOLO training with config:")
    print(cfg)
    train_yolo_detector(cfg)


if __name__ == "__main__":
    main()
