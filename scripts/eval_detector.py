"""
Evaluate detector runs.

Current backend support:
- yolo (Ultralytics)

The CLI name stays generic so we can add DINO/fusion backends later without
changing downstream workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as either:
# - python -m scripts.eval_detector
# - python scripts/eval_detector.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.yolo import eval_yolo_detector, save_yolo_metrics_json
from src.paths import EVAL_DIR, EXPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector run.")
    parser.add_argument("--backend", choices=["yolo"], default="yolo")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=str(EXPORTS_DIR / "yolo" / "pedestrian_v1_exclude_unclear" / "dataset.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained detector weights (for YOLO, typically best.pt).",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--run-name", type=str, default="yolo_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(EVAL_DIR) / args.backend / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "yolo":
        metrics = eval_yolo_detector(
            data_yaml=args.data_yaml,
            weights_path=args.weights,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        out_json = save_yolo_metrics_json(metrics=metrics, out_path=out_dir / "metrics.json")
        print(f"Saved metrics -> {out_json}")
        return

    raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
