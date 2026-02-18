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
import json

# Allow running as either:
# - python -m scripts.eval_detector
# - python scripts/eval_detector.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.yolo import (
    eval_yolo_detector,  # Returns Ultralytics metrics object from model.val().
    save_yolo_metrics_json,  # Returns written metrics.json Path.
    save_metrics_table_csv,  # Returns written 2-column metrics CSV Path.
    save_run_metadata_artifacts,  # Returns (metadata_json_path, metadata_csv_path).
    infer_model_variant_from_weights,  # Returns model variant string (weights stem).
)
from src.paths import EVAL_DIR, EXPORTS_DIR, RUNS_DIR


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for detector evaluation.

    Input:
        CLI flags from the command line.

    Output:
        argparse.Namespace with backend/eval/runtime options.

    Why:
        Keeps eval invocation reproducible while allowing backend expansion.
    """
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
    parser.add_argument("--img-h", type=int, default=704)
    parser.add_argument("--img-w", type=int, default=1248)
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rectangular validation batches to preserve aspect ratio better.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--run-name", type=str, default="yolo_eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unclear-policy",
        type=str,
        default="exclude_unclear",
        help="Data filtering policy used when exporting dataset labels.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run evaluation for the selected backend and persist metrics.

    Input:
        None directly (reads parsed CLI args).

    Output:
        None (writes evaluation metrics files and prints output location).

    Why:
        Central entrypoint for benchmark/eval runs across detector backends.
    """
    args = parse_args()
    out_dir = Path(EVAL_DIR) / args.backend / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "yolo":
        metrics = eval_yolo_detector(
            data_yaml=args.data_yaml,
            weights_path=args.weights,
            split=args.split,
            imgsz=(args.img_h, args.img_w),
            rect=bool(args.rect),
            batch=args.batch,
            device=args.device,
            project=str(RUNS_DIR / "yolo"),
            name=f"{args.run_name}_val",
        )
        out_json = save_yolo_metrics_json(metrics=metrics, out_path=out_dir / "metrics.json")
        metrics_dict = json.loads(out_json.read_text())
        out_csv = save_metrics_table_csv(metrics_dict, out_dir / "metrics_table.csv")
        data_yaml_path = Path(args.data_yaml)
        dataset_export_name = data_yaml_path.parent.name if data_yaml_path.name == "dataset.yaml" else data_yaml_path.stem
        metadata = {
            "model_family": "yolo",
            "model_variant": infer_model_variant_from_weights(args.weights),
            "model_weights": args.weights,
            "run_name": args.run_name,
            "seed": int(args.seed),
            "split": args.split,
            "img_h": int(args.img_h),
            "img_w": int(args.img_w),
            "rect": bool(args.rect),
            "unclear_policy": args.unclear_policy,
            "dataset_export_name": dataset_export_name,
            "data_yaml": str(data_yaml_path),
        }
        meta_json, meta_csv = save_run_metadata_artifacts(
            metadata=metadata,
            out_json_path=out_dir / "run_metadata.json",
            out_csv_path=out_dir / "run_metadata.csv",
        )
        print(f"Saved metrics -> {out_json}")
        print(f"Saved table   -> {out_csv}")
        print(f"Saved run metadata -> {meta_json}")
        print(f"Saved metadata table -> {meta_csv}")
        return

    raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
