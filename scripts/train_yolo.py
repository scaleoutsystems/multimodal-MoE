"""
Train a YOLO baseline from an exported YOLO dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import json

# Allow running as either:
# - python -m scripts.train_yolo
# - python scripts/train_yolo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.yolo import (
    YoloTrainConfig,
    train_yolo_detector,
    save_yolo_training_summary,
    save_run_metadata_artifacts,
    infer_model_variant_from_weights,
    get_git_commit_hash,
)
from src.paths import EVAL_DIR, EXPORTS_DIR, RUNS_DIR


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for YOLO baseline training.

    Input:
        CLI flags from the command line.

    Output:
        argparse.Namespace with training/runtime settings.

    Why:
        Keeps training configuration explicit and easy to reproduce.
    """
    parser = argparse.ArgumentParser(description="Train YOLO baseline detector.")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=str(EXPORTS_DIR / "yolo" / "pedestrian_v1_exclude_unclear" / "dataset.yaml"),
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--img-h", type=int, default=704)
    parser.add_argument("--img-w", type=int, default=1248)
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rectangular batching to preserve aspect ratio better.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="yolo26n_pedestrian_v1")
    parser.add_argument(
        "--unclear-policy",
        type=str,
        default="exclude_unclear",
        help="Data filtering policy used when exporting dataset labels.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Construct YOLO train config and launch training.

    Input:
        None directly (reads parsed CLI args).

    Output:
        None (starts Ultralytics training run and writes run artifacts).

    Why:
        Thin script wrapper around the YOLO adapter to keep script logic minimal.
    """
    args = parse_args()
    cfg = YoloTrainConfig(
        data_yaml=args.data_yaml,
        model=args.model,
        imgsz=(args.img_h, args.img_w),
        rect=bool(args.rect),
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
    t0 = time.perf_counter()
    results = train_yolo_detector(cfg)
    train_wall_time_s = time.perf_counter() - t0

    # Save concise training report artifacts for cross-variant comparisons.
    train_report_dir = Path(EVAL_DIR) / "yolo" / args.run_name
    train_report_dir.mkdir(parents=True, exist_ok=True)
    summary_json, summary_csv = save_yolo_training_summary(
        train_wall_time_s=train_wall_time_s,
        model_name=args.model,
        data_yaml=args.data_yaml,
        run_name=args.run_name,
        out_json_path=train_report_dir / "train_summary.json",
        out_csv_path=train_report_dir / "train_summary.csv",
        results=results,
    )
    print(f"Saved training summary -> {summary_json}")
    print(f"Saved training table   -> {summary_csv}")

    # Also save any directly available training metrics from Ultralytics.
    if hasattr(results, "results_dict"):
        metrics_json = train_report_dir / "train_metrics.json"
        try:
            metrics_json.write_text(json.dumps(dict(results.results_dict), indent=2))
            print(f"Saved train metrics   -> {metrics_json}")
        except Exception:
            pass

    # Save explicit run metadata for future cross-family aggregation.
    data_yaml_path = Path(args.data_yaml)
    dataset_export_name = data_yaml_path.parent.name if data_yaml_path.name == "dataset.yaml" else data_yaml_path.stem
    metadata = {
        "model_family": "yolo",
        "model_variant": infer_model_variant_from_weights(args.model),
        "model_weights": args.model,
        "run_name": args.run_name,
        "seed": int(args.seed),
        "split": "train+val",
        "img_h": int(args.img_h),
        "img_w": int(args.img_w),
        "rect": bool(args.rect),
        "unclear_policy": args.unclear_policy,
        "dataset_export_name": dataset_export_name,
        "data_yaml": str(data_yaml_path),
        "git_commit": get_git_commit_hash(),
    }
    meta_json, meta_csv = save_run_metadata_artifacts(
        metadata=metadata,
        out_json_path=train_report_dir / "run_metadata.json",
        out_csv_path=train_report_dir / "run_metadata.csv",
    )
    print(f"Saved run metadata   -> {meta_json}")
    print(f"Saved metadata table -> {meta_csv}")


if __name__ == "__main__":
    main()
