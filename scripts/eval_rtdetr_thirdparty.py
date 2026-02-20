"""
Evaluate RT-DETRv2 via official third-party PyTorch implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow running as either:
# - python -m scripts.eval_rtdetr_thirdparty
# - python scripts/eval_rtdetr_thirdparty.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.rtdetr_thirdparty import (
    eval_rtdetr_thirdparty,
    save_rtdetr_thirdparty_metrics_json,
    save_rtdetr_thirdparty_run_metadata,
    collect_runtime_info,
)
from src.models.vision.yolo import infer_model_variant_from_weights, save_metrics_table_csv
from src.paths import EVAL_DIR


DEFAULT_BASE_CONFIG_L = (
    PROJECT_ROOT / "third_party" / "rtdetr" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_6x_coco.yml"
)
DEFAULT_BASE_CONFIG_M = (
    PROJECT_ROOT / "third_party" / "rtdetr" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_m_7x_coco.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RT-DETRv2 (third-party) run.")
    parser.add_argument("--model-tier", choices=["l", "m"], default="l")
    parser.add_argument("--base-config", type=str, default=None, help="Optional explicit RT-DETRv2 config path.")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pth), usually best.pth.")
    parser.add_argument("--val-img-dir", type=str, required=True)
    parser.add_argument("--val-ann-json", type=str, required=True)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--img-h", type=int, default=704)
    parser.add_argument("--img-w", type=int, default=1248)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="rtdetrv2_l_thirdparty_eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unclear-policy", type=str, default="exclude_unclear")
    return parser.parse_args()


def _resolve_base_config(args: argparse.Namespace) -> Path:
    if args.base_config:
        return Path(args.base_config).resolve()
    return DEFAULT_BASE_CONFIG_L if args.model_tier == "l" else DEFAULT_BASE_CONFIG_M


def main() -> None:
    args = parse_args()
    base_config = _resolve_base_config(args)

    out_dir = Path(EVAL_DIR) / "rtdetr_thirdparty" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = eval_rtdetr_thirdparty(
        base_config=str(base_config),
        weights_path=args.weights,
        val_img_dir=args.val_img_dir,
        val_ann_json=args.val_ann_json,
        output_dir=str(out_dir),
        split=args.split,
        imgsz=(args.img_h, args.img_w),
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        num_classes=args.num_classes,
    )

    out_json = save_rtdetr_thirdparty_metrics_json(metrics=metrics, out_path=out_dir / "metrics.json")
    out_csv = save_metrics_table_csv(metrics, out_dir / "metrics_table.csv")
    print(f"Saved metrics -> {out_json}")
    print(f"Saved table   -> {out_csv}")

    weights_path = Path(args.weights)
    metadata = {
        "model_family": "rtdetr_thirdparty",
        "model_variant": infer_model_variant_from_weights(base_config.stem),
        "model_weights": str(weights_path),
        "run_name": args.run_name,
        "seed": int(args.seed),
        "split": args.split,
        "img_h": int(args.img_h),
        "img_w": int(args.img_w),
        "unclear_policy": args.unclear_policy,
        "base_config": str(base_config),
        "val_img_dir": str(Path(args.val_img_dir).resolve()),
        "val_ann_json": str(Path(args.val_ann_json).resolve()),
        "weights_file_size_mb": round(weights_path.stat().st_size / (1024**2), 3) if weights_path.exists() else None,
    }
    metadata.update(collect_runtime_info())
    meta_json, meta_csv = save_rtdetr_thirdparty_run_metadata(metadata=metadata, out_dir=out_dir)
    print(f"Saved run metadata -> {meta_json}")
    print(f"Saved metadata table -> {meta_csv}")

    # Convenience copy of key metrics for quick spot-checking.
    key_json = out_dir / "metrics_key.json"
    key_json.write_text(
        json.dumps(
            {
                "map50_95": metrics.get("map50_95"),
                "map50": metrics.get("map50"),
                "recall": metrics.get("recall"),
            },
            indent=2,
        )
    )
    print(f"Saved key metrics -> {key_json}")


if __name__ == "__main__":
    main()

