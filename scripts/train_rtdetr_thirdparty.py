"""
Train RT-DETRv2 via official third-party PyTorch implementation.

cd /home/edgelab/multimodal-MoE
conda activate multimodal-moe

MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
example run: 
FIRST: 
MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
THEN: 
python -m scripts.train_rtdetr_thirdparty \
  --model-tier l \
  --train-img-dir "/home/edgelab/multimodal-MoE/outputs/exports/coco/pedestrian_v1_exclude_unclear/images/train" \
  --train-ann-json "/home/edgelab/multimodal-MoE/outputs/exports/coco/pedestrian_v1_exclude_unclear/annotations/instances_train.json" \
  --val-img-dir "/home/edgelab/multimodal-MoE/outputs/exports/coco/pedestrian_v1_exclude_unclear/images/val" \
  --val-ann-json "/home/edgelab/multimodal-MoE/outputs/exports/coco/pedestrian_v1_exclude_unclear/annotations/instances_val.json" \
  --img-h 704 --img-w 1248 \
  --epochs 1 \
  --batch 2 \
  --grad-accum-steps 8 \
  --device cuda:0 \
  --seed 0 \
  --workers 4 \
  --num-classes 1 \
  --use-amp \
  --run-name "rtdetrv2_l_e1_smoke_1248x704_p10_noaug_seed0_fix3_b2xacc8"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow running as either:
# - python -m scripts.train_rtdetr_thirdparty
# - python scripts/train_rtdetr_thirdparty.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.rtdetr_thirdparty import (
    RtdetrThirdPartyTrainConfig,
    train_rtdetr_thirdparty,
    save_rtdetr_thirdparty_training_summary,
    save_rtdetr_thirdparty_run_metadata,
    collect_runtime_info,
)
from src.models.vision.yolo import infer_model_variant_from_weights
from src.paths import EVAL_DIR, RUNS_DIR


DEFAULT_BASE_CONFIG_L = (
    PROJECT_ROOT / "third_party" / "rtdetr" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_6x_coco.yml"
)
DEFAULT_BASE_CONFIG_M = (
    PROJECT_ROOT / "third_party" / "rtdetr" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_m_7x_coco.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETRv2 (third-party) baseline detector.")
    parser.add_argument("--model-tier", choices=["l", "m"], default="m")
    parser.add_argument("--base-config", type=str, default=None, help="Optional explicit RT-DETRv2 config path.")
    parser.add_argument("--train-img-dir", type=str, required=True)
    parser.add_argument("--train-ann-json", type=str, required=True)
    parser.add_argument("--val-img-dir", type=str, required=True)
    parser.add_argument("--val-ann-json", type=str, required=True)
    parser.add_argument("--img-h", type=int, default=704)
    parser.add_argument("--img-w", type=int, default=1248)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before optimizer step.",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="Training log print frequency in iterations.",
    )
    parser.add_argument(
        "--tqdm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable tqdm progress bar for training iterations.",
    )
    parser.add_argument("--run-name", type=str, default="rtdetrv2_l_thirdparty")
    parser.add_argument("--unclear-policy", type=str, default="exclude_unclear")
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic mixed precision.",
    )
    parser.add_argument(
        "--disable-extra-geom-augs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable RT-DETR geometric augs that YOLO no-geom runs do not use "
            "(e.g., RandomZoomOut/RandomIoUCrop), while keeping fixed resize."
        ),
    )
    return parser.parse_args()


def _resolve_base_config(args: argparse.Namespace) -> Path:
    if args.base_config:
        return Path(args.base_config).resolve()
    return DEFAULT_BASE_CONFIG_L if args.model_tier == "l" else DEFAULT_BASE_CONFIG_M


def main() -> None:
    args = parse_args()
    base_config = _resolve_base_config(args)

    run_dir = Path(RUNS_DIR) / "rtdetr_thirdparty" / args.run_name
    eval_dir = Path(EVAL_DIR) / "rtdetr_thirdparty" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    cfg = RtdetrThirdPartyTrainConfig(
        base_config=str(base_config),
        train_img_dir=args.train_img_dir,
        train_ann_json=args.train_ann_json,
        val_img_dir=args.val_img_dir,
        val_ann_json=args.val_ann_json,
        output_dir=str(run_dir),
        run_name=args.run_name,
        imgsz=(args.img_h, args.img_w),
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        num_classes=args.num_classes,
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        print_freq=max(1, int(args.print_freq)),
        use_tqdm=bool(args.tqdm),
        use_amp=bool(args.use_amp),
        disable_extra_geom_augs=bool(args.disable_extra_geom_augs),
    )

    print("Starting third-party RT-DETRv2 training with config:")
    print(cfg)
    result = train_rtdetr_thirdparty(cfg)

    summary_json, summary_csv = save_rtdetr_thirdparty_training_summary(
        run_name=args.run_name,
        model_name=base_config.stem,
        base_config=str(base_config),
        train_wall_time_s=float(result["train_wall_time_s"]),
        out_json_path=eval_dir / "train_summary.json",
        out_csv_path=eval_dir / "train_summary.csv",
    )
    print(f"Saved training summary -> {summary_json}")
    print(f"Saved training table   -> {summary_csv}")

    metadata = {
        "model_family": "rtdetr_thirdparty",
        "model_variant": infer_model_variant_from_weights(base_config.stem),
        "model_weights": str(result["best_weights_path"]),
        "run_name": args.run_name,
        "seed": int(args.seed),
        "split": "train+val",
        "img_h": int(args.img_h),
        "img_w": int(args.img_w),
        "unclear_policy": args.unclear_policy,
        "base_config": str(base_config),
        "train_img_dir": str(Path(args.train_img_dir).resolve()),
        "train_ann_json": str(Path(args.train_ann_json).resolve()),
        "val_img_dir": str(Path(args.val_img_dir).resolve()),
        "val_ann_json": str(Path(args.val_ann_json).resolve()),
        "run_dir": str(run_dir),
        "resolved_config_path": str(result["resolved_config_path"]),
        "best_weights_path": str(result["best_weights_path"]),
        "last_weights_path": str(result["last_weights_path"]),
        "disable_extra_geom_augs": bool(args.disable_extra_geom_augs),
    }
    metadata.update(collect_runtime_info())
    meta_json, meta_csv = save_rtdetr_thirdparty_run_metadata(metadata=metadata, out_dir=eval_dir)
    print(f"Saved run metadata   -> {meta_json}")
    print(f"Saved metadata table -> {meta_csv}")

    # Store raw adapter return for quick debugging/repro.
    raw_json = eval_dir / "train_adapter_result.json"
    raw_json.write_text(json.dumps(result, indent=2))
    print(f"Saved adapter output -> {raw_json}")


if __name__ == "__main__":
    main()

