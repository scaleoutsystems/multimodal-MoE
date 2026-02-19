"""
Evaluate detector runs.

Current backend support:
- yolo (Ultralytics)
- rtdetr (Ultralytics)

The CLI name stays generic so we can add DINO/fusion backends later without
changing downstream workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import platform
import socket

# Allow running as either:
# - python -m scripts.eval_detector
# - python scripts/eval_detector.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vision.yolo import (
    eval_yolo_detector,  # Returns Ultralytics metrics object from model.val().
    save_yolo_metrics_json,  # Returns written metrics.json Path.
    save_metrics_table_csv,  # Returns written 2-column metrics CSV Path. Generic, used for all model families.
    save_run_metadata_artifacts,  # Returns (metadata_json_path, metadata_csv_path). Generic, used for all model families.
    infer_model_variant_from_weights,  # Returns model variant string (weights stem). Generic, used for all model families.
    get_yolo_model_size_stats_from_weights,  # Returns best-effort params/FLOPs from YOLO weights.
)
from src.models.vision.rtdetr import (
    eval_rtdetr_detector,  # Returns Ultralytics metrics object from model.val().
    save_rtdetr_metrics_json,  # Returns written metrics.json Path.
    get_rtdetr_model_size_stats_from_weights,  # Returns best-effort params/FLOPs from RT-DETR weights.
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
    parser.add_argument("--backend", choices=["yolo", "rtdetr"], default="yolo")
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


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _add_derived_speed_metrics(metrics_dict: dict) -> dict:
    """
    Add thesis-friendly derived throughput metrics.
    """
    preprocess_ms = _safe_float(metrics_dict.get("speed_preprocess_ms_per_img"))
    inference_ms = _safe_float(metrics_dict.get("speed_inference_ms_per_img"))
    postprocess_ms = _safe_float(metrics_dict.get("speed_postprocess_ms_per_img"))

    if inference_ms is not None and inference_ms > 0:
        metrics_dict["fps_inference_only"] = 1000.0 / inference_ms

    if preprocess_ms is not None and inference_ms is not None and postprocess_ms is not None:
        total_ms = preprocess_ms + inference_ms + postprocess_ms
        metrics_dict["speed_total_ms_per_img"] = total_ms
        if total_ms > 0:
            metrics_dict["fps_end_to_end"] = 1000.0 / total_ms

    return metrics_dict


def _collect_runtime_info() -> dict:
    """
    Collect lightweight environment info for reproducibility.
    """
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = str(torch.version.cuda)
        info["cudnn_version"] = int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(float(props.total_memory) / (1024**3), 3)
    except Exception:
        pass
    return info


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
        metrics_dict = _add_derived_speed_metrics(metrics_dict)
        # Fallback: some Ultralytics versions do not expose model size stats in eval object.
        if metrics_dict.get("params_total") is None and metrics_dict.get("flops_g") is None:
            try:
                metrics_dict.update(get_yolo_model_size_stats_from_weights(args.weights))
            except Exception:
                pass
        out_json.write_text(json.dumps(metrics_dict, indent=2))
        out_csv = save_metrics_table_csv(metrics_dict, out_dir / "metrics_table.csv")
        data_yaml_path = Path(args.data_yaml)
        dataset_export_name = data_yaml_path.parent.name if data_yaml_path.name == "dataset.yaml" else data_yaml_path.stem
        weights_path = Path(args.weights)
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
            "weights_file_size_mb": round(weights_path.stat().st_size / (1024**2), 3) if weights_path.exists() else None,
        }
        metadata.update(_collect_runtime_info())
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

    if args.backend == "rtdetr":
        # We reuse the same Ultralytics dataset.yaml + label format here.
        metrics = eval_rtdetr_detector(
            data_yaml=args.data_yaml,
            weights_path=args.weights,
            split=args.split,
            imgsz=(args.img_h, args.img_w),
            batch=args.batch,
            device=args.device,
            project=str(RUNS_DIR / "rtdetr"),
            name=f"{args.run_name}_val",
        )
        out_json = save_rtdetr_metrics_json(metrics=metrics, out_path=out_dir / "metrics.json")
        metrics_dict = json.loads(out_json.read_text())
        metrics_dict = _add_derived_speed_metrics(metrics_dict)
        if metrics_dict.get("params_total") is None and metrics_dict.get("flops_g") is None:
            try:
                metrics_dict.update(get_rtdetr_model_size_stats_from_weights(args.weights))
            except Exception:
                pass
        out_json.write_text(json.dumps(metrics_dict, indent=2))
        out_csv = save_metrics_table_csv(metrics_dict, out_dir / "metrics_table.csv")
        data_yaml_path = Path(args.data_yaml)
        dataset_export_name = data_yaml_path.parent.name if data_yaml_path.name == "dataset.yaml" else data_yaml_path.stem
        weights_path = Path(args.weights)
        metadata = {
            "model_family": "rtdetr",
            "model_variant": infer_model_variant_from_weights(args.weights),
            "model_weights": args.weights,
            "run_name": args.run_name,
            "seed": int(args.seed),
            "split": args.split,
            "img_h": int(args.img_h),
            "img_w": int(args.img_w),
            "unclear_policy": args.unclear_policy,
            "dataset_export_name": dataset_export_name,
            "data_yaml": str(data_yaml_path),
            "weights_file_size_mb": round(weights_path.stat().st_size / (1024**2), 3) if weights_path.exists() else None,
        }
        metadata.update(_collect_runtime_info())
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
