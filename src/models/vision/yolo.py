"""
YOLO adapter wrapper (Ultralytics-specific).

This contains functions to train/evaluate YOLO and to save
run artifacts (metrics, summaries, and metadata) in a stable
format for cross-model benchmark comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union
import csv
import json
import numpy as np


@dataclass
class YoloTrainConfig:
    data_yaml: str
    model: str = "yolo26s.pt" # default baseline is YOLO26s; other variants can still be passed via CLI.
    imgsz: Union[int, tuple[int, int]] = (704, 1248)
    rect: bool = True # enable training on rectangular images (instead of square images)
    epochs: int = 50
    patience: int = 100
    batch: int = 16
    device: str = "0" # GPU 0 is used by default.
    project: str = "outputs/runs/yolo"
    name: str = "baseline"
    seed: int = 0
    workers: int = 8
    # Strict natural-geometry defaults (no random zoom/shift).
    scale: float = 0.0
    translate: float = 0.0
    mosaic: float = 0.0
    close_mosaic: int = 0


def _import_ultralytics_yolo():
    """
    Import Ultralytics YOLO lazily.

    Input:
        None.

    Output:
        YOLO class from ultralytics package.

    Why:
        Keeps import errors localized to YOLO workflows and avoids forcing
        unrelated scripts to require the ultralytics dependency.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise ImportError(
            "Ultralytics is required for YOLO adapter. Install with `pip install ultralytics`."
        ) from e
    return YOLO


def train_yolo_detector(cfg: YoloTrainConfig):
    """
    Train a YOLO detector with Ultralytics.

    Input:
        cfg: YoloTrainConfig containing data/model/training/runtime settings.

    Output:
        Ultralytics training results object.

    Why:
        Provides a single YOLO-specific adapter entrypoint for training.
    """
    YOLO = _import_ultralytics_yolo()
    model = YOLO(cfg.model)
    results = model.train(
        data=cfg.data_yaml,
        imgsz=_format_ultralytics_imgsz(cfg.imgsz),
        rect=cfg.rect,
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        seed=cfg.seed,
        workers=cfg.workers,
        scale=cfg.scale,
        translate=cfg.translate,
        mosaic=cfg.mosaic,
        close_mosaic=cfg.close_mosaic,
    )
    return results


def _extract_yolo_model_size_stats(model_obj) -> dict:
    """
    Best-effort extraction of model size stats from a YOLO model wrapper.
    """
    stats = {
        "params_total": None,
        "params_trainable": None,
        "flops_g": None,
    }
    if model_obj is None or not hasattr(model_obj, "model"):
        return stats

    pt_model = model_obj.model
    try:
        stats["params_total"] = int(sum(p.numel() for p in pt_model.parameters()))
        stats["params_trainable"] = int(sum(p.numel() for p in pt_model.parameters() if p.requires_grad))
    except Exception:
        pass

    # FLOPs may or may not be exposed depending on Ultralytics version/model state.
    for attr in ("flops", "flops_g", "GFLOPs"):
        if hasattr(pt_model, attr):
            try:
                stats["flops_g"] = float(getattr(pt_model, attr))
                break
            except Exception:
                pass
    return stats


def eval_yolo_detector(
    data_yaml: str,
    weights_path: str,
    split: str = "val",
    imgsz: Union[int, tuple[int, int]] = (704, 1248),
    rect: bool = True,
    batch: int = 16,
    device: str = "0",
    project: str | None = None,
    name: str | None = None,
    ):
    """
    Run YOLO evaluation (val/test/train split) using trained weights.

    Input:
        data_yaml: Dataset YAML path.
        weights_path: Path to trained YOLO weights (.pt).
        split: Dataset split to evaluate.
        imgsz, rect, batch, device: Runtime evaluation settings.
        project, name: Optional Ultralytics output location settings for val artifacts.

    Output:
        Ultralytics metrics object.

    Why:
        Keeps evaluation backend-specific logic inside the YOLO adapter.
    """
    YOLO = _import_ultralytics_yolo()
    model = YOLO(weights_path)
    val_kwargs = {}
    if project:
        val_kwargs["project"] = project
    if name:
        val_kwargs["name"] = name

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=_format_ultralytics_imgsz(imgsz),
        rect=rect,
        batch=batch,
        device=device,
        **val_kwargs,
    )
    return metrics


def _format_ultralytics_imgsz(imgsz: Union[int, tuple[int, int]]):
    """
    Convert project-friendly image size config to Ultralytics argument format.
    """
    if isinstance(imgsz, tuple):
        h, w = int(imgsz[0]), int(imgsz[1])
        return [h, w]
    return int(imgsz)


def save_yolo_metrics_json(metrics, out_path: str | Path) -> Path:
    """
    Persist key YOLO validation metrics to JSON for experiment tracking.

    Input:
        metrics: Ultralytics metrics object returned by model.val().
        out_path: Target JSON path.

    Output:
        Path to written JSON file.

    Why:
        We need stable, lightweight metrics artifacts for comparisons across
        experiments and later reporting.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    candidates = {
        "map50": "metrics/mAP50(B)", #IOU threshold of 0.5
        "map50_95": "metrics/mAP50-95(B)", #IOU threshold of 0.5-0.95
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
    }

    # Ultralytics results implement dict-like APIs via .results_dict in many versions.
    if hasattr(metrics, "results_dict"):
        raw = dict(metrics.results_dict)
        for out_key, raw_key in candidates.items():
            if raw_key in raw:
                serializable[out_key] = float(raw[raw_key])

    # Fallbacks for older/newer APIs.
    if not serializable and hasattr(metrics, "box"):
        box = metrics.box
        for out_key, attr in [
            ("map50", "map50"),
            ("map50_95", "map"),
            ("precision", "mp"),
            ("recall", "mr"),
        ]:
            if hasattr(box, attr):
                serializable[out_key] = float(getattr(box, attr))

    # Inference speed (ms/image) if available.
    if hasattr(metrics, "speed") and isinstance(metrics.speed, dict):
        for k, v in metrics.speed.items():
            try:
                serializable[f"speed_{k}_ms_per_img"] = float(v)
            except Exception:
                continue

    # Params/FLOPs (best-effort) from trained model wrapper.
    stats = _extract_yolo_model_size_stats(getattr(metrics, "model", None))
    serializable.update(stats)

    def _to_1d_float_list(values) -> list[float]:
        """
        Convert curve payloads (list/ndarray/tensor-like) to a 1D float list.
        """
        try:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return []
            if arr.ndim == 1:
                return [float(v) for v in arr.tolist()]
            # Common case in detection metrics: class-wise arrays (C, N).
            # For single-class this becomes (1, N); for multi-class we use mean over classes.
            if arr.ndim >= 2:
                if arr.shape[0] == 1:
                    arr = arr[0]
                else:
                    arr = arr.mean(axis=0)
                arr = np.asarray(arr, dtype=float).reshape(-1)
                return [float(v) for v in arr.tolist()]
        except Exception:
            pass
        # Last-resort fallback for odd iterables.
        try:
            return [float(v) for v in list(values)]
        except Exception:
            return []

    # Best-effort curve extraction for PR-style analysis.
    # Ultralytics APIs vary by version, so we keep this defensive.
    try:
        box = getattr(metrics, "box", None)
        if box is not None and hasattr(box, "curves_results"):
            curves_results = getattr(box, "curves_results")
            curve_names = []
            if hasattr(box, "curves"):
                raw_curve_names = getattr(box, "curves")
                if isinstance(raw_curve_names, (list, tuple)):
                    curve_names = [str(name) for name in raw_curve_names]
            if isinstance(curves_results, (list, tuple)):
                serializable["curves_results"] = []
                for i, item in enumerate(curves_results):
                    try:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            x, y = item[0], item[1]
                            x_list = _to_1d_float_list(x)
                            y_list = _to_1d_float_list(y)
                            if len(x_list) == 0 or len(y_list) == 0:
                                continue
                            # Align lengths defensively if backend returns slight shape mismatch.
                            if len(x_list) != len(y_list):
                                n = min(len(x_list), len(y_list))
                                x_list = x_list[:n]
                                y_list = y_list[:n]
                            if len(x_list) == 0:
                                continue
                            curve_entry = {"x": x_list, "y": y_list}
                            if i < len(curve_names):
                                curve_entry["name"] = curve_names[i]
                            serializable["curves_results"].append(curve_entry)
                    except Exception:
                        continue
    except Exception:
        pass

    out_path.write_text(json.dumps(serializable, indent=2))
    return out_path


def save_metrics_table_csv(metrics_dict: dict, out_path: str | Path) -> Path:
    """
    Save a flat metrics dictionary as a 2-column CSV table: metric,value.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in sorted(metrics_dict.keys()):
            writer.writerow([k, metrics_dict[k]])
    return out_path

def infer_model_variant_from_weights(weights_name: str) -> str:
    """
    Infer a compact model variant label from a weights filename.
    Example: 'yolo26n.pt' -> 'yolo26n'
    """
    return Path(weights_name).stem


def save_run_metadata_artifacts(
    metadata: dict,
    out_json_path: str | Path,
    out_csv_path: str | Path,
) -> tuple[Path, Path]:
    """
    Save run metadata as JSON + 2-column CSV table.
    """
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(metadata, indent=2))

    out_csv_path = save_metrics_table_csv(metadata, out_csv_path)
    return out_json_path, out_csv_path


def save_yolo_training_summary(
    *,
    train_wall_time_s: float,
    model_name: str,
    data_yaml: str,
    run_name: str,
    out_json_path: str | Path,
    out_csv_path: str | Path,
    results=None,
) -> tuple[Path, Path]:
    """
    Save training summary artifacts (JSON + CSV table).

    Includes wall-clock training time and model size stats (best-effort).
    """
    summary = {
        "model_name": model_name,
        "data_yaml": data_yaml,
        "run_name": run_name,
        "train_wall_time_s": float(train_wall_time_s),
    }
    stats = _extract_yolo_model_size_stats(getattr(results, "model", None))
    summary.update(stats)

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(summary, indent=2))

    out_csv_path = save_metrics_table_csv(summary, out_csv_path)
    return out_json_path, out_csv_path


def get_yolo_model_size_stats_from_weights(weights_path: str) -> dict:
    """
    Load YOLO weights and return best-effort model size stats.

    I use this as a fallback during eval when metrics objects don't expose
    params/FLOPs directly.
    """
    YOLO = _import_ultralytics_yolo()
    model = YOLO(weights_path)
    return _extract_yolo_model_size_stats(model)
