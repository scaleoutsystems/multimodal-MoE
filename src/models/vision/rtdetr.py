"""
RT-DETR integration layer for Ultralytics.

This file is where we put RT-DETR-specific train/eval calls
(for example: which Ultralytics class we instantiate and how we call .train/.val).

Why this file exists:
- keep backend-specific logic out of top-level scripts
- make RT-DETR easy to maintain separately from YOLO
- still reuse shared artifact helpers so all model families write comparable outputs

Includes the following pieces:
- `RtdetrTrainConfig`: stores RT-DETR train settings in one place.
- `train_rtdetr_detector(...)`: launches Ultralytics RT-DETR training.
- `eval_rtdetr_detector(...)`: runs validation/test evaluation for saved RT-DETR weights.
- `save_rtdetr_metrics_json(...)`: writes evaluation metrics in the shared report format so we can compare RT-DETR and YOLO side by side.
- `save_rtdetr_training_summary(...)`: writes training summary artifacts in the shared report format for the same cross-family comparison workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

# Reuse shared artifact helpers so RT-DETR and YOLO produce the same schema.
from src.models.vision.yolo import (
    infer_model_variant_from_weights,
    save_metrics_table_csv,
    save_run_metadata_artifacts,
    save_yolo_metrics_json,
    save_yolo_training_summary,
)


@dataclass
class RtdetrTrainConfig:
    data_yaml: str
    model: str = "rtdetr-l.pt"
    imgsz: Union[int, tuple[int, int]] = (704, 1248)
    epochs: int = 50
    patience: int = 100
    batch: int = 16
    device: str = "0"
    project: str = "outputs/runs/rtdetr"
    name: str = "baseline"
    seed: int = 0
    workers: int = 8


def _import_ultralytics_rtdetr():
    """
    Import Ultralytics RTDETR lazily.

    We keep this lazy so importing unrelated scripts doesn't require
    Ultralytics unless we actually run RT-DETR.
    """
    try:
        from ultralytics import RTDETR  # type: ignore
    except Exception as e:
        raise ImportError(
            "Ultralytics is required for RT-DETR adapter. Install with `pip install ultralytics`."
        ) from e
    return RTDETR


def _format_ultralytics_imgsz(imgsz: Union[int, tuple[int, int]]):
    """
    Convert our image-size config to Ultralytics argument format.
    """
    if isinstance(imgsz, tuple):
        h, w = int(imgsz[0]), int(imgsz[1])
        return [h, w]
    return int(imgsz)


def train_rtdetr_detector(cfg: RtdetrTrainConfig):
    """
    Train an RT-DETR detector with Ultralytics.
    """
    RTDETR = _import_ultralytics_rtdetr()
    model = RTDETR(cfg.model)
    results = model.train(
        data=cfg.data_yaml,
        imgsz=_format_ultralytics_imgsz(cfg.imgsz),
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        seed=cfg.seed,
        workers=cfg.workers,
    )
    return results


def eval_rtdetr_detector(
    data_yaml: str,
    weights_path: str,
    split: str = "val",
    imgsz: Union[int, tuple[int, int]] = (704, 1248),
    batch: int = 16,
    device: str = "0",
    project: str | None = None,
    name: str | None = None,
):
    """
    Run RT-DETR evaluation (val/test/train split) using trained weights.
    """
    RTDETR = _import_ultralytics_rtdetr()
    model = RTDETR(weights_path)

    val_kwargs = {}
    if project:
        val_kwargs["project"] = project
    if name:
        val_kwargs["name"] = name

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=_format_ultralytics_imgsz(imgsz),
        batch=batch,
        device=device,
        **val_kwargs,
    )
    return metrics


def save_rtdetr_metrics_json(metrics, out_path: str | Path) -> Path:
    """
    Persist RT-DETR eval metrics to JSON using the same schema as YOLO.

    I intentionally reuse the YOLO serializer so comparison/report scripts
    can read one consistent metrics format across model families.
    """
    return save_yolo_metrics_json(metrics=metrics, out_path=out_path)


def save_rtdetr_training_summary(
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
    Persist RT-DETR train summary artifacts using the same schema as YOLO.
    """
    return save_yolo_training_summary(
        train_wall_time_s=train_wall_time_s,
        model_name=model_name,
        data_yaml=data_yaml,
        run_name=run_name,
        out_json_path=out_json_path,
        out_csv_path=out_csv_path,
        results=results,
    )


def get_rtdetr_model_size_stats_from_weights(weights_path: str) -> dict:
    """
    Load RT-DETR weights and return best-effort model size stats.
    """
    RTDETR = _import_ultralytics_rtdetr()
    model = RTDETR(weights_path)
    stats = {
        "params_total": None,
        "params_trainable": None,
        "flops_g": None,
    }
    if model is None or not hasattr(model, "model"):
        return stats

    pt_model = model.model
    try:
        stats["params_total"] = int(sum(p.numel() for p in pt_model.parameters()))
        stats["params_trainable"] = int(sum(p.numel() for p in pt_model.parameters() if p.requires_grad))
    except Exception:
        pass
    for attr in ("flops", "flops_g", "GFLOPs"):
        if hasattr(pt_model, attr):
            try:
                stats["flops_g"] = float(getattr(pt_model, attr))
                break
            except Exception:
                pass
    return stats
