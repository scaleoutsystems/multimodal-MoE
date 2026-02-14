"""
YOLO adapter wrapper (Ultralytics-specific).

Keeping this in a dedicated adapter file avoids coupling the rest of the
project to one framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class YoloTrainConfig:
    data_yaml: str
    model: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 50
    batch: int = 16
    device: str = "0"
    project: str = "outputs/runs/yolo"
    name: str = "baseline"
    seed: int = 0
    workers: int = 8


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
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        seed=cfg.seed,
        workers=cfg.workers,
    )
    return results


def eval_yolo_detector(
    data_yaml: str,
    weights_path: str,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
):
    """
    Run YOLO evaluation (val/test/train split) using trained weights.

    Input:
        data_yaml: Dataset YAML path.
        weights_path: Path to trained YOLO weights (.pt).
        split: Dataset split to evaluate.
        imgsz, batch, device: Runtime evaluation settings.

    Output:
        Ultralytics metrics object.

    Why:
        Keeps evaluation backend-specific logic inside the YOLO adapter.
    """
    YOLO = _import_ultralytics_yolo()
    model = YOLO(weights_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )
    return metrics


def save_yolo_metrics_json(metrics, out_path: str | Path) -> Path:
    """
    Persist key YOLO metrics to JSON for experiment tracking.

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
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
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

    out_path.write_text(json.dumps(serializable, indent=2))
    return out_path
