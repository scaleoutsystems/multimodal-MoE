"""
RT-DETRv2 third-party adapter (official PyTorch repo).

This module keeps third-party invocation logic out of top-level scripts and
normalizes outputs into the same artifact schema used by other backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import re
import socket
import subprocess
import time
from typing import Any

from src.models.vision.yolo import save_metrics_table_csv, save_run_metadata_artifacts


@dataclass
class RtdetrThirdPartyTrainConfig:
    base_config: str
    train_img_dir: str
    train_ann_json: str
    val_img_dir: str
    val_ann_json: str
    output_dir: str
    run_name: str
    imgsz: tuple[int, int] = (704, 1248)
    epochs: int = 50
    batch: int = 16
    device: str = "cuda:0"
    seed: int = 0
    workers: int = 8
    num_classes: int = 1
    use_amp: bool = True


def _repo_root() -> Path:
    # src/models/vision/rtdetr_thirdparty.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def _third_party_rtdetrv2_root() -> Path:
    root = _repo_root() / "third_party" / "rtdetr" / "rtdetrv2_pytorch"
    if not root.exists():
        raise FileNotFoundError(f"RT-DETRv2 repo not found: {root}")
    return root


def _write_runtime_config(
    *,
    base_config: str,
    out_path: Path,
    train_img_dir: str,
    train_ann_json: str,
    val_img_dir: str,
    val_ann_json: str,
    output_dir: str,
    img_h: int,
    img_w: int,
    epochs: int,
    batch: int,
    workers: int,
    num_classes: int,
) -> Path:
    """
    Write a tiny override config that includes a model config and replaces dataset/runtime knobs.
    """
    config_obj: dict[str, Any] = {
        "__include__": [str(Path(base_config).resolve())],
        "output_dir": str(Path(output_dir).resolve()),
        "epoches": int(epochs),  # NOTE: upstream key is intentionally "epoches".
        "num_classes": int(num_classes),
        "remap_mscoco_category": False,
        "eval_spatial_size": [int(img_h), int(img_w)],
        "train_dataloader": {
            "dataset": {
                "img_folder": str(Path(train_img_dir).resolve()),
                "ann_file": str(Path(train_ann_json).resolve()),
                # Force rectangular final resize to match project fairness policy.
                "transforms": {
                    "ops": [
                        {"type": "RandomPhotometricDistort", "p": 0.5},
                        {"type": "RandomHorizontalFlip"},
                        {"type": "Resize", "size": [int(img_h), int(img_w)]},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                },
            },
            # Disable square multiscale collate behavior from upstream include.
            "collate_fn": {"type": "BatchImageCollateFunction"},
            "total_batch_size": int(batch),
            "num_workers": int(workers),
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(Path(val_img_dir).resolve()),
                "ann_file": str(Path(val_ann_json).resolve()),
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [int(img_h), int(img_w)]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ]
                },
            },
            "total_batch_size": int(batch),
            "num_workers": int(workers),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config_obj, indent=2))
    return out_path


def _run_subprocess(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _parse_coco_summary_from_stdout(stdout: str) -> dict[str, float | None]:
    """
    Parse AP/AR values from COCO summary lines printed by upstream eval.
    """
    metrics: dict[str, float | None] = {
        "map50_95": None,
        "map50": None,
        "precision": None,
        "recall": None,
    }

    patterns = {
        "map50_95": r"Average Precision\s+\(AP\)\s+@\[ IoU=0\.50:0\.95 .* =\s+([0-9.]+)",
        "map50": r"Average Precision\s+\(AP\)\s+@\[ IoU=0\.50\s+\|.* =\s+([0-9.]+)",
        "recall": r"Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 .* maxDets=\s*100 \]\s*=\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except Exception:
                metrics[key] = None
    return metrics


def _collect_runtime_info() -> dict:
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


def train_rtdetr_thirdparty(cfg: RtdetrThirdPartyTrainConfig) -> dict[str, Any]:
    repo_root = _third_party_rtdetrv2_root()
    run_dir = Path(cfg.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_path = run_dir / "resolved_config.yml"
    _write_runtime_config(
        base_config=cfg.base_config,
        out_path=resolved_cfg_path,
        train_img_dir=cfg.train_img_dir,
        train_ann_json=cfg.train_ann_json,
        val_img_dir=cfg.val_img_dir,
        val_ann_json=cfg.val_ann_json,
        output_dir=str(run_dir),
        img_h=int(cfg.imgsz[0]),
        img_w=int(cfg.imgsz[1]),
        epochs=int(cfg.epochs),
        batch=int(cfg.batch),
        workers=int(cfg.workers),
        num_classes=int(cfg.num_classes),
    )

    command = [
        "python",
        "tools/train.py",
        "-c",
        str(resolved_cfg_path),
        "-d",
        cfg.device,
        "--seed",
        str(cfg.seed),
        "--output-dir",
        str(run_dir),
    ]
    if cfg.use_amp:
        command.append("--use-amp")

    t0 = time.perf_counter()
    proc = _run_subprocess(command=command, cwd=repo_root)
    elapsed_s = time.perf_counter() - t0

    (run_dir / "stdout.log").write_text(proc.stdout or "")
    (run_dir / "stderr.log").write_text(proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(
            "RT-DETR third-party training failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {proc.returncode}\n"
            f"See logs: {run_dir / 'stdout.log'} and {run_dir / 'stderr.log'}"
        )

    return {
        "run_dir": str(run_dir),
        "resolved_config_path": str(resolved_cfg_path),
        "best_weights_path": str(run_dir / "best.pth"),
        "last_weights_path": str(run_dir / "last.pth"),
        "train_wall_time_s": float(elapsed_s),
    }


def eval_rtdetr_thirdparty(
    *,
    base_config: str,
    weights_path: str,
    val_img_dir: str,
    val_ann_json: str,
    output_dir: str,
    split: str = "val",
    imgsz: tuple[int, int] = (704, 1248),
    batch: int = 16,
    device: str = "cuda:0",
    workers: int = 8,
    num_classes: int = 1,
) -> dict[str, Any]:
    """
    Run third-party RT-DETR evaluation in test-only mode and return normalized metrics.

    NOTE:
    - Upstream test-only path prints COCO summary to stdout. We parse AP/AR from that output.
    - The COCO-style AP metrics are the primary cross-family comparison metrics.
    """
    if split != "val":
        raise ValueError("Third-party v1 adapter currently supports split='val' only.")

    repo_root = _third_party_rtdetrv2_root()
    eval_dir = Path(output_dir).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_path = eval_dir / "resolved_eval_config.yml"
    _write_runtime_config(
        base_config=base_config,
        out_path=resolved_cfg_path,
        train_img_dir=val_img_dir,
        train_ann_json=val_ann_json,
        val_img_dir=val_img_dir,
        val_ann_json=val_ann_json,
        output_dir=str(eval_dir),
        img_h=int(imgsz[0]),
        img_w=int(imgsz[1]),
        epochs=1,
        batch=int(batch),
        workers=int(workers),
        num_classes=int(num_classes),
    )

    command = [
        "python",
        "tools/train.py",
        "-c",
        str(resolved_cfg_path),
        "-r",
        str(Path(weights_path).resolve()),
        "-d",
        device,
        "--test-only",
        "--output-dir",
        str(eval_dir),
    ]

    t0 = time.perf_counter()
    proc = _run_subprocess(command=command, cwd=repo_root)
    elapsed_s = time.perf_counter() - t0

    (eval_dir / "stdout_eval.log").write_text(proc.stdout or "")
    (eval_dir / "stderr_eval.log").write_text(proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            "RT-DETR third-party eval failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {proc.returncode}\n"
            f"See logs: {eval_dir / 'stdout_eval.log'} and {eval_dir / 'stderr_eval.log'}"
        )

    metrics = _parse_coco_summary_from_stdout(proc.stdout or "")
    metrics.update(
        {
            "split": split,
            "speed_total_s_eval_run": float(elapsed_s),
            "speed_total_ms_per_img": None,
            "fps_end_to_end": None,
            "params_total": None,
            "params_trainable": None,
            "flops_g": None,
        }
    )
    return metrics


def save_rtdetr_thirdparty_metrics_json(metrics: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    return out_path


def save_rtdetr_thirdparty_training_summary(
    *,
    run_name: str,
    model_name: str,
    base_config: str,
    train_wall_time_s: float,
    out_json_path: str | Path,
    out_csv_path: str | Path,
) -> tuple[Path, Path]:
    summary = {
        "run_name": run_name,
        "model_name": model_name,
        "base_config": str(base_config),
        "train_wall_time_s": float(train_wall_time_s),
    }
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(summary, indent=2))
    out_csv_path = save_metrics_table_csv(summary, out_csv_path)
    return out_json_path, out_csv_path


def save_rtdetr_thirdparty_run_metadata(
    *,
    metadata: dict[str, Any],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    return save_run_metadata_artifacts(
        metadata=metadata,
        out_json_path=out_dir / "run_metadata.json",
        out_csv_path=out_dir / "run_metadata.csv",
    )


def collect_runtime_info() -> dict:
    return _collect_runtime_info()

