#!/usr/bin/env python3
"""run_single_split_visual_eval.py – Qualitative evaluation with visualizations
on a single test split.

This script calls MMDetection3D's ``tools/test.py`` with visualization enabled
(--show-dir, --task, --score-thr) for one checkpoint on one split.

IMPORTANT NOTE ON VISUALIZATION OUTPUT
---------------------------------------
The exact visual content (point-cloud projections, BEV overlays, camera
image annotations, etc.) depends entirely on the VisualizationHook configured
in the MMDet3D config file and the --task flag.  This script simply activates
the hook via --show-dir; it does NOT guarantee any specific visual format.

OUTPUT LAYOUT
-------------
  outputs/runs/<run_name>/eval/
    visualizations/<split_name>/        saved visualization images from MMDet3D
    raw/<split_name>_visual/
      stdout.txt                        subprocess stdout
      stderr.txt                        subprocess stderr
      metrics.json                      parsed evaluation metrics

USAGE EXAMPLES
--------------
LiDAR-only model on the night lighting split:

  python scripts/bev_scripts/run_single_split_visual_eval.py \\
      mmdetection3d/configs/zod/zod_lidar_only.py \\
      outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth \\
      lighting_test_night.txt \\
      --splits-root /mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/splits \\
      --task lidar_det \\
      --score-thr 0.25

BEVFusion model on the full test split:

  python scripts/bev_scripts/run_single_split_visual_eval.py \\
      mmdetection3d/configs/zod/zod_bevfusion.py \\
      outputs/runs/zod_bevfusion/zod-bevfusion_4454826/best_mAP_0.50_epoch_10.pth \\
      test.txt \\
      --splits-root /mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/splits \\
      --task multi-modality_det \\
      --score-thr 0.3 \\
      --dry-run
SBATCH submission lidar only on the night lighting split:
CONFIG=mmdetection3d/configs/zod/zod_lidar_only.py \
CKPT=outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth \
SPLIT=lighting_test_day.txt \
SPLITS_ROOT=/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/splits \
TASK=lidar_det \
  sbatch mmdetection3d/tools/sbatch/meluxina_single_split_visual_eval.sbatch
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# This script lives at scripts/bev_scripts/; repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ANN_KEY = "test_dataloader.dataset.ann_file"
DEFAULT_ANN_PREFIX = "zod_nuscenes_infos"

VALID_TASKS = [
    "mono_det",
    "multi-view_det",
    "lidar_det",
    "lidar_seg",
    "multi-modality_det",
]


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from run_multi_split_eval for standalone use)
# ---------------------------------------------------------------------------

def infer_run_name(ckpt: Path) -> str:
    """Extract run name from checkpoint path (looks for segment after 'runs/')."""
    parts = ckpt.resolve().parts
    try:
        idx = parts.index("runs")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ckpt.stem


def infer_eval_dir(ckpt: Path, out_dir_override: Optional[Path] = None) -> Path:
    """Return the root eval output directory: outputs/runs/<run_name>/eval/"""
    if out_dir_override is not None:
        return out_dir_override
    return REPO_ROOT / "outputs" / "runs" / infer_run_name(ckpt) / "eval"


def resolve_split_file(split_arg: str, splits_root: Optional[Path]) -> Optional[Path]:
    """Resolve split argument to an existing path.

    1. If split_arg is already an existing path, return it.
    2. Try splits_root / split_arg.
    3. Return None if not found.
    """
    p = Path(split_arg)
    if p.exists():
        return p
    if splits_root is not None:
        q = splits_root / split_arg
        if q.exists():
            return q
    return None


def derive_ann_file(split_stem: str, infos_dir: Path, prefix: str) -> Path:
    """Build the absolute path to the annotation pkl for a split stem.

    e.g. "lighting_test_night" → {infos_dir}/zod_nuscenes_infos_lighting_test_night.pkl
    """
    return infos_dir / f"{prefix}_{split_stem}.pkl"


def build_cfg_options_arg(ann_key: str, ann_file: Path) -> str:
    """Return 'key=value' string for --cfg-options to override the ann_file.

    The intended common case is overriding test_dataloader.dataset.ann_file
    with an absolute path to the subset pkl.  Absolute paths are safe because
    os.path.join(data_root, abs_path) returns abs_path unchanged in Python.
    """
    return f"{ann_key}={ann_file}"


def parse_metrics_from_stdout(stdout_text: str) -> Optional[Dict[str, Any]]:
    """Extract the final metrics dict from MMEngine subprocess stdout.

    Handles both:
      Format A: INFO - {'key': value, ...}
      Format B: Epoch(test) [N/N]  key: value  key: value  ...
    """
    dict_line_re = re.compile(r"INFO\s*-\s*(\{.+\})\s*$", re.MULTILINE)
    for raw in reversed(dict_line_re.findall(stdout_text)):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict) and parsed:
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            pass

    epoch_re = re.compile(r"Epoch\(test\)\s+\[\d+/\d+\]\s+(.+)$", re.MULTILINE)
    for summary in reversed(epoch_re.findall(stdout_text)):
        pairs = re.findall(r"(\S+):\s+([\d.eE+\-]+)", summary)
        if pairs:
            return {k: float(v) for k, v in pairs}

    return None


def find_best_json(work_dir: Path) -> Optional[Path]:
    """Find the most likely metrics JSON in a test.py work dir.

    Prefers files with 'metric', 'eval', or 'result' in the name;
    falls back to the most recently modified JSON file found.
    """
    all_jsons = sorted(
        work_dir.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_jsons:
        return None
    for p in all_jsons:
        if any(kw in p.name.lower() for kw in ("metric", "eval", "result")):
            return p
    return all_jsons[0]


def load_metrics_from_json(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a JSON or JSONL file and return a metrics dict."""
    try:
        text = path.read_text()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1:
            for line in reversed(lines):
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and any(
                        isinstance(v, (int, float)) for v in rec.values()
                    ):
                        return rec
                except Exception:
                    pass
        return json.loads(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run MMDetection3D test.py on one split with visualization enabled "
            "and save both metrics and visual outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Positional
    p.add_argument("config", help="Path to MMDet3D config file (.py).")
    p.add_argument("checkpoint", help="Path to model checkpoint (.pth).")
    p.add_argument(
        "split",
        help=(
            "Split filename or full path (e.g. lighting_test_night.txt). "
            "Resolved relative to --splits-root if not already a valid path."
        ),
    )

    # Split resolution
    p.add_argument(
        "--splits-root", type=Path, default=None,
        help="Directory containing split .txt files.",
    )
    p.add_argument(
        "--infos-dir", type=Path, default=None,
        help=(
            "Directory containing annotation pkl files. "
            "Defaults to {splits-root}/../infos when --splits-root is set."
        ),
    )
    p.add_argument(
        "--ann-prefix", default=DEFAULT_ANN_PREFIX,
        help=f"Prefix for deriving pkl filename from split stem. Default: '{DEFAULT_ANN_PREFIX}'.",
    )
    p.add_argument(
        "--ann-key", default=DEFAULT_ANN_KEY,
        help=f"Dotted config key to override with the annotation pkl path. Default: '{DEFAULT_ANN_KEY}'.",
    )

    # Visualization control
    p.add_argument(
        "--task",
        choices=VALID_TASKS,
        required=True,
        help=(
            "Visualization task type (required for --show-dir to work). "
            "Must match the model architecture: lidar_det, multi-modality_det, etc."
        ),
    )
    p.add_argument(
        "--score-thr", type=float, default=0.1,
        help="Score threshold for visualization. Default: 0.1.",
    )
    p.add_argument(
        "--show-dir-name", default="visualizations",
        help=(
            "Base name of the visualization output subdirectory under eval/. "
            "Default: 'visualizations'."
        ),
    )

    # Output
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override the eval output directory (default: outputs/runs/<run_name>/eval/).",
    )

    # Subprocess control
    p.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter to use. Default: sys.executable.",
    )
    p.add_argument(
        "--test-script", type=Path,
        default=REPO_ROOT / "mmdetection3d" / "tools" / "test.py",
        help="Path to MMDetection3D tools/test.py.",
    )
    p.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Distributed launcher for test.py. Default: none.",
    )
    p.add_argument(
        "--extra-test-args", nargs="*", default=[],
        help="Additional arguments appended verbatim to the test.py invocation.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print command without executing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config = Path(args.config)
    ckpt = Path(args.checkpoint)
    run_name = infer_run_name(ckpt)
    eval_dir = infer_eval_dir(ckpt, args.out_dir)

    # Resolve infos directory
    infos_dir: Optional[Path] = args.infos_dir
    if infos_dir is None and args.splits_root is not None:
        infos_dir = args.splits_root.parent / "infos"

    # Resolve split file
    split_file = resolve_split_file(args.split, args.splits_root)
    if split_file is None:
        raise FileNotFoundError(
            f"Split not found: '{args.split}' (splits-root: {args.splits_root})"
        )
    split_stem = split_file.stem  # e.g. "lighting_test_night"

    # Derive annotation pkl
    if infos_dir is None:
        raise ValueError(
            "Cannot derive ann_file: --infos-dir is not set and "
            "--splits-root was not provided. Supply --infos-dir explicitly."
        )
    ann_file = derive_ann_file(split_stem, infos_dir, args.ann_prefix)
    if not ann_file.exists():
        raise FileNotFoundError(f"Ann pkl not found: {ann_file}")

    # Output directories
    # Visualizations go in: eval/visualizations/<split_name>/
    vis_dir = eval_dir / args.show_dir_name / split_stem
    # Logs and parsed metrics go in: eval/raw/<split_name>_visual/
    work_dir = eval_dir / "raw" / f"{split_stem}_visual"

    print(f"Run name   : {run_name}")
    print(f"Eval dir   : {eval_dir}")
    print(f"Split      : {split_stem}")
    print(f"Ann file   : {ann_file}")
    print(f"Task       : {args.task}")
    print(f"Score thr  : {args.score_thr}")
    print(f"Vis dir    : {vis_dir}")
    print(f"Work dir   : {work_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cfg_override = build_cfg_options_arg(args.ann_key, ann_file)

    # Build the test.py command.
    # We pass --show-dir (not --show) so output is saved to disk, not displayed
    # interactively.  This is important in batch/SLURM contexts.
    cmd: List[str] = [
        args.python,
        str(args.test_script),
        str(config),
        str(ckpt),
        "--work-dir", str(work_dir),
        "--launcher", args.launcher,
        "--cfg-options", cfg_override,
        "--show-dir", str(vis_dir),
        "--task", args.task,
        "--score-thr", str(args.score_thr),
    ] + list(args.extra_test_args or [])

    print(f"\n{'=' * 70}")
    print(f"  Cmd: {shlex.join(cmd)}")
    print(f"{'=' * 70}")

    if args.dry_run:
        print("\n[DRY RUN] skipping subprocess.")
        return

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    (work_dir / "stdout.txt").write_text(proc.stdout)
    (work_dir / "stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        print(
            f"\n[FAILED] exit code {proc.returncode}. "
            f"See {work_dir / 'stderr.txt'}"
        )
        sys.exit(proc.returncode)

    # Parse and save metrics
    metrics = parse_metrics_from_stdout(proc.stdout)
    if metrics is None:
        json_path = find_best_json(work_dir)
        if json_path:
            metrics = load_metrics_from_json(json_path)

    if metrics:
        metrics_path = work_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"\n[OK] {len(metrics)} metric key(s) parsed → {metrics_path}")
        print("\nMetrics:")
        for k, v in sorted(metrics.items()):
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k}: {fmt}")
    else:
        print(
            "\n[WARN] subprocess exited 0 but metrics could not be parsed. "
            f"Check {work_dir / 'stdout.txt'}"
        )

    print(f"\nVisualizations : {vis_dir}")
    print(f"Logs           : {work_dir}")


if __name__ == "__main__":
    main()
