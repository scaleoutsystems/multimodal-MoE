#!/usr/bin/env python3
"""run_camera_ablation_eval.py — Camera contribution ablation for BEVFusion.

Evaluates a BEVFusion checkpoint on the main test split under two conditions:

  1. full_model   — normal forward pass (camera + LiDAR)
  2. camera_zero  — camera BEV replaced with zeros before fusion
                    (LiDAR-only contribution inside the trained network)

Both conditions use the same checkpoint weights.  The camera-zero condition
loads the checkpoint into ``BEVFusionCameraZero`` (see
projects/BEVFusion/bevfusion/bevfusion_camera_zero.py), which overrides
``extract_img_feat`` to return zeros while keeping all learned weights.

Outputs (relative to the checkpoint's run directory):
  ablation/full_model/aggregated_results.json
  ablation/full_model/aggregated_results.md
  ablation/camera_zero/aggregated_results.json
  ablation/camera_zero/aggregated_results.md
  camera_ablation.md          ← side-by-side comparison (main artefact)

USAGE
-----
From the repo root (multimodal-MoE/):

  python scripts/bev_scripts/run_camera_ablation_eval.py \\
      mmdetection3d/configs/zod/zod_bevfusion_finetune.py \\
      mmdetection3d/configs/zod/zod_bevfusion_finetune_camzero.py \\
      outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4456392/best_mAP_0.50_epoch_12.pth \\
      --splits-root /mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/splits

Dry-run (print commands, skip subprocess execution):

  python scripts/bev_scripts/run_camera_ablation_eval.py ... --dry-run

SBATCH submission (see meluxina_bevfusion_camera_ablation.sbatch).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo-root and helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "bev_scripts" / "run_multi_split_eval.py"

# Only the main test split — not the context splits.
MAIN_TEST_SPLIT = "test.txt"


def infer_run_dir(ckpt: Path) -> Path:
    """Return the run directory that contains the checkpoint.

    For a checkpoint at ``.../runs/foo/bar/epoch_N.pth`` this returns
    ``.../runs/foo/bar/``.
    """
    return ckpt.resolve().parent


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_eval(
    *,
    config: Path,
    ckpt: Path,
    out_dir: Path,
    splits_root: Path,
    infos_dir: Optional[Path],
    python: str,
    launcher: str,
    dry_run: bool,
) -> int:
    """Call run_multi_split_eval.py for a single condition.

    Returns the subprocess exit code (0 = success).
    """
    cmd: List[str] = [
        python,
        str(EVAL_SCRIPT),
        str(config),
        str(ckpt),
        "--splits", MAIN_TEST_SPLIT,
        "--splits-root", str(splits_root),
        "--out-dir", str(out_dir),
        "--launcher", launcher,
        "--allow-missing-splits",
    ]
    if infos_dir is not None:
        cmd += ["--infos-dir", str(infos_dir)]

    print(f"\n{'=' * 70}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    if dry_run:
        print("  [DRY RUN] skipping subprocess.")
        return 0

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

def load_metrics(eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the flat metrics dict for the main test split from an eval dir."""
    json_path = eval_dir / "aggregated_results.json"
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    for split_record in data.get("flat_results", []):
        if split_record.get("split_name") == "test":
            return split_record.get("metrics", {})
    # Fallback: first record
    flat = data.get("flat_results", [])
    if flat:
        return flat[0].get("metrics", {})
    return None


# ---------------------------------------------------------------------------
# Markdown comparison writer
# ---------------------------------------------------------------------------

_PREFERRED_METRIC_ORDER = [
    "mAP_0.25",
    "mAP_0.50",
    "mAP_0.5m",
    "mAP_1.0m",
    "mAP_2.0m",
    "mAP_4.0m",
]


def _sort_metrics(keys: List[str]) -> List[str]:
    """Return metric keys sorted: preferred order first, then alphabetical."""
    preferred = [k for k in _PREFERRED_METRIC_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in preferred)
    return preferred + rest


def write_comparison_markdown(
    *,
    ckpt: Path,
    full_config: Path,
    zero_config: Path,
    full_metrics: Optional[Dict[str, Any]],
    zero_metrics: Optional[Dict[str, Any]],
    full_eval_dir: Path,
    zero_eval_dir: Path,
    out_path: Path,
) -> None:
    """Write the side-by-side comparison markdown to *out_path*."""

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        if v is None:
            return "—"
        return str(v)

    lines: List[str] = [
        "# Camera Contribution Ablation\n",
        f"**Checkpoint:** `{ckpt}`  ",
        f"**Full-model config:** `{full_config}`  ",
        f"**Camera-zero config:** `{zero_config}`  ",
        f"**Test split:** `test` (main test set only)  ",
        "",
        "## Method",
        "",
        "Two forward passes use the **same checkpoint weights**:",
        "",
        "| Condition | Description |",
        "| --- | --- |",
        "| **Full model** | Normal BEVFusion forward (camera + LiDAR) |",
        "| **Camera zeroed** | `extract_img_feat` returns `torch.zeros_like(cam_bev)` before fusion; LiDAR branch unchanged |",
        "",
        "The delta column (`Full − Zero`) estimates how much the camera branch",
        "contributes to each metric on top of the LiDAR signal already present",
        "in the checkpoint.",
        "",
        "## Results — main test set\n",
    ]

    all_metrics = set(full_metrics or {}) | set(zero_metrics or {})
    numeric_metrics = [
        k for k in all_metrics
        if isinstance((full_metrics or {}).get(k) or (zero_metrics or {}).get(k), float)
    ]
    sorted_keys = _sort_metrics(numeric_metrics)

    if not sorted_keys:
        lines.append("*No numeric metrics could be parsed from the eval outputs.*\n")
    else:
        header = "| Metric | Full model | Camera zeroed | Delta (Full−Zero) |"
        sep = "| --- | --- | --- | --- |"
        lines += [header, sep]

        for k in sorted_keys:
            full_v = (full_metrics or {}).get(k)
            zero_v = (zero_metrics or {}).get(k)
            if isinstance(full_v, float) and isinstance(zero_v, float):
                delta = full_v - zero_v
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "—"
            lines.append(
                f"| `{k}` | {_fmt(full_v)} | {_fmt(zero_v)} | {delta_str} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- A **positive delta** means the full model (with camera) outperforms",
        "  the camera-zeroed baseline, indicating a **positive camera contribution**.",
        "- A **near-zero or negative delta** suggests the camera branch is not yet",
        "  helpful for that metric, or is even interfering with the LiDAR signal.",
        "",
        "## Raw eval outputs",
        "",
        f"- Full model: `{full_eval_dir}`",
        f"- Camera zeroed: `{zero_eval_dir}`",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nComparison markdown written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Camera-contribution ablation: evaluate a BEVFusion checkpoint "
            "with and without camera BEV (test split only) and compare metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "full_config",
        help="Config for the full BEVFusion model (camera + LiDAR).",
    )
    p.add_argument(
        "zero_config",
        help="Config for the camera-zeroed variant (BEVFusionCameraZero).",
    )
    p.add_argument(
        "checkpoint",
        help="Checkpoint .pth file to evaluate (used for both conditions).",
    )
    p.add_argument(
        "--splits-root", type=Path, required=True,
        help="Directory containing split .txt files.",
    )
    p.add_argument(
        "--infos-dir", type=Path, default=None,
        help=(
            "Directory containing annotation pkl files. "
            "Defaults to {splits-root}/../infos."
        ),
    )
    p.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter. Default: sys.executable.",
    )
    p.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing subprocesses.",
    )
    p.add_argument(
        "--skip-full", action="store_true",
        help="Skip the full-model eval (use if results already exist).",
    )
    p.add_argument(
        "--skip-zero", action="store_true",
        help="Skip the camera-zero eval (use if results already exist).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = Path(args.checkpoint).resolve()
    full_config = Path(args.full_config)
    zero_config = Path(args.zero_config)
    run_dir = infer_run_dir(ckpt)

    ablation_root = run_dir / "ablation"
    full_eval_dir = ablation_root / "full_model"
    zero_eval_dir = ablation_root / "camera_zero"
    comparison_md = run_dir / "camera_ablation.md"

    print(f"Checkpoint     : {ckpt}")
    print(f"Run directory  : {run_dir}")
    print(f"Ablation root  : {ablation_root}")
    print(f"Full-model dir : {full_eval_dir}")
    print(f"Camera-zero dir: {zero_eval_dir}")
    print(f"Comparison MD  : {comparison_md}")

    # ── 1. Full-model eval ──────────────────────────────────────────────────
    if not args.skip_full:
        print(f"\n{'#' * 70}")
        print("# Step 1/2 — Full model eval (camera + LiDAR)")
        print(f"{'#' * 70}")
        rc = run_eval(
            config=full_config,
            ckpt=ckpt,
            out_dir=full_eval_dir,
            splits_root=args.splits_root,
            infos_dir=args.infos_dir,
            python=args.python,
            launcher=args.launcher,
            dry_run=args.dry_run,
        )
        if rc != 0:
            print(f"[ERROR] Full-model eval failed with exit code {rc}.")
            sys.exit(rc)
    else:
        print("\n[--skip-full] Skipping full-model eval.")

    # ── 2. Camera-zero eval ─────────────────────────────────────────────────
    if not args.skip_zero:
        print(f"\n{'#' * 70}")
        print("# Step 2/2 — Camera-zero eval (LiDAR-only contribution)")
        print(f"{'#' * 70}")
        rc = run_eval(
            config=zero_config,
            ckpt=ckpt,
            out_dir=zero_eval_dir,
            splits_root=args.splits_root,
            infos_dir=args.infos_dir,
            python=args.python,
            launcher=args.launcher,
            dry_run=args.dry_run,
        )
        if rc != 0:
            print(f"[ERROR] Camera-zero eval failed with exit code {rc}.")
            sys.exit(rc)
    else:
        print("\n[--skip-zero] Skipping camera-zero eval.")

    # ── 3. Load metrics and write comparison markdown ───────────────────────
    print(f"\n{'#' * 70}")
    print("# Writing comparison markdown")
    print(f"{'#' * 70}")

    full_metrics = load_metrics(full_eval_dir)
    zero_metrics = load_metrics(zero_eval_dir)

    if full_metrics is None and not args.dry_run and not args.skip_full:
        print("[WARN] Could not load full-model metrics from aggregated_results.json.")
    if zero_metrics is None and not args.dry_run and not args.skip_zero:
        print("[WARN] Could not load camera-zero metrics from aggregated_results.json.")

    write_comparison_markdown(
        ckpt=ckpt,
        full_config=full_config,
        zero_config=zero_config,
        full_metrics=full_metrics,
        zero_metrics=zero_metrics,
        full_eval_dir=full_eval_dir,
        zero_eval_dir=zero_eval_dir,
        out_path=comparison_md,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
