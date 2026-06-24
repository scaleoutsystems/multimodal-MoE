#!/usr/bin/env python3
"""run_multi_split_eval.py – Quantitative evaluation of one checkpoint over
multiple test-split annotation files.

This script calls MMDetection3D's ``tools/test.py`` as a subprocess for each
split, captures its output, parses the reported metrics, and writes:

    outputs/runs/<run_name>/eval/
      raw/<split_name>/
        stdout.txt          captured subprocess stdout
        stderr.txt          captured subprocess stderr
        metrics.json        parsed metrics for this split
      aggregated_results.json
      aggregated_results.csv
      aggregated_results.md

The <run_name> is inferred from the checkpoint path automatically:
  outputs/runs/<run_name>/epoch_*.pth  →  <run_name>
  otherwise                            →  checkpoint file stem

HOW THE SPLIT OVERRIDE WORKS
-----------------------------
The intended common case for ZOD is overriding:
    test_dataloader.dataset.ann_file
with the absolute path to the per-subset .pkl file
(e.g. zod_nuscenes_infos_lighting_test_dat.pkl).

That is the only config change needed to switch the evaluation split; all
other config stays the same.  This is passed to test.py via --cfg-options.

USAGE EXAMPLES
--------------

Evaluate only the lighting splits:

  python scripts/bev_scripts/run_multi_split_eval.py \\
      mmdetection3d/configs/zod/zod_lidar_only.py \\
      outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth \\
      --splits lighting_test_day.txt lighting_test_night.txt lighting_test_day.txt \\
      --splits-root /mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits

Dry-run (print commands without executing):

  python scripts/bev_scripts/run_multi_split_eval.py ... --dry-run


SBATCH submission lidar only on all splits:
CONFIG=mmdetection3d/configs/zod/zod_lidar_only.py \
CKPT=outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth \
SPLITS_ROOT=/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits \
  sbatch mmdetection3d/tools/sbatch/meluxina_multi_split_eval.sbatch

SBATCH submission camera only on all splits:
CONFIG= mmdetection3d/configs/zod/zod_camera_only.py \
CKPT=outputs/runs/zod_camera_only/zod-cam-only_4469392/best_mAP_0.50_epoch_11.pth \
SPLITS_ROOT=/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits \
  sbatch mmdetection3d/tools/sbatch/meluxina_multi_split_eval.sbatch


SBATCH submission bevfusion_dual_initialization on all splits:
CONFIG=mmdetection3d/configs/zod/zod_bevfusion_dualinit.py \
CKPT=outputs/runs/zod_bevfusion_dualinit/bevfusion-dualinit_4481497/best_mAP_0.50_epoch_12.pth \
SPLITS_ROOT=/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits \
  sbatch mmdetection3d/tools/sbatch/meluxina_multi_split_eval.sbatch
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# This script lives at scripts/bev_scripts/; repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default config key used to override the annotation pkl file.
# This is the most common case for ZOD configs.  Override with --ann-key if
# your config uses a different nested path.
DEFAULT_ANN_KEY = "test_dataloader.dataset.ann_file"

# Prefix prepended to a split stem to form the pkl filename.
# e.g. "lighting_test_day" → "zod_nuscenes_infos_lighting_test_day.pkl"
DEFAULT_ANN_PREFIX = "zod_nuscenes_infos"

# Stable ordering of groups for all outputs.
GROUP_ORDER: List[str] = [
    "full_test",
    "complexity",
    "lighting",
    "road_type",
    "scraped_weather",
    "weather_group",
    "other",
]

GROUP_TITLE: Dict[str, str] = {
    "full_test": "Full test",
    "complexity": "Complexity",
    "lighting": "Lighting",
    "road_type": "Road type",
    "scraped_weather": "Scraped weather",
    "weather_group": "Weather group",
    "other": "Other",
}

# Default splits used when --splits is not provided.
# These are filenames relative to --splits-root.
DEFAULT_SPLITS: List[str] = [
    "test.txt",
    "complexity_test_low.txt",
    "complexity_test_medium.txt",
    "complexity_test_high.txt",
    "lighting_test_day.txt",
    "lighting_test_night.txt",
    "road_type_test_highway.txt",
    "road_type_test_smaller_rural.txt",
    "road_type_test_arterial_rural.txt",
    "road_type_test_arterial_urban.txt",
    "road_type_test_city.txt",
    "weather_test_clear_day.txt",
    "weather_test_clear_night.txt",
    "weather_test_cloudy.txt",
    "weather_test_partly_cloudy_day.txt",
    "weather_test_partly_cloudy_night.txt",
    "weather_test_fog.txt",
    "weather_test_precipitation.txt", # snow + rain
    "weather_group_test_clear_like.txt",
    "weather_group_test_cloud_like.txt",
    #"weather_group_test_precipitation.txt",
]


# ---------------------------------------------------------------------------
# Split classification helpers
# ---------------------------------------------------------------------------

def infer_group(split_name: str) -> str:
    """Map a split stem to a reporting group (first match wins).

      test                  → full_test
      complexity_test_*     → complexity
      lighting_test_*       → lighting
      road_type_test_*      → road_type
      weather_group_test_*  → weather_group   (must come before weather_test_*)
      weather_test_*        → scraped_weather
      anything else         → other
    """
    n = split_name
    if n == "test":
        return "full_test"
    if n.startswith("complexity_test_"):
        return "complexity"
    if n.startswith("lighting_test_"):
        return "lighting"
    if n.startswith("road_type_test_"):
        return "road_type"
    if n.startswith("weather_group_test_"):
        return "weather_group"
    if n.startswith("weather_test_"):
        return "scraped_weather"
    return "other"


def infer_label(split_name: str) -> str:
    """Strip the group prefix to produce a short human-readable label.

    Examples:
      lighting_test_day            → day
      complexity_test_high         → high
      weather_test_partly_cloudy_day → partly_cloudy_day
      road_type_test_highway       → highway
      test                         → test
    """
    for prefix in (
        "complexity_test_",
        "lighting_test_",
        "road_type_test_",
        "weather_group_test_",
        "weather_test_",
    ):
        if split_name.startswith(prefix):
            return split_name[len(prefix):]
    return split_name


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def infer_run_name(ckpt: Path) -> str:
    """Extract the run name from a checkpoint path.

    Looks for the segment immediately after 'runs/' in the resolved path.
    If not found, falls back to the checkpoint file stem.

    Example:
      outputs/runs/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth
      → zod-lidar-only_4454825
    """
    parts = ckpt.resolve().parts
    try:
        idx = parts.index("runs")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ckpt.stem


def infer_eval_dir(ckpt: Path, out_dir_override: Optional[Path] = None) -> Path:
    """Return the root evaluation output directory.

    Default layout: outputs/runs/<run_name>/eval/
    Pass out_dir_override to use a custom path instead.
    """
    if out_dir_override is not None:
        return out_dir_override
    run_name = infer_run_name(ckpt)
    return REPO_ROOT / "outputs" / "runs" / run_name / "eval"


def resolve_split_file(split_arg: str, splits_root: Optional[Path]) -> Optional[Path]:
    """Resolve a split argument to an existing Path.

    Resolution order:
      1. If split_arg is already an existing file path, return it as-is.
      2. If splits_root is set, try splits_root / split_arg.
      3. Return None if not found (caller decides whether to skip or abort).
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

    Example:
      stem="lighting_test_day", prefix="zod_nuscenes_infos"
      → {infos_dir}/zod_nuscenes_infos_lighting_test_day.pkl
    """
    return infos_dir / f"{prefix}_{split_stem}.pkl"


# ---------------------------------------------------------------------------
# Config-override helper
# ---------------------------------------------------------------------------

def build_cfg_options_arg(ann_key: str, ann_file: Path) -> str:
    """Return the key=value string for --cfg-options to override ann_file.

    The intended common case is:
        test_dataloader.dataset.ann_file=/abs/path/to/split.pkl

    MMEngine's DictAction parses this and sets the nested config value.
    Absolute paths are safe because Python's os.path.join(data_root, abs_path)
    returns abs_path, so the dataloader resolves our path unchanged.
    """
    return f"{ann_key}={ann_file}"


# ---------------------------------------------------------------------------
# Metrics parsing helpers
# ---------------------------------------------------------------------------

def parse_metrics_from_stdout(stdout_text: str) -> Optional[Dict[str, Any]]:
    """Extract the final metrics dict from MMEngine subprocess stdout.

    MMEngine prints metrics in one of two common formats:

    Format A – dict repr on an INFO line (preferred):
        04/07 ... mmengine - INFO - {'pedestrian/AP_BEV_0.50': 0.25, ...}

    Format B – Epoch summary line (fallback):
        Epoch(test) [10000/10000]  pedestrian/AP_BEV_0.50: 0.2500  ...

    We scan all matching lines and return the last non-empty match.
    """
    # Format A: look for an INFO line ending with a Python dict literal.
    dict_line_re = re.compile(r"INFO\s*-\s*(\{.+\})\s*$", re.MULTILINE)
    for raw in reversed(dict_line_re.findall(stdout_text)):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict) and parsed:
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            pass

    # Format B: "Epoch(test) [N/N]  key: value  key: value  ..."
    epoch_re = re.compile(r"Epoch\(test\)\s+\[\d+/\d+\]\s+(.+)$", re.MULTILINE)
    for summary in reversed(epoch_re.findall(stdout_text)):
        pairs = re.findall(r"(\S+):\s+([\d.eE+\-]+)", summary)
        if pairs:
            return {k: float(v) for k, v in pairs}

    return None


def find_best_json(work_dir: Path) -> Optional[Path]:
    """Find the most likely final-metrics JSON file in a test.py work dir.

    MMEngine's JsonLoggerHook writes scalars to:
        {work_dir}/<timestamp>/vis_data/scalars.json  (JSONL format)

    Prefer files whose name contains "metric", "eval", or "result".
    Fall back to the most recently modified JSON found.
    """
    all_jsons = sorted(
        work_dir.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_jsons:
        return None
    priority_kws = ("metric", "eval", "result")
    for p in all_jsons:
        if any(kw in p.name.lower() for kw in priority_kws):
            return p
    return all_jsons[0]


def load_metrics_from_json(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a JSON or JSONL file and return its metrics dict.

    For JSONL (scalars.json): each line is one record; return the last line
    that contains at least one numeric value (the final test-step metrics).
    For plain JSON: parse and return directly.
    """
    try:
        text = path.read_text()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1:
            # JSONL: scan from the end for a line with numeric values
            for line in reversed(lines):
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and any(
                        isinstance(v, (int, float)) for v in rec.values()
                    ):
                        return rec
                except Exception:
                    pass
        # Plain JSON
        return json.loads(text)
    except Exception:
        return None


def flatten_dict(d: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested dict into a dict with dotted-key scalars.

    Non-dict leaves (numbers, strings, booleans) are kept as-is.
    """
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, full_key))
    else:
        out[prefix] = d
    return out


# ---------------------------------------------------------------------------
# Core per-split evaluation runner
# ---------------------------------------------------------------------------

def run_one_split(
    *,
    config: Path,
    ckpt: Path,
    split_path: Path,
    ann_file: Path,
    split_name: str,
    work_dir: Path,
    python: str,
    test_script: Path,
    launcher: str,
    ann_key: str,
    extra_test_args: List[str],
    dry_run: bool,
) -> Dict[str, Any]:
    """Run tools/test.py for one split; return a structured result record.

    Saves stdout, stderr, and parsed metrics under work_dir.
    This function deliberately does NOT pass --show or --show-dir; this
    script is for quantitative evaluation only.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg_override = build_cfg_options_arg(ann_key, ann_file)

    cmd: List[str] = [
        python,
        str(test_script),
        str(config),
        str(ckpt),
        "--work-dir", str(work_dir),
        "--launcher", launcher,
        "--cfg-options", cfg_override,
    ] + extra_test_args

    print(f"\n{'=' * 70}")
    print(f"  Split : {split_name}")
    print(f"  Ann   : {ann_file}")
    print(f"  Cmd   : {shlex.join(cmd)}")
    print(f"{'=' * 70}")

    record: Dict[str, Any] = {
        "split_name": split_name,
        "group": infer_group(split_name),
        "label": infer_label(split_name),
        "split_file": str(split_path),
        "ann_file": str(ann_file),
        "work_dir": str(work_dir),
        "status": "unknown",
        "metrics": {},
        "metrics_file": None,
    }

    if dry_run:
        print("  [DRY RUN] skipping subprocess.")
        record["status"] = "dry_run"
        return record

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
            f"  [FAILED] exit code {proc.returncode}. "
            f"See {work_dir / 'stderr.txt'}"
        )
        record["status"] = "failed"
        return record

    # Parse metrics: prefer stdout, fall back to JSON files in work_dir.
    metrics: Optional[Dict[str, Any]] = parse_metrics_from_stdout(proc.stdout)
    if metrics is None:
        json_path = find_best_json(work_dir)
        if json_path:
            metrics = load_metrics_from_json(json_path)
            if metrics is not None:
                record["metrics_file"] = str(json_path)

    if metrics:
        metrics_path = work_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        record["metrics"] = metrics
        record["metrics_file"] = str(metrics_path)
        record["status"] = "success"
        print(f"  [OK] {len(metrics)} metric key(s) parsed → {metrics_path}")
    else:
        print(
            "  [WARN] subprocess exited 0 but metrics could not be parsed. "
            f"Check {work_dir / 'stdout.txt'}"
        )
        record["status"] = "success_no_metrics"

    return record


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def collect_metric_keys(records: List[Dict[str, Any]]) -> List[str]:
    """Collect all flat metric keys across all records, preserving first-seen order."""
    seen: Dict[str, None] = {}
    for r in records:
        for k in flatten_dict(r.get("metrics", {})):
            seen.setdefault(k, None)
    return list(seen.keys())


def build_aggregated(
    records: List[Dict[str, Any]],
    ckpt: Path,
    config: Path,
    run_name: str,
    eval_dir: Path,
) -> Dict[str, Any]:
    """Assemble the top-level aggregated results structure."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)
    return {
        "checkpoint": str(ckpt),
        "config": str(config),
        "run_name": run_name,
        "eval_dir": str(eval_dir),
        "groups": {g: groups[g] for g in GROUP_ORDER if g in groups},
        "flat_results": records,
    }


def write_aggregated_json(aggregated: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"  Saved: {path}")


def write_csv(
    records: List[Dict[str, Any]],
    metric_keys: List[str],
    path: Path,
) -> None:
    """Write one row per split with base columns and flattened metric columns."""
    base_cols = ["split_name", "group", "label", "status", "metrics_file"]
    fieldnames = base_cols + metric_keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            flat = flatten_dict(r.get("metrics", {}))
            row: Dict[str, Any] = {col: r.get(col, "") for col in base_cols}
            row.update({k: flat.get(k, "") for k in metric_keys})
            writer.writerow(row)
    print(f"  Saved: {path}")


def _md_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Render a list of dicts as a Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    aggregated: Dict[str, Any],
    metric_keys: List[str],
    path: Path,
) -> None:
    """Write a human-readable Markdown report grouped by split family."""
    lines: List[str] = [
        "# Evaluation Results\n",
        f"**Checkpoint:** `{aggregated['checkpoint']}`  ",
        f"**Config:** `{aggregated['config']}`  ",
        f"**Run:** `{aggregated['run_name']}`  ",
        "",
    ]

    for g in GROUP_ORDER:
        group_records = aggregated["groups"].get(g, [])
        if not group_records:
            continue

        lines.append(f"## {GROUP_TITLE.get(g, g)}\n")

        # Build flat rows for this group
        rows: List[Dict[str, Any]] = []
        for r in group_records:
            flat = flatten_dict(r.get("metrics", {}))
            row: Dict[str, Any] = {"label": r["label"], "status": r["status"]}
            row.update(flat)
            rows.append(row)

        # Only include metric columns that appear in at least one row of this group
        group_metric_keys = [
            k for k in metric_keys
            if any(k in flatten_dict(r.get("metrics", {})) for r in group_records)
        ]
        cols = ["label", "status"] + group_metric_keys
        lines.append(_md_table(rows, cols))
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run MMDetection3D test.py over multiple test splits and "
            "aggregate the metrics into JSON / CSV / Markdown."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Positional
    p.add_argument("config", help="Path to MMDet3D config file (.py).")
    p.add_argument("checkpoint", help="Path to model checkpoint (.pth).")

    # Split resolution
    p.add_argument(
        "--splits", nargs="+", default=None,
        help=(
            "Split filenames or full paths (e.g. lighting_test_day.txt). "
            "If omitted, all DEFAULT_SPLITS are used."
        ),
    )
    p.add_argument(
        "--splits-root", type=Path, default=None,
        help="Directory containing split .txt files. Used to resolve bare filenames.",
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
        help=(
            f"Prefix for deriving the pkl filename from a split stem. "
            f"Default: '{DEFAULT_ANN_PREFIX}'."
        ),
    )

    # Config override
    p.add_argument(
        "--ann-key", default=DEFAULT_ANN_KEY,
        help=(
            f"Dotted config key to override with the annotation pkl path. "
            f"Default: '{DEFAULT_ANN_KEY}'. "
            "Override with e.g. --ann-key test_evaluator.ann_file if needed."
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

    # Passthrough
    p.add_argument(
        "--score-thr", type=float, default=None,
        help="Score threshold passed to test.py (rarely needed for quantitative eval).",
    )
    p.add_argument(
        "--task", type=str, default=None,
        help="Task type passed to test.py (pass-through; visualization is NOT enabled).",
    )
    p.add_argument(
        "--extra-test-args", nargs="*", default=[],
        help="Additional arguments appended verbatim to the test.py invocation.",
    )

    # Behavior flags
    p.add_argument(
        "--fail-fast", action="store_true",
        help="Stop after the first failed split.",
    )
    p.add_argument(
        "--allow-missing-splits", action="store_true",
        help="Skip missing split files with a warning instead of raising an error.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    return p.parse_args()


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

    print(f"Run name  : {run_name}")
    print(f"Eval dir  : {eval_dir}")
    print(f"Infos dir : {infos_dir}")
    print(f"Ann key   : {args.ann_key}")

    # Determine which splits to run
    split_args = args.splits if args.splits else DEFAULT_SPLITS

    # Resolve each split argument to (split_name, split_file, ann_file)
    splits_to_run: List[Tuple[str, Path, Path]] = []
    for split_arg in split_args:
        split_file = resolve_split_file(split_arg, args.splits_root)
        if split_file is None:
            msg = (
                f"Split not found: '{split_arg}' "
                f"(splits-root: {args.splits_root})"
            )
            if args.allow_missing_splits:
                print(f"  [WARN] {msg} – skipping.")
                continue
            raise FileNotFoundError(msg)

        split_stem = split_file.stem  # e.g. "lighting_test_day"

        if infos_dir is None:
            raise ValueError(
                "Cannot derive ann_file: --infos-dir is not set and "
                "--splits-root was not provided. Supply --infos-dir explicitly."
            )
        ann_file = derive_ann_file(split_stem, infos_dir, args.ann_prefix)
        if not ann_file.exists():
            msg = f"Ann pkl not found: {ann_file}"
            if args.allow_missing_splits:
                print(f"  [WARN] {msg} – skipping split '{split_stem}'.")
                continue
            raise FileNotFoundError(msg)

        splits_to_run.append((split_stem, split_file, ann_file))

    if not splits_to_run:
        print("No splits to evaluate. Exiting.")
        return

    print(f"\nWill evaluate {len(splits_to_run)} split(s):")
    for name, _, af in splits_to_run:
        print(f"  {name}  →  {af}")

    # Build any extra test.py pass-through args
    extra: List[str] = list(args.extra_test_args or [])
    if args.score_thr is not None:
        extra += ["--score-thr", str(args.score_thr)]
    if args.task is not None:
        extra += ["--task", args.task]

    # Run each split
    records: List[Dict[str, Any]] = []
    for split_name, split_file, ann_file in splits_to_run:
        work_dir = eval_dir / "raw" / split_name
        record = run_one_split(
            config=config,
            ckpt=ckpt,
            split_path=split_file,
            ann_file=ann_file,
            split_name=split_name,
            work_dir=work_dir,
            python=args.python,
            test_script=args.test_script,
            launcher=args.launcher,
            ann_key=args.ann_key,
            extra_test_args=extra,
            dry_run=args.dry_run,
        )
        records.append(record)

        if args.fail_fast and record["status"] == "failed":
            print(f"\n[FAIL-FAST] Stopping after failed split: {split_name}")
            break

    # Sort records by stable group order then split name
    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    records.sort(key=lambda r: (group_rank.get(r["group"], 99), r["split_name"]))

    # Assemble and write outputs
    aggregated = build_aggregated(records, ckpt, config, run_name, eval_dir)
    metric_keys = collect_metric_keys(records)

    print(f"\n{'=' * 70}")
    print("Writing aggregated outputs …")
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_aggregated_json(aggregated, eval_dir / "aggregated_results.json")
    write_csv(records, metric_keys, eval_dir / "aggregated_results.csv")
    write_markdown(aggregated, metric_keys, eval_dir / "aggregated_results.md")

    n_ok = sum(1 for r in records if r["status"] in ("success", "dry_run"))
    n_fail = sum(1 for r in records if r["status"] == "failed")
    n_no_met = sum(1 for r in records if r["status"] == "success_no_metrics")
    print(
        f"\nDone.  {n_ok}/{len(records)} succeeded, "
        f"{n_fail} failed, {n_no_met} parsed no metrics."
    )
    print(f"Results: {eval_dir}")


if __name__ == "__main__":
    main()
