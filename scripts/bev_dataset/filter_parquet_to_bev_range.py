#!/usr/bin/env python3
"""
Filter the ZOD-MoE dataset parquet to keep only pedestrians whose 3D box
center falls inside a specified BEV (x, y) range.

This script reads the existing dataset parquet, loads each frame's label JSON,
applies the BEV range filter on the ego-frame 3D box centers, and rewrites all
pedestrian-related parquet columns to reflect only the retained instances.

The label JSON files on disk are NOT modified — only the parquet is updated.

Example
-------
    python scripts/bev_dataset/filter_parquet_to_bev_range.py \
        --input  /home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset.parquet \
        --output /home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset_bev108.parquet \
        --x-min 0.0  --x-max 108.0 \
        --y-min -54.0 --y-max 54.0

If --input points to the weather-group version, use that path instead:
    --input /mnt/tier2/project/p201222/u103958/zod_moe/index/zod_moe_dataset_with_weather_group.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── pedestrian-related columns that get recomputed ──────────────────────────
PED_COUNT_COLS = [
    "ped_count_clear",
    "ped_count_unclear",
    "ped_occ_none",
    "ped_occ_light",
    "ped_occ_medium",
    "ped_occ_heavy",
    "ped_occ_veryheavy",
    "ped_occ_missing",
    "ped_occ_unknown",
]
PED_LIST_COLS = [
    "ped_uuid",
    "ped_unclear_list",
    "ped_occlusion_list",
]
PED_SCALAR_COLS = [
    "ped_present",
    "num_pedestrians_final",
]


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_label_json(path_str: str | None) -> dict | None:
    """Safely load a label JSON file.  Returns None on any failure."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _xyxy_from_projected_corners(corners: list[list[float]]) -> list[float] | None:
    """
    Compute a tight [x1, y1, x2, y2] image bbox from 8 projected corner
    points.  Returns None if all corners are non-finite.
    """
    arr = np.asarray(corners, dtype=np.float64)        # (8, 2)
    valid = np.all(np.isfinite(arr), axis=1)
    if not valid.any():
        return None
    u_min, v_min = arr[valid].min(axis=0)
    u_max, v_max = arr[valid].max(axis=0)
    return [float(u_min), float(v_min), float(u_max), float(v_max)]


def _parse_list_col(val: Any) -> list:
    """Parquet may store list columns as ndarray or native list."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import ast
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return list(val)


# ── per-row filtering ──────────────────────────────────────────────────────

def filter_row(
    row: dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    strict: bool,
) -> tuple[dict[str, Any], int, int]:
    """
    Apply BEV range filter to one parquet row.

    Returns (updated_row, n_before, n_after).
    """
    label = _load_label_json(row.get("label_file_path"))

    # If no label JSON, zero out pedestrian fields
    if label is None:
        msg = f"frame {row.get('frame_id')}: label JSON missing or unreadable"
        if strict:
            raise FileNotFoundError(msg)
        log.warning(msg)
        return _zero_ped_fields(row), 0, 0

    instances = label.get("instances", [])
    n_before = len(instances)

    # --- aligned lists from the parquet (same length as label instances) ---
    ped_uuids = _parse_list_col(row.get("ped_uuid"))
    ped_unclear = _parse_list_col(row.get("ped_unclear_list"))
    ped_occ = _parse_list_col(row.get("ped_occlusion_list"))

    # Sanity: these should all equal n_before
    if len(ped_uuids) != n_before:
        msg = (
            f"frame {row.get('frame_id')}: ped_uuid length ({len(ped_uuids)}) "
            f"!= label instances ({n_before})"
        )
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    # --- determine which instances pass the BEV range filter ---
    keep_mask: list[bool] = []
    for inst in instances:
        center = inst.get("box_center_ego_geometric")
        if center is None or len(center) < 2:
            keep_mask.append(False)
            continue
        x, y = float(center[0]), float(center[1])
        keep_mask.append(x_min <= x <= x_max and y_min <= y <= y_max)

    # --- apply mask to aligned lists ---
    kept_uuids: list[str] = []
    kept_unclear: list[bool] = []
    kept_occ: list[str] = []
    kept_xyxy: list[list[float]] = []

    for i, keep in enumerate(keep_mask):
        if not keep:
            continue
        # UUID
        if i < len(ped_uuids):
            kept_uuids.append(str(ped_uuids[i]))
        # unclear
        if i < len(ped_unclear):
            kept_unclear.append(bool(ped_unclear[i]))
        else:
            kept_unclear.append(False)
        # occlusion
        if i < len(ped_occ):
            kept_occ.append(str(ped_occ[i]))
        else:
            kept_occ.append("missing")
        # recompute tight 2D bbox from projected corners
        corners = instances[i].get("projected_corners_uv")
        if corners is not None:
            bbox = _xyxy_from_projected_corners(corners)
            if bbox is not None:
                kept_xyxy.append(bbox)

    n_after = len(kept_uuids)

    # --- recompute count columns from kept lists ---
    occ_counts = {k: 0 for k in [
        "none", "light", "medium", "heavy", "veryheavy", "missing", "unknown"
    ]}
    for occ in kept_occ:
        bucket = occ if occ in occ_counts else "unknown"
        occ_counts[bucket] += 1

    n_unclear = sum(1 for u in kept_unclear if u)
    n_clear = n_after - n_unclear

    new_row = dict(row)
    new_row["ped_uuid"] = kept_uuids
    new_row["ped_unclear_list"] = kept_unclear
    new_row["ped_occlusion_list"] = kept_occ
    new_row["xyxy_bboxes"] = kept_xyxy
    new_row["ped_count_clear"] = n_clear
    new_row["ped_count_unclear"] = n_unclear
    new_row["ped_occ_none"] = occ_counts["none"]
    new_row["ped_occ_light"] = occ_counts["light"]
    new_row["ped_occ_medium"] = occ_counts["medium"]
    new_row["ped_occ_heavy"] = occ_counts["heavy"]
    new_row["ped_occ_veryheavy"] = occ_counts["veryheavy"]
    new_row["ped_occ_missing"] = occ_counts["missing"]
    new_row["ped_occ_unknown"] = occ_counts["unknown"]
    new_row["ped_present"] = int(n_after > 0)
    new_row["num_pedestrians_final"] = n_after

    return new_row, n_before, n_after


def _zero_ped_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return row with all pedestrian fields zeroed / emptied."""
    new_row = dict(row)
    for col in PED_COUNT_COLS:
        new_row[col] = 0
    for col in PED_LIST_COLS:
        new_row[col] = []
    new_row["ped_present"] = 0
    new_row["num_pedestrians_final"] = 0
    new_row["xyxy_bboxes"] = []
    return new_row


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter ZOD-MoE parquet pedestrians to a BEV (x, y) range."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input parquet path.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output (filtered) parquet path.")
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=108.0)
    p.add_argument("--y-min", type=float, default=-54.0)
    p.add_argument("--y-max", type=float, default=54.0)
    p.add_argument("--strict", action="store_true",
                   help="Raise on malformed label schema instead of warning.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Input:  %s", args.input)
    log.info("Output: %s", args.output)
    log.info("BEV range: x=[%.1f, %.1f]  y=[%.1f, %.1f]",
             args.x_min, args.x_max, args.y_min, args.y_max)

    df = pd.read_parquet(args.input)
    log.info("Loaded %d rows", len(df))

    records = df.to_dict(orient="records")
    updated_records: list[dict[str, Any]] = []

    total_before = 0
    total_after = 0
    frames_changed = 0
    frames_with_warnings = 0

    for rec in records:
        try:
            new_rec, n_before, n_after = filter_row(
                rec,
                args.x_min, args.x_max,
                args.y_min, args.y_max,
                strict=args.strict,
            )
        except Exception as exc:
            log.error("frame %s: %s", rec.get("frame_id"), exc)
            raise

        updated_records.append(new_rec)
        total_before += n_before
        total_after += n_after
        if n_before != n_after:
            frames_changed += 1

    df_out = pd.DataFrame(updated_records)

    # Preserve original column order
    orig_cols = list(df.columns)
    out_cols = [c for c in orig_cols if c in df_out.columns]
    for c in df_out.columns:
        if c not in out_cols:
            out_cols.append(c)
    df_out = df_out[out_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)

    # ── summary ──
    log.info("=" * 60)
    log.info("Frames processed:          %d", len(records))
    log.info("Total pedestrians before:  %d", total_before)
    log.info("Total pedestrians after:   %d", total_after)
    log.info("Pedestrians removed:       %d (%.2f%%)",
             total_before - total_after,
             100.0 * (total_before - total_after) / max(total_before, 1))
    log.info("Frames whose counts changed: %d", frames_changed)
    log.info("Output written to: %s", args.output)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
