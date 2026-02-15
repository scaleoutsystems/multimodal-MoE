"""
Helpers for loading frame-level data from parquet + split CSVs.
"glue layer" that takes split CSV IDs + canonical parquet table and produces
the concrete split DataFrame rows that downstream model/export pipelines need. 

Why this module exists
----------------------
Our parquet index is the source-of-truth table for frame_id -> image path + labels/metadata.
Model training code should not re-implement split filtering and frame_id normalization in
multiple places, because that leads to subtle split mismatch bugs.

What this module does
---------------------
- Loads split frame IDs from train/val/test CSV files (train_ids.csv, val_ids.csv, test_ids.csv).
- These splits are stratified by ped_bin_4 and time_of_day.
- Normalizes frame IDs to one canonical format (6-digit zero-padded strings).
    - Example: "123" -> "000123"
- Filters parquet rows to exactly one split (train, test, or val).
- Returns rows in deterministic split order for reproducibility/debugging (the returned
DataFrame is sorted by split CSV order). Same input split --> same output DataFrame every run. 

Where it is used
----------------
- Used by `scripts/export_yolo_dataset.py` to build YOLO train/val/test exports.
- Intended to be reused by future DINO/fusion exporters and evaluators so every backend
  reads the exact same split membership.

Why framework-agnostic matters
------------------------------
The split/index logic is data plumbing, not model logic. Keeping it independent of YOLO
lets us swap model backends (YOLO, DINO, multimodal fusion) without changing how data
selection works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_frame_id_series(values: Iterable) -> pd.Series:
    """
    Normalize frame IDs to a consistent 6-digit string format.

    Input:
        values: Any iterable of frame IDs (ints/strings/mixed).

    Output:
        pd.Series of zero-padded string IDs like "000123".

    --> get one canonical ID format so split CSVs and parquet rows match
        reliably across every training/export pipeline.
    """
    return (
        pd.Series(values)
        .astype(str) # convert to string
        .str.strip() # remove whitespace
        .str.replace(r"\.0$", "", regex=True) # remove trailing .0
        .str.zfill(6) # zero-pad to 6 digits
    )


def load_split_frame_ids(split_csv: str | Path, frame_id_col: str = "frame_id") -> list[str]:
    """
    Load frame IDs from a split CSV and normalize formatting.

    Input:
        split_csv: Path to train/val/test CSV.
        frame_id_col: Column name that stores frame IDs.

    Output:
        List of normalized 6-digit frame ID strings.

    Why:
        Every downstream stage (export/train/eval) depends on split IDs being
        clean and consistently formatted.
    """
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"split_csv not found: {split_csv}")

    split_df = pd.read_csv(split_csv) # read the split CSV into a DataFrame
    if frame_id_col not in split_df.columns: # check if the frame_id_col is in the DataFrame
        raise ValueError(
            f"split_csv missing '{frame_id_col}'. Columns: {split_df.columns.tolist()}"
        ) # raise an error if the frame_id_col is not in the DataFrame

    frame_ids = normalize_frame_id_series(split_df[frame_id_col]).tolist()
    return frame_ids


def load_split_frames(
    frames_parquet: str | Path, 
    split_csv: str | Path, 
    frame_id_col: str = "frame_id", 
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load parquet rows for one split and return a deterministic DataFrame.

    Input:
        frames_parquet: Path to the frame-level parquet index.
        split_csv: Path to split CSV (train/val/test).
        frame_id_col: Shared frame ID column name.
        required_columns: Optional list of columns to read/validate.

    If required_columns is provided, we load only those columns (plus frame_id_col if missing from that list).
    Otherwise we load all parquet columns.

    Output:
        pd.DataFrame filtered to the requested split, ordered by split CSV order.

    Why:
        This gives all model adapters the same split filtering behavior and
        avoids hard-to-debug split mismatch bugs.
    """
    frames_parquet = Path(frames_parquet)
    if not frames_parquet.exists():
        raise FileNotFoundError(f"frames_parquet not found: {frames_parquet}")

    split_ids = load_split_frame_ids(split_csv=split_csv, frame_id_col=frame_id_col)

    # Always include frame_id_col in parquet read even if user forgot.
    req_cols = list(required_columns or [])
    if frame_id_col not in req_cols:
        req_cols = [frame_id_col] + req_cols

    frames_df = pd.read_parquet(frames_parquet, columns=req_cols if req_cols else None)
    if frame_id_col not in frames_df.columns:
        raise ValueError(
            f"parquet missing '{frame_id_col}'. Columns: {frames_df.columns.tolist()}"
        )

    frames_df[frame_id_col] = normalize_frame_id_series(frames_df[frame_id_col])
    split_set = set(split_ids)
    # filter the DataFrame to only include rows where the frame ID is in the split set
    frames_df = frames_df[frames_df[frame_id_col].isin(split_set)].copy()

    # Deterministic sort by split order so debugging is easier.
    order = {fid: i for i, fid in enumerate(split_ids)}
    frames_df["_split_order"] = frames_df[frame_id_col].map(order)
    frames_df = frames_df.sort_values("_split_order").drop(columns=["_split_order"])
    frames_df = frames_df.reset_index(drop=True)

    if len(frames_df) == 0:
        raise RuntimeError(
            "No rows matched split IDs. Check frame_id formatting and split/parquet alignment."
        )

    return frames_df
