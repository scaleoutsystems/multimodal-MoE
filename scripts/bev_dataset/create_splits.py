#!/usr/bin/env python3
"""
Create reproducible train / val / test splits for the ZOD-MoE dataset.

Reads the final dataset parquet and writes three text files (one frame_id
per line) plus a CSV that records every frame's split assignment.

Stratification
--------------
Splits are stratified on ``solar_context_bin + road_type``.  This ensures
that each split has a representative mix of:

  * **Illumination conditions** (night, twilight, low/mid/high sun) —
    because detector performance varies strongly with lighting, and an
    imbalanced split would bias evaluation metrics.
  * **Scene layout** (city, highway, arterial-urban/rural, smaller-rural) —
    because road type correlates with pedestrian density, occlusion
    patterns, and LiDAR point density.

Weather (``weather_group``) is intentionally *not* part of the split key.
It stays in the parquet so it can be used later as a context signal for
the MoE router, but we don't want to couple the primary train/val/test
partitioning to it.

Rare-group handling
-------------------
If any ``solar_context_bin × road_type`` combination has fewer than
``--min-group-size`` samples, those rows are merged into a temporary
"RARE" bucket for stratification purposes only.  The original column
values are never changed.

Example
-------
python scripts/dataset/create_splits.py \\
  --input-parquet /mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/index/zod_moe_dataset_with_weather_group.parquet \\
  --output-dir    /mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/splits \\
  --train-frac 0.80 --val-frac 0.10 --test-frac 0.10 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_INPUT = Path(
    "/mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/index/"
    "zod_moe_dataset_with_weather_group.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/splits"
)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for ZOD-MoE.",
    )
    p.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        help="Strat-key groups smaller than this are merged into a RARE bucket.",
    )
    return p.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def build_strat_key(
    df: pd.DataFrame,
    min_group_size: int,
) -> pd.Series:
    """Build a stratification key from solar_context_bin + road_type.

    Groups with fewer than ``min_group_size`` members are replaced with
    the string ``"RARE"`` so that ``train_test_split`` can still perform
    stratified sampling without raising an error.
    """
    raw_key = (
        df["solar_context_bin"].fillna("NULL").astype(str)
        + "__"
        + df["road_type"].fillna("NULL").astype(str)
    )

    counts = raw_key.value_counts()
    rare_keys = set(counts[counts < min_group_size].index)

    if rare_keys:
        print(f"  Merging {len(rare_keys)} rare strat-key group(s) into 'RARE':")
        for rk in sorted(rare_keys):
            print(f"    {rk}  ({counts[rk]} samples)")
        return raw_key.where(~raw_key.isin(rare_keys), other="RARE")

    return raw_key


def write_split_file(path: Path, frame_ids: list[str]) -> None:
    """Write one frame_id per line, sorted for readability."""
    with open(path, "w") as f:
        for fid in sorted(frame_ids):
            f.write(f"{fid}\n")


def print_cross_distribution(
    df: pd.DataFrame,
    split_col: str,
    field: str,
) -> pd.DataFrame:
    """Print and return a split × field cross-tabulation."""
    ct = pd.crosstab(df[split_col], df[field], margins=True)
    print(f"\n--- {field} distribution per split ---")
    print(ct.to_string())
    return ct


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ---- Validate fractions ----
    total_frac = args.train_frac + args.val_frac + args.test_frac
    assert abs(total_frac - 1.0) < 1e-6, (
        f"Fractions must sum to 1.0, got {total_frac:.4f}"
    )

    # ---- Read parquet ----
    assert args.input_parquet.exists(), f"Not found: {args.input_parquet}"
    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} rows from {args.input_parquet}")

    for col in ("frame_id", "solar_context_bin", "road_type"):
        assert col in df.columns, f"Missing column: {col}"

    frame_ids = df["frame_id"].astype(str).values

    # ---- Build stratification key ----
    strat_key = build_strat_key(df, args.min_group_size)
    print(f"Unique strat-key groups (after rare merge): {strat_key.nunique()}")

    # ---- First split: train vs temp (val + test) ----
    temp_frac = args.val_frac + args.test_frac

    train_ids, temp_ids, train_strat, temp_strat = train_test_split(
        frame_ids,
        strat_key.values,
        test_size=temp_frac,
        random_state=args.seed,
        stratify=strat_key.values,
    )

    # ---- Second split: val vs test ----
    # temp_frac is (val + test); within temp, test's share is:
    #   test_frac / (val_frac + test_frac)
    test_share_of_temp = args.test_frac / temp_frac

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_share_of_temp,
        random_state=args.seed,
        stratify=temp_strat,
    )

    print(f"\nSplit sizes:  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")
    print(
        f"Split fracs:  train={len(train_ids)/len(df):.4f}  "
        f"val={len(val_ids)/len(df):.4f}  "
        f"test={len(test_ids)/len(df):.4f}"
    )

    # ---- Write split text files ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train.txt"
    val_path = args.output_dir / "val.txt"
    test_path = args.output_dir / "test.txt"

    write_split_file(train_path, train_ids.tolist())
    write_split_file(val_path, val_ids.tolist())
    write_split_file(test_path, test_ids.tolist())

    print(f"\nWrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {test_path}")

    # ---- Write split_assignments.csv ----
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    def assign(fid: str) -> str:
        if fid in train_set:
            return "train"
        if fid in val_set:
            return "val"
        return "test"

    df["split"] = df["frame_id"].astype(str).apply(assign)

    csv_path = args.output_dir / "split_assignments.csv"
    df[["frame_id", "split", "solar_context_bin", "road_type"]].to_csv(
        csv_path, index=False
    )
    print(f"Wrote: {csv_path}")

    # ---- Cross-distribution summaries ----
    ct_solar = print_cross_distribution(df, "split", "solar_context_bin")
    ct_road = print_cross_distribution(df, "split", "road_type")

    # Save summary CSVs
    ct_solar.to_csv(args.output_dir / "split_by_solar_context_bin.csv")
    ct_road.to_csv(args.output_dir / "split_by_road_type.csv")

    print(f"\nSaved summary CSVs to {args.output_dir}")

    # ---- Final report ----
    n_unknown_weather = 0
    if "weather_group" in df.columns:
        n_unknown_weather = int((df["weather_group"] == "unknown").sum())

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Input parquet:  {args.input_parquet}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Total rows:     {len(df)}")
    print(f"Train:          {len(train_ids)}")
    print(f"Val:            {len(val_ids)}")
    print(f"Test:           {len(test_ids)}")
    print(f"Seed:           {args.seed}")
    print(f"Strat key:      solar_context_bin + road_type")
    if "weather_group" in df.columns:
        print(f"Weather unknown:{n_unknown_weather} (not used in split key)")


if __name__ == "__main__":
    main()
