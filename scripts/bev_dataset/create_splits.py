#!/usr/bin/env python3
"""
Create reproducible train / val / test splits for the ZOD-MoE dataset.

Reads the enriched parquet (with complexity_bin) and writes three text files
(one frame_id per line) plus a CSV recording every frame's split assignment.

Stratification
--------------
Splits are stratified on ``scraped_weather + complexity_bin``.  This ensures
each split has a representative mix of:

  * **Weather conditions** — detector performance varies with rain, snow, fog.
  * **Scene complexity** — empty, low, medium, high pedestrian scenes must be
    balanced to avoid biased evaluation.

Rare-group handling
-------------------
If any stratification-key combination has fewer than ``--min-group-size``
samples, those rows are merged into a temporary "RARE" bucket for
stratification only.  Original column values are never changed.

Example
-------
python scripts/bev_dataset/create_splits.py \\
  --input-parquet /mnt/tier2/.../index/zod_moe_dataset_bev108_with_complexity_bin.parquet \\
  --output-dir    /mnt/tier2/.../index \\
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
    "/mnt/tier2/project/p201222/u103958/zod_moe/index/"
    "zod_moe_dataset_bev108_with_complexity_bin.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/tier2/project/p201222/u103958/zod_moe/index"
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
    """Build a stratification key from scraped_weather + complexity_bin.

    Groups with fewer than ``min_group_size`` members are replaced with
    "RARE" so ``train_test_split`` can still perform stratified sampling.
    """
    raw_key = (
        df["scraped_weather"].fillna("NULL").astype(str)
        + "__"
        + df["complexity_bin"].fillna("NULL").astype(str)
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
    """Print and return a split x field cross-tabulation."""
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

    for col in ("frame_id", "scraped_weather", "complexity_bin"):
        assert col in df.columns, f"Missing required column: {col}"

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

    def assign(fid: str) -> str:
        if fid in train_set:
            return "train"
        if fid in val_set:
            return "val"
        return "test"

    df["split"] = df["frame_id"].astype(str).apply(assign)

    csv_path = args.output_dir / "split_assignments.csv"
    df[["frame_id", "split", "scraped_weather", "complexity_bin",
        "solar_context_bin", "road_type"]].to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # ---- Cross-distribution summaries ----
    ct_weather = print_cross_distribution(df, "split", "scraped_weather")
    ct_complexity = print_cross_distribution(df, "split", "complexity_bin")
    ct_solar = print_cross_distribution(df, "split", "solar_context_bin")
    ct_road = print_cross_distribution(df, "split", "road_type")

    ct_weather.to_csv(args.output_dir / "split_by_scraped_weather.csv")
    ct_complexity.to_csv(args.output_dir / "split_by_complexity_bin.csv")
    ct_solar.to_csv(args.output_dir / "split_by_solar_context_bin.csv")
    ct_road.to_csv(args.output_dir / "split_by_road_type.csv")

    print(f"\nSaved summary CSVs to {args.output_dir}")

    # ---- Final report ----
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
    print(f"Strat key:      scraped_weather + complexity_bin")


if __name__ == "__main__":
    main()
