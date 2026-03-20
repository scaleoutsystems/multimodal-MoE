#!/usr/bin/env python3
"""
Add a ``weather_group`` column to an existing ZOD-MoE dataset parquet.

The column is derived from ``scraped_weather`` using a simple many-to-one
mapping that groups the raw weather strings into five coarse categories
(clear_like, cloud_like, precipitation, fog, wind) plus an
``unknown`` fallback for missing / unrecognised values.

The input parquet is never modified; a new file is written instead.

Example
-------
python scripts/dataset/add_weather_group.py \
  --input-parquet /home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset.parquet \
  --output-parquet /home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset_with_weather_group.parquet

WHY?:
clear-day and clear-night are both “clear,” but their lighting difference is already captured by solar_context_bin
partly-cloudy-day and cloudy are similar visually
The raw weather labels mix together:
- true weather effects
- lighting effects
- unnecessarily fine distinctions
--> context-aware routing is harder.
grouping --> turn raw weather labels into a smaller number of meaningful weather regimes.
--> more samples per category, reduced redundancy.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Mapping: scraped_weather value  ->  weather_group
# ------------------------------------------------------------------
WEATHER_TO_GROUP: dict[str, str] = {
    "clear-day":          "clear_like",
    "clear-night":        "clear_like",
    "cloudy":             "cloud_like",
    "partly-cloudy-day":  "cloud_like",
    "partly-cloudy-night":"cloud_like",
    "rain":               "precipitation",
    "snow":               "precipitation",
    "fog":                "fog",
    "wind":               "wind",
}


def map_weather_group(value: object) -> str:
    """Return the coarse weather group for a single ``scraped_weather`` value.

    Strips whitespace, lowercases, and looks up the result in
    ``WEATHER_TO_GROUP``.  Returns ``"unknown"`` when the value is null,
    empty, or not present in the mapping.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"

    key = str(value).strip().lower()
    if not key:
        return "unknown"

    return WEATHER_TO_GROUP.get(key, "unknown")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
DEFAULT_INPUT = Path(
    "/home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset.parquet"
)
DEFAULT_OUTPUT = Path(
    "/home/edgelab/multimodal-MoE/outputs/index/zod_moe_dataset_with_weather_group.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a weather_group column to a ZOD-MoE dataset parquet.",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source parquet (read-only).",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the new parquet with weather_group added.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    assert args.input_parquet.exists(), f"Input not found: {args.input_parquet}"
    assert args.input_parquet != args.output_parquet, (
        "Input and output paths must differ so the original is not overwritten."
    )

    df = pd.read_parquet(args.input_parquet)
    print(f"Read {len(df)} rows from {args.input_parquet}")

    assert "scraped_weather" in df.columns, (
        "Input parquet is missing the 'scraped_weather' column."
    )

    # --- Apply mapping ---
    df["weather_group"] = df["scraped_weather"].apply(map_weather_group)

    # --- Summary: original scraped_weather distribution ---
    print("\n--- scraped_weather value counts ---")
    sw_counts = df["scraped_weather"].value_counts(dropna=False).sort_values(ascending=False)
    for val, cnt in sw_counts.items():
        print(f"  {str(val):30s} {cnt}")

    # --- Summary: new weather_group distribution ---
    print("\n--- weather_group value counts ---")
    wg_counts = df["weather_group"].value_counts().sort_values(ascending=False)
    for val, cnt in wg_counts.items():
        print(f"  {val:30s} {cnt}")

    n_unknown = int((df["weather_group"] == "unknown").sum())

    # --- Write output ---
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)

    # --- Final report ---
    print(f"\nInput:   {args.input_parquet}")
    print(f"Output:  {args.output_parquet}")
    print(f"Rows:    {len(df)}")
    print(f"Unknown: {n_unknown}")


if __name__ == "__main__":
    main()
