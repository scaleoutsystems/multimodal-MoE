"""
Add coarse solar bins and scene-complexity bins to the source frame parquet.

Reads zod_moe_dataset_bev108.parquet and writes a derived parquet with two
new/updated columns:

  solar_context_bin   — coarse 3-class illumination (night / twilight / day)
  complexity_bin      — scene difficulty (empty / low / medium / high)

The old 5-class solar_context_bin is overwritten with the coarse version.

Solar bin cutoffs (US Naval Observatory civil-twilight definitions):
  < -6°    night
  -6° – 0° twilight
  >= 0°    day

Complexity score:
  ped_count_total + 0.25*light + 0.5*medium + 1.0*heavy + 1.5*veryheavy
  Bins: 0 → empty, (0,4] → low, (4,10] → medium, >10 → high
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INDEX_DIR = Path("/mnt/tier2/project/p201392/u103958/zod_moe/index")
DEFAULT_IN = INDEX_DIR / "zod_moe_dataset_bev108.parquet"
DEFAULT_OUT = INDEX_DIR / "zod_moe_dataset_bev108_with_complexity_bin.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add coarse solar bins + complexity bins to parquet.")
    p.add_argument("--in-parquet", type=str, default=str(DEFAULT_IN),
                   help="Input source-of-truth parquet.")
    p.add_argument("--out-parquet", type=str, default=str(DEFAULT_OUT),
                   help="Output derived parquet.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output if it already exists.")
    return p.parse_args()


# ── Complexity scoring ───────────────────────────────────────────────

def compute_complexity_score(row: pd.Series) -> float:
    """Weighted sum of pedestrian count + occlusion difficulty."""
    return (
        row["num_pedestrians_final"]
        + 0.25 * row["ped_occ_light"]
        + 0.50 * row["ped_occ_medium"]
        + 1.00 * row["ped_occ_heavy"]
        + 1.50 * row["ped_occ_veryheavy"]
    )


def assign_complexity_bin(score: float) -> str:
    if score == 0:
        return "empty"
    elif score <= 4:
        return "low"
    elif score <= 10:
        return "medium"
    else:
        return "high"


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    in_parquet = Path(args.in_parquet).expanduser().resolve()
    out_parquet = Path(args.out_parquet).expanduser().resolve()

    if not in_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_parquet}")
    if out_parquet.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_parquet}. Use --overwrite to replace.")

    df = pd.read_parquet(in_parquet)
    print(f"Loaded {len(df)} rows from {in_parquet}")

    # ── 1. Coarse solar bins (replaces old 5-class solar_context_bin) ──
    if "solar_angle_elevation" not in df.columns:
        raise ValueError("Expected 'solar_angle_elevation' column.")

    solar = pd.to_numeric(df["solar_angle_elevation"], errors="coerce")
    solar_bins = [-1e9, -3.0, 1e9]
    solar_labels = ["night", "day"]

    df["solar_context_bin"] = (
        pd.cut(solar, bins=solar_bins, labels=solar_labels, include_lowest=True)
        .astype("string")
        .fillna("missing")
    )

    # ── 2. Complexity bin ──
    required = ["num_pedestrians_final", "ped_occ_light", "ped_occ_medium",
                "ped_occ_heavy", "ped_occ_veryheavy"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column for complexity: {col}")

    df["complexity_score"] = df.apply(compute_complexity_score, axis=1)
    df["complexity_bin"] = df["complexity_score"].apply(assign_complexity_bin)

    # ── 3. Save ──
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    print(f"\nInput:  {in_parquet}")
    print(f"Output: {out_parquet}")
    print(f"\nsolar_context_bin counts:")
    print(df["solar_context_bin"].value_counts(dropna=False).to_string())
    print(f"\ncomplexity_bin counts:")
    print(df["complexity_bin"].value_counts(dropna=False).to_string())
    print(f"\ncomplexity_score stats:")
    print(df["complexity_score"].describe().to_string())


if __name__ == "__main__":
    main()
