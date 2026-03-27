"""
Add binned solar-elevation context labels to the source frame parquet.

Why this script exists:
- We use categorical solar context for routing (not raw continuous angle),
  because MoE routing is more stable/interpretable with coarse illumination regimes.
- We keep the source parquet immutable and write a derived parquet with the new bins.

How we chose the cutoff values:
- The U.S. Naval Observatory defines standard illumination boundaries based on
  solar elevation angle, where civil twilight occurs when the Sun is between
  -6° and 0° below the horizon, and sunrise/sunset occurs at 0°:
  https://aa.usno.navy.mil/faq/RST_defs
- Using these definitions, we group solar elevation into three illumination regimes:
  - (< -6°) night-like conditions
  - (-6° to 0°) civil twilight transition
  - (>= 0°) daytime
- This categorization is useful in computer vision because illumination strongly
  affects visibility, contrast, and detection difficulty, making solar elevation
  a meaningful context signal for analysis and context-aware routing (e.g., MoE).
- We further split daytime into 0..15, 15..45, and >45 deg bins.
  These are practical ML routing bands (not strict astronomy classes) used to
  separate low-sun / mid-sun / high-sun lighting geometry, which often changes
  shadows, glare, and overall scene appearance for detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Allow running as either:
# - python -m scripts.add_solar_context_bins
# - python scripts/add_solar_context_bins.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    INDEX_DIR,
    ZODMOE_FRAMES_WITH_BOXES_PARQUET,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add solar context bins and save derived parquet.")
    parser.add_argument(
        "--in-parquet",
        type=str,
        default=str(ZODMOE_FRAMES_WITH_BOXES_PARQUET),
        help="Input source-of-truth parquet path.",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=str(INDEX_DIR / "ZODmoe_frames_with_xyxy_bboxes_and_solar_bins.parquet"),
        help="Output derived parquet path with solar_context_bin column.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output parquet if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_parquet = Path(args.in_parquet).expanduser().resolve()
    out_parquet = Path(args.out_parquet).expanduser().resolve()

    if not in_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_parquet}")
    if out_parquet.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output parquet already exists: {out_parquet}. "
            "Use --overwrite to replace it."
        )

    df = pd.read_parquet(in_parquet)
    if "solar_angle_elevation" not in df.columns:
        raise ValueError("Expected 'solar_angle_elevation' column in input parquet.")

    solar = pd.to_numeric(df["solar_angle_elevation"], errors="coerce")

    # I use fixed bins aligned with common illumination regimes.
    solar_bins = [-1e9, -6.0, 0.0, 15.0, 45.0, 1e9]
    solar_labels = [
        "night(<-6)",
        "twilight(-6..0)",
        "low_sun(0..15)",
        "mid_sun(15..45)",
        "high_sun(>45)",
    ]

    solar_binned = pd.cut(
        solar,
        bins=solar_bins,
        labels=solar_labels,
        include_lowest=True,
    )

    # Store as plain strings for portability in downstream code/parquet reads.
    df["solar_context_bin"] = solar_binned.astype("string").fillna("missing")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    print(f"Input parquet  -> {in_parquet}")
    print(f"Output parquet -> {out_parquet}")
    print("solar_context_bin counts:")
    print(df["solar_context_bin"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
