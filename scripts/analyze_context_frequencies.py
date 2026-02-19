"""
Generate context-frequency analysis artifacts for routing features.

This script reads the source-of-truth frame parquet and writes:
1) A table-style plot of category frequencies (Plot A)
2) A multi-panel frequency bar plot (Plot B)
3) A CSV with the exact counts/percentages used by both plots

Output location:
    outputs/analysis/camera/detection
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running as either:
# - python -m scripts.analyze_context_frequencies
# - python scripts/analyze_context_frequencies.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import ZODMOE_FRAMES_WITH_BOXES_PARQUET, OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build context frequency table/plots for routing features.")
    parser.add_argument(
        "--frames-parquet",
        type=str,
        default=str(ZODMOE_FRAMES_WITH_BOXES_PARQUET),
        help="Source-of-truth frame parquet with context fields.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUTPUTS_DIR / "analysis" / "camera" / "detection"),
        help="Directory where plots/CSV should be written.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="final",
        help="Filename suffix tag (e.g., final, v1, ablation_a).",
    )
    return parser.parse_args()


def _build_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-form frequency table for selected routing context fields.
    """
    # Final context set (no leakage fields like ped_bin_4 / ped_* counts).
    context_cat_fields = [
        "scraped_weather",
        "time_of_day",
        "road_type",
        "road_condition",
    ]

    solar = pd.to_numeric(df["solar_angle_elevation"], errors="coerce")
    solar_bins = [-1e9, -6.0, 0.0, 15.0, 45.0, 1e9]
    solar_labels = [
        "night(<-6)",
        "twilight(-6..0)",
        "low_sun(0..15)",
        "mid_sun(15..45)",
        "high_sun(>45)",
    ]
    df = df.copy()
    df["solar_context_bin"] = pd.cut(
        solar,
        bins=solar_bins,
        labels=solar_labels,
        include_lowest=True,
    )

    fields = context_cat_fields + ["solar_context_bin"]
    n_total = len(df)
    rows = []
    for field in fields:
        s = df[field].astype("string").fillna("missing")
        vc = s.value_counts(dropna=False)
        for cat, count in vc.items():
            rows.append(
                {
                    "field": field,
                    "category": str(cat),
                    "count": int(count),
                    "percent": float(count) * 100.0 / n_total,
                }
            )

    freq_df = pd.DataFrame(rows)
    freq_df["field"] = pd.Categorical(freq_df["field"], categories=fields, ordered=True)
    freq_df = freq_df.sort_values(["field", "count"], ascending=[True, False]).reset_index(drop=True)
    return freq_df


def _plot_table(freq_df: pd.DataFrame, out_path: Path) -> None:
    show = freq_df.copy()
    show["count"] = show["count"].map(lambda x: f"{x:,}")
    show["percent"] = show["percent"].map(lambda x: f"{x:.2f}%")
    show = show.rename(
        columns={
            "field": "Context field",
            "category": "Category",
            "count": "Frames",
            "percent": "Share",
        }
    )

    fig_h = max(5.5, 0.30 * len(show) + 1.2)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=show[["Context field", "Category", "Frames", "Share"]].values,
        colLabels=["Context field", "Category", "Frames", "Share"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.18)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DDEBF7")
        elif r % 2 == 1:
            cell.set_facecolor("#F7F7F7")

    ax.set_title("A) Context Frequencies Table (Routing Features)", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_frequency_panels(freq_df: pd.DataFrame, out_path: Path) -> None:
    fields = list(freq_df["field"].cat.categories)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    for i, field in enumerate(fields):
        ax = axes[i]
        sub = freq_df[freq_df["field"] == field].copy().sort_values("percent", ascending=True)
        y = np.arange(len(sub))
        ax.barh(y, sub["percent"].values, color="#4C78A8", alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["category"].tolist(), fontsize=9)
        ax.set_xlabel("Share of frames (%)")
        ax.set_title(field)
        ax.grid(axis="x", alpha=0.25)
    for j in range(len(fields), len(axes)):
        axes[j].axis("off")
    fig.suptitle("B) Context Frequency Plots (Routing Features)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    frames_parquet = Path(args.frames_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames_parquet.exists():
        raise FileNotFoundError(f"frames parquet not found: {frames_parquet}")

    needed_cols = [
        "scraped_weather",
        "time_of_day",
        "road_type",
        "road_condition",
        "solar_angle_elevation",
    ]
    df = pd.read_parquet(frames_parquet, columns=needed_cols)
    freq_df = _build_frequency_table(df)

    tag = args.tag.strip() or "final"
    csv_path = out_dir / f"context_field_frequencies_{tag}.csv"
    plot_a_path = out_dir / f"context_field_frequencies_table_{tag}.png"
    plot_b_path = out_dir / f"context_field_frequencies_plot_{tag}.png"

    freq_df.to_csv(csv_path, index=False)
    _plot_table(freq_df, plot_a_path)
    _plot_frequency_panels(freq_df, plot_b_path)

    print(f"Saved frequency CSV  -> {csv_path}")
    print(f"Saved table plot A   -> {plot_a_path}")
    print(f"Saved frequency plot B -> {plot_b_path}")


if __name__ == "__main__":
    main()
