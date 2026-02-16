"""
Aggregate detector run artifacts into comparison tables and plots.

What this script reads
----------------------
- Per-run artifact folders under `outputs/eval/<model_family>/<run_name>/`
- `metrics.json` (evaluation metrics, optional curve payloads)
- `run_metadata.json` (model/split/policy metadata)
- `train_summary.json` (training-time summary)

What this script writes
-----------------------
- `baseline_runs_aggregated.csv`: one row per run with merged metrics + metadata
- `speed_vs_accuracy_table.csv`: compact table for latency/accuracy tradeoff
- `precision_recall_tradeoff.csv`: compact table for precision/recall tradeoff
- `speed_vs_accuracy.png`: scatter plot (x=inference ms/img, y=mAP50-95)
- `precision_recall_tradeoff.png`: scatter plot (x=recall, y=precision)
- `pr_curve_overlay.png` (optional): only if curve arrays exist in run metrics

Why this exists
---------------
The training/eval adapters already save per-run artifacts. This script adds the
final reporting layer that compares many runs side-by-side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Allow running as either:
# - python -m scripts.report_detector_benchmarks
# - python scripts/report_detector_benchmarks.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EVAL_DIR


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for detector benchmark reporting."""
    parser = argparse.ArgumentParser(description="Build detector benchmark reports.")
    parser.add_argument(
        "--eval-root",
        type=str,
        default=str(EVAL_DIR),
        help="Root eval artifact directory (expects family/run layout).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(EVAL_DIR) / "reports"),
        help="Directory to write comparison tables and plots.",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="*",
        default=[],
        help="Optional model families to include (e.g. yolo dino).",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pick_first_numeric(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in row:
            val = _to_float_or_none(row.get(key))
            if val is not None:
                return val
    return None


def _collect_rows(eval_root: Path, families: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pr_curve_rows: list[dict[str, Any]] = []

    if not eval_root.exists():
        return rows, pr_curve_rows

    for family_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
        family = family_dir.name
        if families and family not in families:
            continue
        if family == "reports":
            continue

        for run_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            metrics = _read_json(run_dir / "metrics.json")
            metadata = _read_json(run_dir / "run_metadata.json")
            train_summary = _read_json(run_dir / "train_summary.json")

            # Skip directories with no relevant artifacts.
            if not metrics and not metadata and not train_summary:
                continue

            row: dict[str, Any] = {}
            row.update(metadata)
            row.update(train_summary)
            row.update(metrics)

            row.setdefault("model_family", family)
            row.setdefault("run_name", run_dir.name)
            row["artifact_dir"] = str(run_dir.resolve())

            # Standardized derived columns used by tables/plots.
            row["mAP50"] = _pick_first_numeric(row, ["map50", "mAP50"])
            row["mAP50_95"] = _pick_first_numeric(row, ["map50_95", "mAP50-95", "mAP50_95"])
            row["precision_at_default_conf"] = _pick_first_numeric(row, ["precision"])
            row["recall_at_default_conf"] = _pick_first_numeric(row, ["recall"])
            row["inference_ms_per_img"] = _pick_first_numeric(
                row,
                [
                    "speed_inference_ms_per_img",
                    "inference_ms_per_img",
                    "speed_ms_per_img",
                ],
            )
            row["train_wall_time_s"] = _pick_first_numeric(
                row,
                ["train_wall_time_s", "training_time_s"],
            )
            rows.append(row)

            # Optional per-run PR-style curves (if metrics.json includes curves_results).
            curves = metrics.get("curves_results")
            if isinstance(curves, list):
                for curve in curves:
                    if not isinstance(curve, dict):
                        continue
                    name = str(curve.get("name", ""))
                    x_vals = curve.get("x")
                    y_vals = curve.get("y")
                    if not isinstance(x_vals, list) or not isinstance(y_vals, list):
                        continue
                    if len(x_vals) == 0 or len(y_vals) == 0 or len(x_vals) != len(y_vals):
                        continue
                    # Keep all curve data, but tag probable PR curves for plotting.
                    curve_type = "other"
                    lowered = name.lower()
                    if "precision-recall" in lowered or lowered.startswith("pr"):
                        curve_type = "pr"
                    for x, y in zip(x_vals, y_vals):
                        x_f = _to_float_or_none(x)
                        y_f = _to_float_or_none(y)
                        if x_f is None or y_f is None:
                            continue
                        pr_curve_rows.append(
                            {
                                "model_family": row.get("model_family"),
                                "run_name": row.get("run_name"),
                                "model_variant": row.get("model_variant"),
                                "curve_name": name,
                                "curve_type": curve_type,
                                "x": x_f,
                                "y": y_f,
                            }
                        )

    return rows, pr_curve_rows


def _save_speed_vs_accuracy_plot(df: pd.DataFrame, out_path: Path) -> None:
    usable = df.dropna(subset=["inference_ms_per_img", "mAP50_95"]).copy()
    if usable.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(usable["inference_ms_per_img"], usable["mAP50_95"], alpha=0.85)
    for _, r in usable.iterrows():
        label = str(r.get("model_variant") or r.get("run_name"))
        plt.annotate(label, (r["inference_ms_per_img"], r["mAP50_95"]), fontsize=8, alpha=0.9)
    plt.xlabel("Inference time (ms / image)")
    plt.ylabel("mAP50-95")
    plt.title("Speed vs Accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_precision_recall_plot(df: pd.DataFrame, out_path: Path) -> None:
    usable = df.dropna(subset=["precision_at_default_conf", "recall_at_default_conf"]).copy()
    if usable.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(usable["recall_at_default_conf"], usable["precision_at_default_conf"], alpha=0.85)
    for _, r in usable.iterrows():
        label = str(r.get("model_variant") or r.get("run_name"))
        plt.annotate(
            label,
            (r["recall_at_default_conf"], r["precision_at_default_conf"]),
            fontsize=8,
            alpha=0.9,
        )
    plt.xlabel("Recall (default confidence)")
    plt.ylabel("Precision (default confidence)")
    plt.title("Precision-Recall Tradeoff (Point Comparison)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_pr_curve_overlay(pr_curves_df: pd.DataFrame, out_path: Path) -> bool:
    if pr_curves_df.empty:
        return False
    # Prefer explicit PR curves if names are available; otherwise plot all.
    pr_only = pr_curves_df[pr_curves_df["curve_type"] == "pr"]
    plot_df = pr_only if not pr_only.empty else pr_curves_df
    if plot_df.empty:
        return False

    plt.figure(figsize=(8, 6))
    grouped = plot_df.groupby(["run_name", "model_variant", "curve_name"], dropna=False)
    plotted = 0
    for (run_name, model_variant, curve_name), grp in grouped:
        grp_sorted = grp.sort_values("x")
        if grp_sorted.empty:
            continue
        label_variant = model_variant if isinstance(model_variant, str) and model_variant else run_name
        label = f"{label_variant} | {curve_name}" if isinstance(curve_name, str) and curve_name else str(label_variant)
        plt.plot(grp_sorted["x"], grp_sorted["y"], linewidth=1.6, alpha=0.9, label=label)
        plotted += 1
    if plotted == 0:
        plt.close()
        return False

    plt.xlabel("Recall (or curve x-axis)")
    plt.ylabel("Precision (or curve y-axis)")
    plt.title("PR Curve Comparison (When Available)")
    plt.grid(alpha=0.25)
    if plotted <= 12:
        plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    families = {f.strip() for f in args.families if f.strip()}

    rows, pr_curve_rows = _collect_rows(eval_root=eval_root, families=families)
    if not rows:
        raise ValueError(
            f"No run artifacts found under: {eval_root}. "
            "Expected directories like outputs/eval/<family>/<run_name>/"
        )

    df = pd.DataFrame(rows)
    df.sort_values(by=["model_family", "model_variant", "run_name"], inplace=True, na_position="last")

    agg_csv = out_dir / "baseline_runs_aggregated.csv"
    df.to_csv(agg_csv, index=False)

    speed_table_cols = [
        "model_family",
        "model_variant",
        "run_name",
        "inference_ms_per_img",
        "mAP50",
        "mAP50_95",
        "train_wall_time_s",
        "params_million",
        "flops_billion",
        "unclear_policy",
        "dataset_export_name",
        "seed",
    ]
    speed_table = df[[c for c in speed_table_cols if c in df.columns]].copy()
    speed_table.to_csv(out_dir / "speed_vs_accuracy_table.csv", index=False)

    pr_table_cols = [
        "model_family",
        "model_variant",
        "run_name",
        "precision_at_default_conf",
        "recall_at_default_conf",
        "mAP50",
        "mAP50_95",
        "unclear_policy",
        "dataset_export_name",
        "seed",
    ]
    pr_table = df[[c for c in pr_table_cols if c in df.columns]].copy()
    pr_table.to_csv(out_dir / "precision_recall_tradeoff.csv", index=False)

    _save_speed_vs_accuracy_plot(df, out_dir / "speed_vs_accuracy.png")
    _save_precision_recall_plot(df, out_dir / "precision_recall_tradeoff.png")

    pr_curves_df = pd.DataFrame(pr_curve_rows) if pr_curve_rows else pd.DataFrame()
    if not pr_curves_df.empty:
        pr_curves_df.to_csv(out_dir / "pr_curve_points.csv", index=False)
    has_overlay = _save_pr_curve_overlay(pr_curves_df, out_dir / "pr_curve_overlay.png")

    print(f"Saved aggregated runs table -> {agg_csv}")
    print(f"Saved speed/accuracy table  -> {out_dir / 'speed_vs_accuracy_table.csv'}")
    print(f"Saved precision/recall table-> {out_dir / 'precision_recall_tradeoff.csv'}")
    print(f"Saved speed/accuracy plot   -> {out_dir / 'speed_vs_accuracy.png'}")
    print(f"Saved PR tradeoff plot      -> {out_dir / 'precision_recall_tradeoff.png'}")
    if has_overlay:
        print(f"Saved PR overlay plot       -> {out_dir / 'pr_curve_overlay.png'}")
    elif not pr_curves_df.empty:
        print("PR curve points were found but no overlay lines were drawable.")
    else:
        print("No curve payloads found yet (PR overlay requires curves in metrics.json).")


if __name__ == "__main__":
    main()
