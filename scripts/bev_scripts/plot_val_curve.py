"""
Plot a per-epoch validation metric curve from a completed training run.

Reads per-epoch metric values from the MMEngine scalars log
(``<run_dir>/<timestamp>/vis_data/scalars.json``) and writes a JSON sidecar
and PNG into ``<run_dir>/visualizations/``.  The output filenames are derived
from the metric key unless overridden with ``--output-name``.

Usage
-----
    python scripts/plot_val_curve.py \\
        --run-dir outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4445346 \\
        --metric mAP_0.50

    python scripts/plot_val_curve.py \\
        --run-dir outputs/runs/zod_lidar_only/zod-lidar-only_4440636 \\
        --metric mAP_1.0m \\
        --output-name val_curve_ap_1m

    # List all available metrics in a run without plotting:
    python scripts/plot_val_curve.py \\
        --run-dir outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4445346 \\
        --list-metrics

Outputs
-------
    <run_dir>/visualizations/<output_name>.json   – epoch/value pairs
    <run_dir>/visualizations/<output_name>.png    – annotated curve plot
"""

import argparse
import json
import os
import os.path as osp
import re
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_scalars(run_dir: str) -> str:
    """Return path to the scalars.json inside the run's vis_data folder.

    MMEngine writes logs under ``<run_dir>/<timestamp>/vis_data/scalars.json``.
    If multiple timestamp directories exist, the most-recently-modified one is
    used.
    """
    candidates = []
    for entry in os.scandir(run_dir):
        if not entry.is_dir():
            continue
        path = osp.join(entry.path, 'vis_data', 'scalars.json')
        if osp.isfile(path):
            candidates.append((entry.stat().st_mtime, path))
    if not candidates:
        sys.exit(
            f'[plot_val_curve] No vis_data/scalars.json found under {run_dir!r}.\n'
            'Make sure the run directory is correct and training has completed.'
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _load_val_records(scalars_path: str) -> list[dict]:
    """Return only the validation records (those that contain metric keys)."""
    records = []
    with open(scalars_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # Validation records contain metric keys like mAP_* or AP_*; training
    # records contain loss/lr keys.  Distinguish by presence of any AP key.
    return [r for r in records if any(k.startswith(('mAP', 'AP', 'pedestrian')) for k in r)]


def _metric_key(name: str) -> str:
    """Convert a metric name to a safe filename stem, e.g. 'mAP_0.50' → 'val_curve_mAP_0_50'."""
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return f'val_curve_{safe}'


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_curve(run_dir: str, metric: str, output_name: str | None, dpi: int = 150) -> None:
    scalars_path = _find_scalars(run_dir)
    val_records  = _load_val_records(scalars_path)

    if not val_records:
        sys.exit(f'[plot_val_curve] No validation records found in {scalars_path!r}.')

    # Verify metric exists
    available = sorted({k for r in val_records for k in r if not k.startswith(('data_time', 'time', 'step', 'epoch', 'iter', 'memory'))})
    if metric not in available:
        print(f'[plot_val_curve] Metric {metric!r} not found.\nAvailable metrics:')
        for m in available:
            print(f'  {m}')
        sys.exit(1)

    # Extract (epoch, value) pairs — use 'step' as the 1-based epoch index
    pts = [(int(r['step']), float(r[metric])) for r in val_records if metric in r]
    if not pts:
        sys.exit(f'[plot_val_curve] Metric {metric!r} has no data points.')

    filename = output_name or _metric_key(metric)
    vis_dir  = osp.join(run_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # ── JSON sidecar ──────────────────────────────────────────────────────────
    json_path = osp.join(vis_dir, f'{filename}.json')
    with open(json_path, 'w') as f:
        json.dump({metric: pts}, f, indent=2)
    print(f'[plot_val_curve] Wrote {json_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs, values = zip(*pts)
    best_val = max(values)
    best_ep  = epochs[values.index(best_val)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values, marker='o', markersize=4, label=metric)
    ax.annotate(
        f'{best_val:.4f}',
        xy=(best_ep, best_val),
        xytext=(0, 8),
        textcoords='offset points',
        fontsize=8,
        ha='center',
        color='green',
        fontweight='bold',
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AP')
    ax.set_title(f'Validation {metric} over training')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xticks(epochs)
    fig.tight_layout()

    png_path = osp.join(vis_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f'[plot_val_curve] Wrote {png_path}')
    print(f'[plot_val_curve] Best {metric}: {best_val:.4f} at epoch {best_ep}')


def list_metrics(run_dir: str) -> None:
    scalars_path = _find_scalars(run_dir)
    val_records  = _load_val_records(scalars_path)
    if not val_records:
        sys.exit(f'[plot_val_curve] No validation records found in {scalars_path!r}.')
    available = sorted({k for r in val_records for k in r if not k.startswith(('data_time', 'time', 'step', 'epoch', 'iter', 'memory'))})
    print(f'Available metrics in {run_dir!r}:')
    for m in available:
        print(f'  {m}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--run-dir', required=True,
        help='Path to the run output directory (contains the timestamp sub-folder).',
    )
    parser.add_argument(
        '--metric',
        help='Metric key to plot, e.g. mAP_0.50, mAP_1.0m, pedestrian_AP_0.50.',
    )
    parser.add_argument(
        '--output-name',
        help='Output filename stem (without extension). '
             'Defaults to val_curve_<sanitised_metric>.',
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='PNG resolution (default: 150).',
    )
    parser.add_argument(
        '--list-metrics', action='store_true',
        help='Print all available metric keys and exit.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_metrics:
        list_metrics(args.run_dir)
        return
    if not args.metric:
        sys.exit('[plot_val_curve] --metric is required unless --list-metrics is set.')
    plot_val_curve(args.run_dir, args.metric, args.output_name, args.dpi)


if __name__ == '__main__':
    main()
