"""
Print key project/data paths resolved by src.paths.

Why this script exists:
- quickly verify what paths the project is using in this environment
- catch path/env mistakes before starting long runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow running as either:
# - python -m scripts.print_paths
# - python scripts/print_paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import paths as P


def _path_to_record(name: str, path: Path) -> dict:
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print resolved project paths.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON instead of a text table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # I keep this explicit so it is obvious which paths we depend on.
    path_items = [
        ("PROJECT_ROOT", P.PROJECT_ROOT),
        ("ZOD_MOE_DATA", P.ZOD_MOE_DATA),
        ("RESIZED_IMAGES_DIR", P.RESIZED_IMAGES_DIR),
        ("SPLITS_DIR", P.SPLITS_DIR),
        ("TRAIN_SPLIT_CSV", P.TRAIN_SPLIT_CSV),
        ("VAL_SPLIT_CSV", P.VAL_SPLIT_CSV),
        ("TEST_SPLIT_CSV", P.TEST_SPLIT_CSV),
        ("OUTPUTS_DIR", P.OUTPUTS_DIR),
        ("INDEX_DIR", P.INDEX_DIR),
        ("EXPORTS_DIR", P.EXPORTS_DIR),
        ("RUNS_DIR", P.RUNS_DIR),
        ("EVAL_DIR", P.EVAL_DIR),
        ("ZODMOE_FRAMES_WITH_BOXES_PARQUET", P.ZODMOE_FRAMES_WITH_BOXES_PARQUET),
    ]

    records = [_path_to_record(name, path) for name, path in path_items]

    if args.json:
        print(json.dumps(records, indent=2))
        return

    max_name = max(len(r["name"]) for r in records)
    for r in records:
        status = "exists" if r["exists"] else "missing"
        kind = "dir" if r["is_dir"] else ("file" if r["is_file"] else "-")
        print(f"{r['name']:<{max_name}}  [{status:7}] [{kind:4}]  {r['path']}")


if __name__ == "__main__":
    main()
