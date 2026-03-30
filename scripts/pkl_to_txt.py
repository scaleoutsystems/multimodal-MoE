#!/usr/bin/env python3

import argparse
import os
import pickle
import pprint
from typing import Any, Tuple

try:
    import pickle5  # type: ignore
except ImportError:
    pickle5 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the first item from one or more pickle files into .txt files."
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/infos",
        help="Directory containing .pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/projects/multimodal-MoE/outputs/analysis/bev_dataset"),
        help="Directory to write output .txt files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "zod_nuscenes_infos_test.pkl",
            "zod_nuscenes_infos_train.pkl",
            "zod_nuscenes_infos_val.pkl",
        ],
        help="List of .pkl filenames (relative to --input-dir).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="Line width used by pprint formatting.",
    )
    return parser.parse_args()


def get_first_item(data: Any) -> Tuple[Any, str, int]:
    """
    Return (first_item, source_description, total_items).
    Supports common pickle top-level structures:
    - list/tuple: first element
    - dict with data_list/infos key: first element from that list
    - generic dict: first key-value pair
    """
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("Top-level list/tuple is empty.")
        return data[0], "top-level list/tuple", len(data)

    if isinstance(data, dict):
        for key in ("data_list", "infos"):
            if key in data and isinstance(data[key], (list, tuple)):
                if not data[key]:
                    raise ValueError(f"Dictionary key '{key}' exists but is empty.")
                return data[key][0], f"dict['{key}']", len(data[key])

        if not data:
            raise ValueError("Top-level dict is empty.")
        first_key = next(iter(data))
        return {first_key: data[first_key]}, f"dict first key '{first_key}'", len(data)

    raise TypeError(
        f"Unsupported top-level type: {type(data)}. Expected list, tuple, or dict."
    )


def load_pickle_compat(path: str) -> Any:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ValueError as exc:
            if "unsupported pickle protocol" not in str(exc):
                raise
            if pickle5 is None:
                raise RuntimeError(
                    "This file requires pickle protocol 5. Use Python >=3.8, or install "
                    "'pickle5' in your current environment and re-run."
                ) from exc
            f.seek(0)
            return pickle5.load(f)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for pkl_filename in args.files:
        input_path = os.path.join(args.input_dir, pkl_filename)
        base_name = os.path.splitext(pkl_filename)[0]
        output_name = f"{base_name}_first_item.txt"
        output_path = os.path.join(args.output_dir, output_name)

        print(f"Processing: {input_path}")
        if not os.path.isfile(input_path):
            print(f"  ERROR: file not found: {input_path}\n")
            continue

        try:
            data = load_pickle_compat(input_path)

            first_item, source_desc, total_items = get_first_item(data)

            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(f"Source file : {input_path}\n")
                out_f.write(f"Top-level type : {type(data)}\n")
                out_f.write(f"Total items : {total_items}\n")
                out_f.write(f"First item source : {source_desc}\n")
                out_f.write("=" * 60 + "\n\n")
                out_f.write(pprint.pformat(first_item, width=args.width))
                out_f.write("\n")

            print(f"  Loaded object type : {type(data)}")
            print(f"  First item source  : {source_desc}")
            print(f"  Written to         : {output_path}\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")

    print("Done.")


if __name__ == "__main__":
    main()