#!/usr/bin/env python3
"""Build context-specific evaluation-only TEST info pickles.

Purpose
    Create filtered context-dependent TEST subsets from an existing finalized
    MMDetection3D-compatible TEST info pickle without changing the original.

Input
    A TEST info pickle (for example: zod_nuscenes_infos_test.pkl) whose top
    level is either:
      - dict containing a list under "data_list" or "infos", or
      - list of per-sample dict entries.

Output
    Additional TEST subset pickle files in the same schema/structure as the
    input pickle, filtered only by context fields. This is evaluation-only:
    no data files are moved/rebuilt and no split membership is changed.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SubsetSpec:
    """Definition of one derived subset."""

    name: str
    source_field: str
    condition_text: str
    allowed_values: Tuple[str, ...]
    output_suffix: str


SUBSET_SPECS: Tuple[SubsetSpec, ...] = (
    # ── 1) Lighting (solar_context_bin) ──
    SubsetSpec(
        name='lighting_test_day',
        source_field='solar_context_bin',
        condition_text='solar_context_bin == "day"',
        allowed_values=('day',),
        output_suffix='lighting_test_day',
    ),
    SubsetSpec(
        name='lighting_test_night',
        source_field='solar_context_bin',
        condition_text='solar_context_bin == "night"',
        allowed_values=('night',),
        output_suffix='lighting_test_night',
    ),
    # ── 2) Scraped weather (scraped_weather) ──
    SubsetSpec(
        name='weather_test_clear_day',
        source_field='scraped_weather',
        condition_text='scraped_weather == "clear-day"',
        allowed_values=('clear-day',),
        output_suffix='weather_test_clear_day',
    ),
    SubsetSpec(
        name='weather_test_partly_cloudy_day',
        source_field='scraped_weather',
        condition_text='scraped_weather == "partly-cloudy-day"',
        allowed_values=('partly-cloudy-day',),
        output_suffix='weather_test_partly_cloudy_day',
    ),
    SubsetSpec(
        name='weather_test_cloudy',
        source_field='scraped_weather',
        condition_text='scraped_weather == "cloudy"',
        allowed_values=('cloudy',),
        output_suffix='weather_test_cloudy',
    ),
    SubsetSpec(
        name='weather_test_partly_cloudy_night',
        source_field='scraped_weather',
        condition_text='scraped_weather == "partly-cloudy-night"',
        allowed_values=('partly-cloudy-night',),
        output_suffix='weather_test_partly_cloudy_night',
    ),
    SubsetSpec(
        name='weather_test_clear_night',
        source_field='scraped_weather',
        condition_text='scraped_weather == "clear-night"',
        allowed_values=('clear-night',),
        output_suffix='weather_test_clear_night',
    ),
    # ── 3) Adverse weather (scraped_weather) ──
    SubsetSpec(
        name='weather_test_precipitation',
        source_field='scraped_weather',
        condition_text='scraped_weather IN ["rain", "snow"]',
        allowed_values=('rain', 'snow'),
        output_suffix='weather_test_precipitation',
    ),
    SubsetSpec(
        name='weather_test_fog',
        source_field='scraped_weather',
        condition_text='scraped_weather == "fog"',
        allowed_values=('fog',),
        output_suffix='weather_test_fog',
    ),
    # ── 4) Complexity (complexity_bin) ──
    SubsetSpec(
        name='complexity_test_low',
        source_field='complexity_bin',
        condition_text='complexity_bin == "low"',
        allowed_values=('low',),
        output_suffix='complexity_test_low',
    ),
    SubsetSpec(
        name='complexity_test_medium',
        source_field='complexity_bin',
        condition_text='complexity_bin == "medium"',
        allowed_values=('medium',),
        output_suffix='complexity_test_medium',
    ),
    SubsetSpec(
        name='complexity_test_high',
        source_field='complexity_bin',
        condition_text='complexity_bin == "high"',
        allowed_values=('high',),
        output_suffix='complexity_test_high',
    ),
    # ── 5) Coarse weather group (weather_group) ──
    SubsetSpec(
        name='weather_group_test_clear_like',
        source_field='weather_group',
        condition_text='weather_group == "clear_like"',
        allowed_values=('clear_like',),
        output_suffix='weather_group_test_clear_like',
    ),
    SubsetSpec(
        name='weather_group_test_cloud_like',
        source_field='weather_group',
        condition_text='weather_group == "cloud_like"',
        allowed_values=('cloud_like',),
        output_suffix='weather_group_test_cloud_like',
    ),
    SubsetSpec(
        name='weather_group_test_precipitation',
        source_field='weather_group',
        condition_text='weather_group == "precipitation"',
        allowed_values=('precipitation',),
        output_suffix='weather_group_test_precipitation',
    ),
    # ── 6) Road type (road_type) ──
    SubsetSpec(
        name='road_type_test_arterial_rural',
        source_field='road_type',
        condition_text='road_type == "arterial-rural"',
        allowed_values=('arterial-rural',),
        output_suffix='road_type_test_arterial_rural',
    ),
    SubsetSpec(
        name='road_type_test_arterial_urban',
        source_field='road_type',
        condition_text='road_type == "arterial-urban"',
        allowed_values=('arterial-urban',),
        output_suffix='road_type_test_arterial_urban',
    ),
    SubsetSpec(
        name='road_type_test_city',
        source_field='road_type',
        condition_text='road_type == "city"',
        allowed_values=('city',),
        output_suffix='road_type_test_city',
    ),
    SubsetSpec(
        name='road_type_test_highway',
        source_field='road_type',
        condition_text='road_type == "highway"',
        allowed_values=('highway',),
        output_suffix='road_type_test_highway',
    ),
    SubsetSpec(
        name='road_type_test_smaller_rural',
        source_field='road_type',
        condition_text='road_type == "smaller-rural"',
        allowed_values=('smaller-rural',),
        output_suffix='road_type_test_smaller_rural',
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create context-specific TEST subsets from an existing TEST info pickle.'
    )
    parser.add_argument(
        '--input-pkl',
        required=True,
        help='Path to the source TEST info pickle (e.g. zod_nuscenes_infos_test.pkl).',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where derived TEST subset pickles are written.',
    )
    parser.add_argument(
        '--splits-dir',
        default=None,
        help='Directory where split .txt files are written (one token per line). '
             'If omitted, inferred as <data_root>/splits next to the infos dir.',
    )
    parser.add_argument(
        '--prefix',
        default='zod_nuscenes_infos',
        help='Filename prefix for info pickle outputs. Default: zod_nuscenes_infos',
    )
    return parser.parse_args()


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path: str, obj: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_sample_list_container(data: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Return (sample_list, container_kind) while preserving top-level schema."""
    if isinstance(data, dict):
        for key in ('data_list', 'infos'):
            if key in data:
                value = data[key]
                if not isinstance(value, list):
                    raise TypeError(
                        f'Expected data["{key}"] to be list, got {type(value).__name__}.'
                    )
                return value, f'dict:{key}'
        raise KeyError('Input dict must contain either "data_list" or "infos".')

    if isinstance(data, list):
        return data, 'list'

    raise TypeError(
        f'Unsupported top-level pickle type: {type(data).__name__}. '
        'Expected dict or list.'
    )


def build_filtered_copy(
    original_data: Any,
    container_kind: str,
    filtered_samples: List[Dict[str, Any]],
) -> Any:
    """Build an output object with the exact top-level schema preserved."""
    if container_kind == 'list':
        return filtered_samples

    assert container_kind.startswith('dict:'), container_kind
    list_key = container_kind.split(':', 1)[1]
    out = dict(original_data)
    out[list_key] = filtered_samples
    return out


def validate_required_fields(
    samples: Sequence[Dict[str, Any]], required_context_fields: Iterable[str]
) -> None:
    """Fail fast with clear errors if context fields are missing."""
    required_set = set(required_context_fields)
    missing_context_indices: List[int] = []
    missing_field_examples: Dict[str, List[int]] = {k: [] for k in sorted(required_set)}

    for i, sample in enumerate(samples):
        ctx = sample.get('context')
        if not isinstance(ctx, dict):
            missing_context_indices.append(i)
            continue
        for field in required_set:
            if field not in ctx and len(missing_field_examples[field]) < 5:
                missing_field_examples[field].append(i)

    if missing_context_indices:
        examples = ', '.join(str(i) for i in missing_context_indices[:10])
        raise ValueError(
            'Missing/invalid "context" field in some samples. '
            f'Example indices: [{examples}]'
        )

    missing_field_msgs = []
    for field, idxs in missing_field_examples.items():
        if idxs:
            idx_text = ', '.join(str(i) for i in idxs)
            missing_field_msgs.append(f'{field}: [{idx_text}]')
    if missing_field_msgs:
        raise ValueError(
            'Required context fields missing in some samples. '
            f'Examples -> {"; ".join(missing_field_msgs)}'
        )


def filter_samples(samples: Sequence[Dict[str, Any]], spec: SubsetSpec) -> List[Dict[str, Any]]:
    allowed = set(spec.allowed_values)
    return [
        sample
        for sample in samples
        if sample['context'][spec.source_field] in allowed
    ]


def extract_tokens(samples: Sequence[Dict[str, Any]]) -> List[str]:
    """Extract per-sample token/frame ID for the split .txt file."""
    tokens: List[str] = []
    for sample in samples:
        tok = sample.get('token', sample.get('sample_idx', None))
        if tok is None:
            raise KeyError(
                'Cannot determine sample token: neither "token" nor '
                '"sample_idx" found in sample dict.'
            )
        tokens.append(str(tok))
    return tokens


def write_split_txt(path: str, tokens: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(tokens))
        if tokens:
            f.write('\n')


def print_summary(
    original_count: int,
    rows: Sequence[Tuple[str, str, str, int, str]],
) -> None:
    headers = ('subset_name', 'source_field', 'condition', 'count', 'output_path')
    col_widths = [
        max(len(headers[i]), max(len(str(row[i])) for row in rows) if rows else 0)
        for i in range(len(headers))
    ]

    def fmt_row(cols: Sequence[Any]) -> str:
        return ' | '.join(str(col).ljust(col_widths[i]) for i, col in enumerate(cols))

    print()
    print(f'Original test count: {original_count}')
    print('Original input pickle was not modified.')
    print('-' * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    print(fmt_row(headers))
    print('-' * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for row in rows:
        print(fmt_row(row))
    print('-' * (sum(col_widths) + 3 * (len(col_widths) - 1)))


def main() -> None:
    args = parse_args()

    input_pkl = os.path.abspath(args.input_pkl)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.splits_dir is not None:
        splits_dir = os.path.abspath(args.splits_dir)
    else:
        splits_dir = os.path.join(os.path.dirname(output_dir), 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    if not os.path.isfile(input_pkl):
        raise FileNotFoundError(f'Input pickle not found: {input_pkl}')

    data = load_pickle(input_pkl)
    samples, container_kind = get_sample_list_container(data)
    original_count = len(samples)

    required_fields = {spec.source_field for spec in SUBSET_SPECS}
    validate_required_fields(samples, required_fields)

    summary_rows: List[Tuple[str, str, str, int, str]] = []
    for spec in SUBSET_SPECS:
        filtered = filter_samples(samples, spec)

        pkl_name = f'{args.prefix}_{spec.output_suffix}.pkl'
        pkl_path = os.path.join(output_dir, pkl_name)
        out_obj = build_filtered_copy(data, container_kind, filtered)
        save_pickle(pkl_path, out_obj)

        txt_name = f'{spec.output_suffix}.txt'
        txt_path = os.path.join(splits_dir, txt_name)
        write_split_txt(txt_path, extract_tokens(filtered))

        summary_rows.append(
            (
                spec.name,
                spec.source_field,
                spec.condition_text,
                len(filtered),
                pkl_path,
            )
        )

    print(f'\nSplit .txt files written to: {splits_dir}')
    print_summary(original_count=original_count, rows=summary_rows)


if __name__ == '__main__':
    main()
