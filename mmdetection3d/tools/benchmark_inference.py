#!/usr/bin/env python
"""
Customized script for MoE Thesis
Offline inference benchmarking for trained MMDetection3D checkpoints.

Measures pure model-forward latency (no loss computation, no metric eval)
using the test pipeline.  Includes warm-up, CUDA synchronization, and
per-sample timing for accurate GPU latency measurement.

Usage (single GPU, after training):
    python tools/benchmark_inference.py \\
        configs/zod/zod_lidar_only.py \\
        /path/to/epoch_20.pth \\
        --num-samples 200 \\
        --warmup 20

On Slurm (interactive):
    srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 \\
        --constraint=40ga100 --time=00:30:00 --account=p201222 --qos=default \\
        python tools/benchmark_inference.py \\
            configs/zod/zod_lidar_only.py /path/to/checkpoint.pth

Output (stdout + optional JSON):
    - average latency (ms/sample)
    - median latency
    - p95 latency
    - throughput (samples/sec)
    - peak GPU memory (MB)
"""

import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(
        description='Benchmark inference latency / throughput')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint file')
    p.add_argument('--num-samples', type=int, default=200,
                   help='number of samples to benchmark (after warmup)')
    p.add_argument('--warmup', type=int, default=20,
                   help='warmup iterations (not timed)')
    p.add_argument('--batch-size', type=int, default=1,
                   help='inference batch size (default 1 for latency)')
    p.add_argument('--out-json', type=str, default=None,
                   help='save results to JSON file')
    p.add_argument('--cfg-options', nargs='+', default=None,
                   help='override config options (key=value pairs)')
    return p.parse_args()


def build_model_and_dataloader(cfg, checkpoint_path, batch_size=1):
    """Build model in eval mode and a test dataloader."""
    cfg.launcher = 'none'
    cfg.work_dir = osp.join('./work_dirs', '_benchmark_tmp')
    os.makedirs(cfg.work_dir, exist_ok=True)

    cfg.test_dataloader.batch_size = batch_size
    cfg.test_dataloader.num_workers = 2
    if 'persistent_workers' in cfg.test_dataloader:
        cfg.test_dataloader.persistent_workers = (
            cfg.test_dataloader.num_workers > 0)

    runner = Runner.from_cfg(cfg)
    model = runner.model
    if is_model_wrapper(model):
        model = model.module
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cuda().eval()

    dataloader = runner.test_dataloader
    return model, dataloader, runner


@torch.no_grad()
def run_benchmark(model, dataloader, num_samples, warmup, batch_size):
    """Time model.predict() with proper CUDA sync."""
    latencies = []
    total_done = 0
    warmup_done = 0

    torch.cuda.reset_peak_memory_stats()

    for data_batch in dataloader:
        data = model.data_preprocessor(data_batch, training=False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        if warmup_done < warmup:
            model.predict(batch_inputs, batch_data_samples)
            warmup_done += 1
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(batch_inputs, batch_data_samples)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        total_done += 1
        if total_done >= num_samples:
            break

    if total_done == 0:
        print('ERROR: no samples benchmarked (dataset too small for warmup?)')
        return {}

    latencies_np = np.array(latencies)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    results = {
        'num_samples': total_done,
        'batch_size': batch_size,
        'warmup': warmup,
        'avg_latency_ms': float(np.mean(latencies_np)),
        'median_latency_ms': float(np.median(latencies_np)),
        'p95_latency_ms': float(np.percentile(latencies_np, 95)),
        'p99_latency_ms': float(np.percentile(latencies_np, 99)),
        'min_latency_ms': float(np.min(latencies_np)),
        'max_latency_ms': float(np.max(latencies_np)),
        'throughput_samples_per_sec': float(
            1000.0 * batch_size / np.mean(latencies_np)),
        'peak_gpu_memory_mb': round(peak_mem_mb, 1),
    }
    return results


def print_results(results):
    sep = '=' * 60
    print(f'\n{sep}')
    print('  INFERENCE BENCHMARK RESULTS')
    print(sep)
    print(f"  samples benchmarked  : {results['num_samples']}")
    print(f"  batch size           : {results['batch_size']}")
    print(f"  warmup iterations    : {results['warmup']}")
    print(f"  avg latency          : {results['avg_latency_ms']:.2f} ms")
    print(f"  median latency       : {results['median_latency_ms']:.2f} ms")
    print(f"  p95 latency          : {results['p95_latency_ms']:.2f} ms")
    print(f"  p99 latency          : {results['p99_latency_ms']:.2f} ms")
    print(f"  min latency          : {results['min_latency_ms']:.2f} ms")
    print(f"  max latency          : {results['max_latency_ms']:.2f} ms")
    print(f"  throughput           : "
          f"{results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  peak GPU memory      : {results['peak_gpu_memory_mb']:.0f} MB")
    print(sep)
    print()
    print('  NOTE: These timings measure end-to-end model.predict() which')
    print('  includes data preprocessing (voxelization), network forward,')
    print('  and post-processing (NMS / bbox decoding). Data loading from')
    print('  disk is excluded (batch is already in memory when timing starts).')
    print()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        from mmengine.config import DictAction
        options = {}
        for item in args.cfg_options:
            k, v = item.split('=', 1)
            options[k] = v
        cfg.merge_from_dict(options)

    model, dataloader, runner = build_model_and_dataloader(
        cfg, args.checkpoint, args.batch_size)

    results = run_benchmark(
        model, dataloader, args.num_samples, args.warmup, args.batch_size)

    if not results:
        return

    results['config'] = args.config
    results['checkpoint'] = args.checkpoint

    print_results(results)

    if args.out_json:
        os.makedirs(osp.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.out_json}')


if __name__ == '__main__':
    main()
