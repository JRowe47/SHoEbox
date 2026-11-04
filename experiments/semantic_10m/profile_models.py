"""Isolated CPU microbenchmark for the two full-lace architectures.

Run after training, with no other training job active. This times the same
batch/sequence shape and records profiler scope explicitly; it is not a dense
Transformer benchmark. Run models in separate subprocesses for clean peak RSS.
"""
import argparse
from dataclasses import asdict
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from model import Config, SHoEGPT


def measure(scale, threads, steps):
    torch.set_num_threads(threads)
    torch.manual_seed(20261103)
    model = SHoEGPT(Config.scaled_10m() if scale == '10m' else Config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9,.95), weight_decay=.01)
    tokens = torch.randint(65, (4, 128))
    targets = torch.randint(65, (4, 128))
    train_times = []
    for i in range(steps + 3):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        result = model(tokens, targets)
        (result['ce'] + .01*result['aux']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        elapsed = time.perf_counter()-tick
        if i >= 3:
            train_times.append(elapsed)
    model.eval()
    forward_times = []
    with torch.inference_mode():
        for i in range(steps + 3):
            tick = time.perf_counter()
            model(tokens)
            elapsed = time.perf_counter()-tick
            if i >= 3:
                forward_times.append(elapsed)
    return {
        'scale': scale, 'config': asdict(model.cfg), 'parameters': model.parameter_summary(),
        'threads': threads, 'batch': 4, 'sequence': 128, 'warmup_steps': 3,
        'measured_steps': steps, 'torch': str(torch.__version__),
        'train_step_seconds_median': statistics.median(train_times),
        'train_step_seconds_mean': statistics.mean(train_times),
        'train_characters_per_second_median': 512/statistics.median(train_times),
        'forward_seconds_median': statistics.median(forward_times),
        'forward_characters_per_second_median': 512/statistics.median(forward_times),
        'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'train_step_seconds': train_times, 'forward_seconds': forward_times,
        'scope': 'randomly initialized full-lace models; full train step incl. zero_grad, CE+aux, backward, clip, AdamW; forward is a complete 512-character batch, not cached decoding',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=['1m','10m'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--out', default='performance.json')
    args = parser.parse_args()
    if args.scale:
        print(json.dumps(measure(args.scale, args.threads, args.steps)))
    else:
        records = []
        for scale in ['1m', '10m']:
            output = subprocess.check_output([
                sys.executable, __file__, '--scale', scale, '--threads', str(args.threads),
                '--steps', str(args.steps)], text=True)
            records.append(json.loads(output))
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(json.dumps(records, indent=2))
