"""Alternate isolated old/new expert runs at the same fused MinKron baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoebox.train_fineweb import atomic_json, source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Need at least one matched pair.')
    args.data, args.out = args.data.resolve(), args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use an empty output directory.')
    args.out.mkdir(parents=True, exist_ok=True)
    digest = source_digest()
    rows = {'baseline': [], 'indexed': []}
    differences = []
    for repeat in range(args.repeats):
        order = ('baseline', 'indexed') if repeat % 2 == 0 else ('indexed', 'baseline')
        for mode in order:
            out = args.out/f'{mode}_{repeat+1}'
            env = {**os.environ, 'SHOEBOX_FUSED':'1', 'SHOEBOX_EXPERTS':'0' if mode == 'baseline' else '1'}
            command = [sys.executable, '-m', 'shoebox.profile_wide', '--mode', 'benchmark',
                       '--backend', 'fused', '--data', str(args.data), '--out', str(out),
                       '--targets', '1000', '--batch', '4', '--context', '32', '--threads', '1']
            with (args.out/f'{mode}_{repeat+1}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads((out/'benchmark_fused.json').read_text())
            assert row['expert_kernel']['requested'] == (mode == 'indexed')
            rows[mode].append(row)
            print(json.dumps({'repeat': repeat+1, 'mode':mode,
                              'seconds':row['forward_seconds']+row['backward_seconds'],
                              'peak_rss_mib':row['peak_rss_mib']}), flush=True)
        left = torch.load(args.out/f'baseline_{repeat+1}/benchmark_fused_logits.pt', weights_only=True)
        right = torch.load(args.out/f'indexed_{repeat+1}/benchmark_fused_logits.pt', weights_only=True)
        delta = float((left-right).abs().max())
        same = bool((left.argmax(-1) == right.argmax(-1)).all())
        assert delta < 1e-5 and same
        differences.append({'max_logit_difference': delta, 'greedy_predictions_identical': same})
    assert digest == source_digest(), 'Source changed during measurement.'
    medians = {}
    for mode, records in rows.items():
        fields = ('forward_seconds', 'backward_seconds', 'bytes_per_second', 'peak_rss_mib',
                  'saved_unique_storage_bytes', 'saved_tensor_calls', 'rss_after_forward_mib')
        medians[mode] = {key: statistics.median(r[key] for r in records) for key in fields}
        medians[mode]['total_seconds'] = statistics.median(r['forward_seconds']+r['backward_seconds'] for r in records)
        medians[mode]['total_seconds_range'] = [min(r['forward_seconds']+r['backward_seconds'] for r in records),
                                                max(r['forward_seconds']+r['backward_seconds'] for r in records)]
    old, new = medians['baseline'], medians['indexed']
    summary = {'source_sha256': digest, 'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
               'repeats': args.repeats, 'parameters':rows['baseline'][0]['parameters'],
               'batch':4, 'context':32, 'targets_per_batch':128, 'threads':1,
               'torch': str(torch.__version__), 'medians':medians, 'runs':rows, 'comparisons':differences,
               'throughput_gain_percent':100*(new['bytes_per_second']/old['bytes_per_second']-1),
               'time_reduction_percent':100*(1-new['total_seconds']/old['total_seconds']),
               'peak_rss_reduction_percent':100*(1-new['peak_rss_mib']/old['peak_rss_mib']),
               'saved_storage_reduction_percent':100*(1-new['saved_unique_storage_bytes']/old['saved_unique_storage_bytes'])}
    atomic_json(args.out/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
