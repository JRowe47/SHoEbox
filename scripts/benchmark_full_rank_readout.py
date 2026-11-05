"""Measure decoder-only CPU prediction latency after all fitting has finished."""
import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import statistics
import sys

import torch
from torch.utils.benchmark import Timer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.cohort_diversity import bundles_for_count
from experiments.text_regions.compiled_readout import compile_readout
from experiments.text_regions.emission_adapter import ADAPTER_SEEDS, sha
from experiments.text_regions.full_rank_readout import bundle_for_count, load_inputs, source_hashes
from experiments.text_regions.residual_adapter import write_json


def benchmark(source, prior_source, *, repeats=3, min_seconds=.05):
    if repeats < 1 or min_seconds <= 0:
        raise ValueError('Positive timing settings required.')
    status = json.loads((source/'status.json').read_text())
    if not status['complete']:
        raise ValueError('Finish all fits before benchmarking.')
    manifest = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())
    rows = []; inputs = {}
    for row in manifest['runs']:
        base, training, validation, evaluations, previous, _ = load_inputs(prior_source, row)
        fitted = torch.load(source/row['name']/'fitted.pt', map_location='cpu', weights_only=True)
        full = bundle_for_count(256, base, training[256], validation, fitted['heads'])
        prior = bundles_for_count(256, training[256], validation, base, previous['heads'])['selected']
        seed = ADAPTER_SEEDS[0]
        heads = {'base': base, 'linear_preserved_selected': full['heads']['preserved'],
            'identity_selected': prior['heads'][f'identity/{seed}'],
            'gelu_selected': prior['heads'][f'gelu/{seed}'],
            'identity_full': replace(prior['heads'][f'identity/{seed}'], strength=1.),
            'gelu_full': replace(prior['heads'][f'gelu/{seed}'], strength=1.)}
        inputs[row['name']] = {'fitted_sha256': sha(source/row['name']/'fitted.pt'),
            'test_inputs_sha256': evaluations['test']['protocol']['inputs_sha256']}
        with torch.inference_mode():
            for family, head in heads.items():
                compiled = compile_readout(head)
                for batch in (1, 64):
                    x = evaluations['test']['features']['final/raw'][:batch].clone()
                    if not torch.equal(head.predict(x), compiled.predict(x)):
                        raise ValueError('Benchmark compilation changed predicted bytes.')
                    times = {'reference': [], 'compiled': []}
                    # Alternate order to reduce systematic warmup/order effects.
                    for repeat in range(repeats):
                        pairs = [('reference', head), ('compiled', compiled)]
                        if repeat % 2: pairs.reverse()
                        for name, h in pairs:
                            timer = Timer('head.predict(features)', globals={'head': h, 'features': x}, num_threads=1)
                            measurement = timer.blocked_autorange(min_run_time=min_seconds)
                            times[name].append(measurement.median*1e6)
                    medians = {n: statistics.median(t) for n, t in times.items()}
                    rows.append({'graph': row['name'], 'family': family, 'batch': batch,
                        'strength': getattr(head, 'strength', None), 'microseconds_per_batch': medians,
                        'repeat_microseconds': times, 'compiled_cost': compiled.cost(),
                        'speedup': medians['reference']/medians['compiled']})
        print(json.dumps({'timed_graph': row['name'], 'completed_rows': len(rows)}), flush=True)
    groups = {}
    for family in sorted({r['family'] for r in rows}):
        groups[family] = {}
        for batch in (1, 64):
            members = [r for r in rows if r['family'] == family and r['batch'] == batch]
            groups[family][str(batch)] = {n: statistics.median(r['microseconds_per_batch'][n] for r in members)
                for n in ('reference', 'compiled')}
            groups[family][str(batch)]['median_paired_speedup'] = statistics.median(r['speedup'] for r in members)
    cpu = 'unknown'
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
            if s.startswith('model name')), cpu)
    return {'schema_version': 1, 'scope': 'decoder predict only: input conversion, finite check, logits and argmax; excludes graph execution',
        'settings': {'threads': 1, 'repeats': repeats, 'min_seconds_per_repeat': min_seconds,
            'batch_sizes': [1, 64], 'normalization_folded': True, 'output_dtype': 'float64',
            'input_dtype': str(x.dtype), 'correction_seed': seed, 'graphs': 12},
        'runtime': {'cpu': cpu, 'torch': str(torch.__version__), 'python': platform.python_version(), 'platform': platform.platform()},
        'source_sha256': {**source_hashes(), str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        'inputs': inputs, 'groups': groups, 'measurements': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('artifacts/full_rank_v1'))
    p.add_argument('--prior-source', type=Path, default=Path('artifacts/cohort_reproduction_v1'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    result = benchmark(ROOT/args.source, ROOT/args.prior_source)
    args.out.parent.mkdir(parents=True, exist_ok=True); write_json(args.out, result)


if __name__ == '__main__':
    main()
