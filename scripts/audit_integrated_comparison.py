"""Audit matched task exposure and select across runs using validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.audit_integrated import assert_equal
from experiments.text_regions.integrated_eval import selection_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/integrated'))
    args = parser.parse_args()
    root = args.root
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    original = load(root/'joint/initial.pt')
    balanced_initial = load(root/'balanced/initial.pt')
    baseline = load(root/'baseline_step100.pt')
    balanced = load(root/'balanced/last.pt')
    for key in ('weights', 'optimizer', 'rng', 'torch_rng', 'streams', 'states'):
        assert_equal(original[key], balanced_initial[key])
    assert_equal(original['config'], balanced_initial['config'])
    assert_equal(baseline['streams'], balanced['streams'])
    assert_equal(baseline['rng'], balanced['rng'])
    assert_equal(baseline['torch_rng'], balanced['torch_rng'])
    assert baseline['step'] == balanced['step'] == 100
    allowed = {'steps', 'source_hashes', 'gradient_policy', 'optimizer', 'matched_control'}
    for key in (original['protocol'].keys() | balanced_initial['protocol'].keys())-allowed:
        assert original['protocol'][key] == balanced_initial['protocol'][key], key
    for path, digest in original['protocol']['source_hashes'].items():
        assert balanced_initial['protocol']['source_hashes'][path] == digest
    counters = ('expert_executions', 'source_occurrences', 'external_events',
                'composition_episodes', 'correction_final_queries', 'views_final_queries',
                'correction_episodes_started', 'views_episodes_started',
                'correction_supervised', 'views_supervised')
    for key in counters: assert baseline['totals'][key] == balanced['totals'][key], key
    baseline_rows = [row for row in baseline['history'] if row['step']]
    balanced_rows = [row for row in balanced['history'] if row['step']]
    for left, right in zip(baseline_rows, balanced_rows):
        assert left['step'] == right['step']
        for key in (*counters, 'lr'): assert left[key] == right[key], (left['step'], key)
        assert right['gradient_norm'] <= 1.000001
    for key in ('composition_nats', 'correction_nats', 'views_nats'):
        assert baseline_rows[0][key] == balanced_rows[0][key]
    accepted, candidates = [], []
    for run in ('joint', 'balanced'):
        endpoint = load(root/run/'last.pt')
        assert endpoint['step'] == endpoint['protocol']['steps']
        for filename, records, sliced in [('best.pt', accepted, False),
                                           ('best_candidate.pt', candidates, True)]:
            path = root/run/filename
            checkpoint = load(path)
            validation = checkpoint['history'][-1]['validation']
            key = selection_key(validation)
            records.append({'run': run, 'checkpoint': str(path.relative_to(root)),
                            'step': checkpoint['step'], 'validation': validation,
                            'key': list(key[2:] if sliced else key),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    selected = max(accepted, key=lambda row: row['key'])
    diagnostic = max(candidates, key=lambda row: row['key'])
    report = {'selection_uses': 'validation only; test results not opened by this script',
              'initial_weights_optimizer_rng_and_states_identical': True,
              'matched_sampling_rng_and_stream_cursors_identical': True,
              'matched_steps': 100,
              'matched_counters': {key: baseline['totals'][key] for key in counters},
              'all_update_budgets_identical': True,
              'selected': selected, 'diagnostic_candidate': diagnostic,
              'accepted_candidates': accepted, 'diagnostic_candidates': candidates}
    (root/'selection.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copyfile(root/selected['checkpoint'], root/'selected.pt')
    shutil.copyfile(root/diagnostic['checkpoint'], root/'diagnostic_candidate.pt')
    print(json.dumps({'matched_counters': report['matched_counters'],
                      'selected': {k: selected[k] for k in ('checkpoint', 'step')},
                      'diagnostic': {k: diagnostic[k] for k in ('checkpoint', 'step')}}), flush=True)


if __name__ == '__main__': main()
