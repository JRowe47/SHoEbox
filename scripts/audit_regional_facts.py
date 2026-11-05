"""Audit causal-supervision continuations and lock validation-only selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.integrated_eval import selection_key
from experiments.text_regions.integrated_train import load_integrated
from scripts.audit_integrated import assert_equal, export_inference
from scripts.audit_integrated_evaluations import audit_one

COUNTERS = ('expert_executions', 'source_occurrences', 'external_events', 'composition_episodes',
            'correction_final_queries', 'views_final_queries', 'correction_episodes_started',
            'views_episodes_started', 'correction_supervised', 'views_supervised')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_parent(parent, initial):
    pmodel, popt, _, _, p = load_integrated(parent)
    imodel, iopt, _, _, i = load_integrated(initial)
    for key in ('rng', 'torch_rng', 'states', 'streams', 'config'):
        assert_equal(p[key], i[key])
    for name, tensor in p['weights'].items(): assert_equal(tensor, i['weights'][name])
    iparams = dict(imodel.named_parameters())
    for name, parameter in pmodel.named_parameters():
        assert_equal(popt.state.get(parameter, {}), iopt.state.get(iparams[name], {}))
    for pg, ig in zip(popt.param_groups, iopt.param_groups):
        assert_equal({k:v for k,v in pg.items() if k != 'params'},
                     {k:v for k,v in ig.items() if k != 'params'})
    for name, parameter in imodel.named_parameters():
        if name not in dict(pmodel.named_parameters()): assert not iopt.state.get(parameter)
    return True


def audit_run(parent, directory):
    load = lambda name: torch.load(directory/name, map_location='cpu', weights_only=True)
    initial, final, best, candidate = map(load, ('initial.pt', 'last.pt', 'best.pt', 'best_candidate.pt'))
    audit_parent(parent, directory/'initial.pt')
    assert final['protocol']['parent_sha256'] == digest(parent)
    for relative, expected in final['protocol']['source_hashes'].items():
        path = (ROOT/'experiments'/relative).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == expected, relative
    assert final['step'] == final['protocol']['steps']
    updates = [row for row in final['history'] if row['step']]
    assert [row['step'] for row in updates] == list(range(1, final['step']+1))
    checks = [row for row in final['history'] if 'validation' in row]
    selected = max(checks, key=lambda row: selection_key(row['validation']))
    ungated = max(checks, key=lambda row: selection_key(row['validation'])[2:])
    assert best['step'] == selected['step'] == final['totals']['best_step']
    assert candidate['step'] == ungated['step'] == final['totals']['candidate_step']
    for key in COUNTERS: assert sum(row[key] for row in updates) == final['totals'][key]
    for row in updates: assert row['gradient_norm'] <= 1.000001
    return {'parent_weights_moments_rng_and_streams_preserved': True,
            'source_hashes_match': True, 'completed_updates': final['step'],
            'selection_recomputed': True, 'selected_step': best['step'],
            'selected_validation': selected['validation'], 'candidate_step': candidate['step'],
            'counters': {key: final['totals'][key] for key in COUNTERS}}, initial, final, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/regional_facts'))
    parser.add_argument('--parent', type=Path, default=Path('artifacts/integrated/selected.pt'))
    parser.add_argument('--evaluations', action='store_true')
    parser.add_argument('--count', type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records, candidates, bundles = {}, [], []
    for name in ('control', 'regional'):
        run = args.root/name
        record, initial, final, best = audit_run(args.parent, run)
        records[name] = record; bundles.append((initial, final))
        candidates.append({'run': name, 'checkpoint': name+'/best.pt', 'step': best['step'],
                           'key': list(selection_key(record['selected_validation'])),
                           'sha256': digest(run/'best.pt'), 'validation': record['selected_validation']})
        if not args.evaluations:
            record['endpoint_export'] = export_inference(run/'last.pt', run/'endpoint_inference.pt')
            record['selected_export'] = export_inference(run/'best.pt', run/'inference.pt')
    (ai, af), (bi, bf) = bundles
    for key in ('config', 'weights', 'optimizer', 'rng', 'torch_rng', 'streams', 'states'):
        assert_equal(ai[key], bi[key])
    ap, bp = dict(ai['protocol']), dict(bi['protocol'])
    assert ap.pop('fact_supervision')['weight'] == 0.
    assert bp.pop('fact_supervision')['weight'] == 1.
    assert_equal(ap, bp)
    for key in ('streams', 'rng', 'torch_rng'): assert_equal(af[key], bf[key])
    ar = [row for row in af['history'] if row['step']]
    br = [row for row in bf['history'] if row['step']]
    assert len(ar) == len(br)
    for left, right in zip(ar, br):
        for key in ('step', 'lr', *COUNTERS): assert left[key] == right[key], (left['step'], key)
    for key in ('composition_nats', 'correction_nats', 'views_nats'):
        assert ar[0][key] == br[0][key]
    selection = max(candidates, key=lambda row: row['key'])
    report = {'all_passed': True, 'matched_steps': af['step'], 'runs': records,
              'initial_weights_moments_rng_and_states_identical': True,
              'all_update_examples_and_execution_budgets_identical': True,
              'selection_uses': 'Validation only; test scores do not rank checkpoints',
              'candidates': candidates, 'selected': selection}
    if args.evaluations:
        lock = json.loads((args.root/'selection.json').read_text())
        assert_equal(lock['selected'], selection)
        assert digest(args.root/'selected.pt') == selection['sha256']
        reports, audits = [], []
        for name in ('control', 'regional'):
            evaluation, audit = audit_one(args.root, name+'/last.pt', name+'/evaluation.json', af['step'], args.count)
            reports.append(evaluation); audits.append(audit)
        for cohort in ('heldout_views', 'independent_views'):
            for control, rows in reports[0][cohort].items():
                for kind, row in rows.items():
                    for key in ('episode_ids','targets','external_events','source_occurrences','expert_executions'):
                        assert row[key] == reports[1][cohort][control][kind][key]
        for control, row in reports[0]['composition_controls'].items():
            for key in ('keys','targets','expert_executions','source_occurrences'):
                assert row[key] == reports[1]['composition_controls'][control][key]
        for gap, tasks in reports[0]['memory_controls'].items():
            for kind, evaluation in tasks.items():
                for control, row in evaluation['controls'].items():
                    other=reports[1]['memory_controls'][gap][kind]['controls'][control]
                    for key in ('episode_ids','episodes','expert_executions'): assert row[key] == other[key]
        report['evaluation_audits'] = audits
        report['prediction_groups_recomputed'] = sum(a['prediction_groups_recomputed'] for a in audits)
        filename = 'evaluation_audit.json'
    else:
        shutil.copyfile(args.root/selection['checkpoint'], args.root/'selected.pt')
        filename = 'selection.json'
    (args.root/filename).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'all_passed': True, 'matched_steps': af['step'],
                      'selected': selection['checkpoint'], 'selected_step': selection['step'],
                      'output': str(args.root/filename)}), flush=True)


if __name__ == '__main__': main()
