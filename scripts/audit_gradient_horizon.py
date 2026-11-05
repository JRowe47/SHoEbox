"""Verify paired training, lock selection, then audit held-out predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.horizon_train import COUNTERS
from experiments.text_regions.integrated_eval import selection_key
from experiments.text_regions.integrated_train import make_streams, load_integrated
from experiments.text_regions.gradient_horizon import batch_digest
from experiments.text_regions.composition_data import training_batch
from scripts.audit_integrated import assert_equal, export_inference
from scripts.audit_integrated_evaluations import audit_one


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_run(root, name):
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    parent = load(root/'parent.pt')
    initial, final, best = [load(root/name/file) for file in ('initial.pt','last.pt','best.pt')]
    for key in ('weights','optimizer','rng','torch_rng','config'):
        assert_equal(parent[key], initial[key])
    assert all(state is None for state in initial['states'].values())
    assert all(state is None for state in final['states'].values())
    assert final['protocol']['parent_sha256'] == digest(root/'parent.pt')
    for relative, expected in final['protocol']['source_hashes'].items():
        path = (ROOT/'experiments'/relative).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == expected, relative
    assert final['step'] == final['protocol']['steps']
    updates = [r for r in final['history'] if r['step']]
    assert [r['step'] for r in updates] == list(range(1, final['step']+1))
    checks = [r for r in final['history'] if 'validation' in r]
    expected = max(checks, key=lambda r: selection_key(r['validation']))
    assert best['step'] == expected['step'] == final['totals']['best_step']
    for key in COUNTERS: assert sum(r[key] for r in updates) == final['totals'][key]
    for row in updates:
        assert row['gradient_norm'] <= 1.000001
        for kind in ('correction','views'):
            assert row[kind+'_final_queries'] == final['protocol']['stream_batch']
            assert row[kind+'_episodes_started'] == final['protocol']['stream_batch']
    # Reconstruct each training cohort from its generators, independently of logs.
    _, _, rng, (streams, _), _ = load_integrated(root/name/'initial.pt')
    fresh = make_streams(final['protocol'])
    for kind in streams: assert_equal(streams[kind].state_dict(), fresh[kind].state_dict())
    for row in updates:
        batches = {'composition': training_batch(rng, final['protocol']['composition_batch'])}
        for kind, stream in streams.items(): batches[kind] = stream.next_chunk(stream.batch.length)
        assert row['batch_sha256'] == {kind:batch_digest(batch) for kind,batch in batches.items()}
    assert_equal(rng.getstate(), final['rng'])
    assert_equal({k:v.state_dict() for k,v in streams.items()}, final['streams'])
    return initial, final, best, {'completed_step':final['step'], 'selected_step':best['step'],
            'selected_validation':expected['validation'], 'parent_preserved':True,
            'fresh_reset_streams_verified':True, 'training_cohort_hashes_regenerated':True,
            'source_hashes_verified':True, 'counters':{k:final['totals'][k] for k in COUNTERS}}


def compare_evaluations(a, b):
    for cohort in ('heldout_views','independent_views'):
        for control, rows in a[cohort].items():
            for kind, row in rows.items():
                for key in ('episode_ids','targets','external_events','source_occurrences','expert_executions'):
                    assert row[key] == b[cohort][control][kind][key]
    for control, row in a['composition_controls'].items():
        for key in ('keys','targets','expert_executions','source_occurrences'):
            assert row[key] == b['composition_controls'][control][key]
    for gap, tasks in a['memory_controls'].items():
        for kind, record in tasks.items():
            for control, row in record['controls'].items():
                for key in ('episode_ids','episodes','expert_executions'):
                    assert row[key] == b['memory_controls'][gap][kind]['controls'][control][key]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/gradient_horizon'))
    p.add_argument('--evaluations', action='store_true')
    args = p.parse_args(); torch.set_num_threads(1)
    lock = json.loads((args.root/'protocol_lock.json').read_text())
    assert digest(ROOT/lock['protocol_path']) == lock['sha256']
    assert digest(args.root/'parent.pt') == lock['parent_sha256']
    bundles, records, candidates = {}, {}, []
    for name in ('truncated','full'):
        initial, final, best, record = audit_run(args.root, name)
        assert final['protocol']['steps'] == lock['steps']
        assert final['protocol']['evaluation_count'] == lock['evaluation_count']
        records[name] = record; bundles[name] = (initial, final, best)
        candidates.append({'run':name, 'checkpoint':name+'/best.pt', 'step':best['step'],
                           'key':list(selection_key(record['selected_validation'])),
                           'sha256':digest(args.root/name/'best.pt')})
    ai, af, _ = bundles['truncated']; bi, bf, _ = bundles['full']
    for key in ('weights','optimizer','config','rng','torch_rng','streams','states'):
        assert_equal(ai[key], bi[key])
    ap, bp = dict(ai['protocol']), dict(bi['protocol'])
    assert ap.pop('detach_between') is True and bp.pop('detach_between') is False
    assert_equal(ap, bp)
    for key in ('rng','torch_rng','streams','states'): assert_equal(af[key], bf[key])
    for a,b in zip(af['history'][1:], bf['history'][1:]):
        for key in (*COUNTERS, 'step','lr','batch_sha256'): assert_equal(a[key], b[key])
    for key in ('composition_nats','correction_nats','views_nats','correction_local_nats','views_local_nats'):
        assert af['history'][1][key] == bf['history'][1][key]
    selected = max(candidates, key=lambda row:row['key'])
    chosen_initial, chosen_final, chosen_best = bundles[selected['run']]
    if selected['step'] == chosen_final['step']:
        assert_equal(chosen_best['weights'], chosen_final['weights'])
        assert_equal(chosen_best['config'], chosen_final['config'])
    report = dict(all_passed=True, runs=records, candidates=candidates, selected=selected,
                  matched_data_budgets_initial_values=True, only_protocol_difference='detach_between',
                  selection_uses='Validation only; held-out scores never rank checkpoints')
    report['selected_endpoint_weights_equal'] = selected['step'] == chosen_final['step']
    if args.evaluations:
        lock = json.loads((args.root/'selection.json').read_text())
        assert_equal(lock['selected'], selected)
        assert digest(args.root/'selected.pt') == selected['sha256']
        evaluated, audits = [], []
        for name in ('truncated','full'):
            path = name+'/evaluation.json'
            result, audit = audit_one(args.root, name+'/last.pt', path, bundles[name][1]['step'], ap['evaluation_count'])
            manifest = result['_manifest']
            assert manifest['checkpoint_sha256'] == digest(args.root/name/'last.pt')
            assert manifest['count'] == ap['evaluation_count']
            assert set(result['_completed']) == {'composition_progress','composition','memory/16/recall',
                'memory/16/correction','memory/32/recall','memory/32/correction','heldout_views',
                'independent/all','independent/a_only','independent/b_only','independent/c_only',
                'independent/no_private','independent/reset_query','capabilities'}
            evaluated.append(result); audits.append(audit)
            diagnostic = json.loads((args.root/name/'local_diagnostic.json').read_text())
            assert diagnostic['checkpoint_sha256'] == digest(args.root/name/'last.pt')
            for task in diagnostic['records'].values():
                for factors in task['nodes'].values():
                    for row in factors.values():
                        for prefix in ('','write_'):
                            correct = sum(a==b for a,b in zip(row[prefix+'predictions'],row['targets']))
                            assert correct == row[prefix+'correct']
                            assert correct/row['count'] == row[prefix+'accuracy']
        compare_evaluations(*evaluated)
        # A selected non-endpoint checkpoint is evaluated independently.
        if selected['step'] != af['step']:
            extra, audit = audit_one(args.root, 'selected.pt', 'selected_evaluation.json', selected['step'], ap['evaluation_count'])
            compare_evaluations(evaluated[0], extra); audits.append(audit)
        report.update(evaluation_audits=audits,
                      prediction_groups_recomputed=sum(a['prediction_groups_recomputed'] for a in audits),
                      local_diagnostic_arithmetic_verified=True)
        name = 'evaluation_audit.json'
    else:
        shutil.copyfile(args.root/selected['checkpoint'], args.root/'selected.pt')
        # Lock the selection on disk before any held-out export checks.
        (args.root/'selection.json').write_text(json.dumps(report, indent=2)+'\n')
        for arm in ('truncated','full'):
            records[arm]['endpoint_export'] = export_inference(args.root/arm/'last.pt', args.root/arm/'endpoint_inference.pt')
        report['selected_export'] = export_inference(args.root/'selected.pt', args.root/'inference.pt')
        name = 'selection.json'
    (args.root/name).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'all_passed':True, 'selected':selected, 'output':str(args.root/name)}), flush=True)


if __name__ == '__main__': main()
