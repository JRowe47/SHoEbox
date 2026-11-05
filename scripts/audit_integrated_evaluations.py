"""Recompute reported predictions, generated labels and matched evaluation costs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.audit_integrated import assert_equal, audit_predictions
from experiments.text_regions.composition_data import batch_from_keys, evaluation_keys, shuffled_bridge
from experiments.text_regions.view_data import make_view_batch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_one(root, checkpoint_path, evaluation_path, expected_step):
    checkpoint = torch.load(root/checkpoint_path, map_location='cpu', weights_only=True)
    report = json.loads((root/evaluation_path).read_text())
    assert report['checkpoint_step'] == checkpoint['step'] == expected_step
    groups = audit_predictions(report)
    seed = checkpoint['protocol']['seed']
    keys = evaluation_keys('test')
    original = batch_from_keys(keys)
    changed = [shuffled_bridge(batch_from_keys(keys[start:start+64]))
               for start in range(0, len(keys), 64)]
    changed_keys = [list(key) for batch in changed for key in batch.keys]
    changed_answers = [answer for batch in changed for answer in batch.answers.tolist()]
    controls = report['composition_controls']
    for name, row in controls.items():
        assert row['episodes'] == len(keys) == 592
        assert row['keys'] == (changed_keys if name in ('k2_shuffled_bridge', 'k2_counterfactual')
                               else [list(key) for key in keys])
        assert row['targets'] == (changed_answers if name == 'k2_counterfactual' else original.answers.tolist())
        rounds = 1 if name == 'k1' else 4 if name == 'k4' else 2
        assert row['prefix_expert_executions'] == 96*len(keys)
        assert row['query_expert_executions'] == 84*rounds*len(keys)
        assert row['expert_executions'] == (96+84*rounds)*len(keys)
    assert controls['k2_shuffled_bridge']['predictions'] == controls['k2_counterfactual']['predictions']
    metrics = report['test_capabilities']['composition']['metrics']
    for name, predicate in [('overall', lambda a,b,q: True),
                             ('lookup_b', lambda a,b,q: a != 0 and b == 0),
                             ('lookup_c', lambda a,b,q: a == 0 and b != 0),
                             ('composition', lambda a,b,q: a != 0 and b != 0)]:
        mask = [predicate(*key) for key in keys]
        correct = sum(enabled and pred == target for enabled, pred, target in
                      zip(mask, controls['k2']['predictions'], controls['k2']['targets']))
        assert metrics[name] == dict(correct=correct, count=sum(mask), accuracy=correct/sum(mask))
    for cohort, split, cohort_seed in [('heldout_views', 'test', seed),
                                       ('independent_views', 'independent', seed+104729)]:
        for control, rows in report[cohort].items():
            for kind, row in rows.items():
                batch = make_view_batch(split, 0, row['episodes'], seed=cohort_seed,
                                        gap=8, control=control, query_type=kind)
                assert row['episode_ids'] == list(batch.identities)
                assert row['targets'] == batch.answers.tolist()
                assert row['episodes'] == 192
                assert row['external_events'] == len(batch.identities)*batch.length
                assert row['expert_executions'] == 168*len(batch.identities)*batch.length
                for value, class_row in row['by_answer'].items():
                    mask = [target == ord(value) for target in row['targets']]
                    assert class_row['episodes'] == sum(mask) == 48
                    assert class_row['correct'] == sum(enabled and pred == target for enabled,pred,target
                                                      in zip(mask,row['predictions'],row['targets']))
    capabilities = report['test_capabilities']
    for gap, tasks in report['memory_controls'].items():
        for evaluation in tasks.values():
            assert evaluation['seed'] == seed
            assert evaluation['split'] == 'test' and evaluation['gap'] == int(gap)
            assert evaluation['query_every'] == 4
            assert all(row['episodes'] == 192 for row in evaluation['controls'].values())
    for kind in ('recall', 'correction'):
        assert capabilities['scores'][kind] == report['memory_controls']['16'][kind]['controls']['carried']['exact_accuracy']
    for kind in ('shared', 'left', 'right'):
        assert capabilities['scores'][kind] == report['heldout_views']['all'][kind]['exact_accuracy']
    for control, kinds in [('a_only', ('shared','left')), ('b_only', ('shared',)), ('c_only', ('shared','right'))]:
        for kind in kinds:
            assert capabilities['scores'][control+'/'+kind] == report['independent_views'][control][kind]['exact_accuracy']
    return report, {'checkpoint': checkpoint_path, 'checkpoint_step': expected_step,
                    'checkpoint_sha256': digest(root/checkpoint_path),
                    'evaluation': evaluation_path, 'evaluation_sha256': digest(root/evaluation_path),
                    'prediction_groups_recomputed': groups, 'generated_targets_verified': True,
                    'composition_subsets_recomputed': True, 'matched_control_budgets_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/integrated'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports, records = [], []
    for checkpoint, evaluation, step in [('baseline_step100.pt', 'evaluation_baseline100.json', 100),
                                         ('joint/best.pt', 'joint/evaluation.json', 200),
                                         ('balanced/last.pt', 'balanced/evaluation.json', 100)]:
        report, record = audit_one(args.root, checkpoint, evaluation, step)
        reports.append(report); records.append(record)
    # Every endpoint sees the exact same generated cases and intervention costs.
    for report in reports[1:]:
        for cohort in ('heldout_views', 'independent_views'):
            for control, rows in report[cohort].items():
                for kind, row in rows.items():
                    previous = reports[0][cohort][control][kind]
                    for key in ('episode_ids','targets','external_events','source_occurrences','expert_executions'):
                        assert row[key] == previous[key]
        for control, row in report['composition_controls'].items():
            for key in ('keys','targets','expert_executions','source_occurrences'):
                assert row[key] == reports[0]['composition_controls'][control][key]
        for gap, tasks in report['memory_controls'].items():
            for kind, evaluation in tasks.items():
                for control, row in evaluation['controls'].items():
                    previous = reports[0]['memory_controls'][gap][kind]['controls'][control]
                    for key in ('episode_ids', 'episodes', 'expert_executions'):
                        assert row[key] == previous[key]
    load = lambda path: torch.load(args.root/path, map_location='cpu', weights_only=True)
    # The balanced audit exported best.pt; the evaluated last.pt has identical weights.
    assert_equal(load('balanced/best.pt')['weights'], load('balanced/last.pt')['weights'])
    selection = json.loads((args.root/'selection.json').read_text())
    assert digest(args.root/'selected.pt') == digest(args.root/selection['selected']['checkpoint'])
    result = {'all_passed': True, 'endpoints': records,
              'cross_model_cases_and_costs_identical': True,
              'balanced_export_matches_evaluated_weights': True,
              'selected_checkpoint_matches_validation_manifest': True,
              'prediction_groups_recomputed': sum(row['prediction_groups_recomputed'] for row in records)}
    (args.root/'evaluation_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
