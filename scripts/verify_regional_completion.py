"""Verify completion, provenance and arithmetic of the regional-fact artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.composition_train import source_hashes


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    selection = read(root / 'selection.json')
    audit = read(root / 'evaluation_audit.json')
    assert selection['all_passed'] and audit['all_passed']
    assert selection['selected'] == audit['selected']
    assert digest(root / 'selected.pt') == selection['selected']['sha256']
    stages = ['composition_progress', 'composition']
    stages += [f'memory/{gap}/{kind}' for gap in (16, 32) for kind in ('recall', 'correction')]
    stages += ['heldout_views']
    stages += ['independent/' + name for name in ('all', 'a_only', 'b_only', 'c_only', 'no_private', 'reset_query')]
    stages += ['capabilities']
    evaluated = []
    diagnostic_groups = 0
    for arm in ('control', 'regional'):
        run = root / arm
        summary = read(run / 'summary.json')
        history = read(run / 'history.json')
        assert summary['completed_step'] == 100
        assert [r['step'] for r in history] == list(range(101))
        report = read(run / 'evaluation.json')
        manifest = report['_manifest']
        assert report['checkpoint_step'] == 100
        assert report['_completed'] == stages
        assert set(report['_seconds']) == set(stages)
        assert manifest == {'checkpoint_sha256': digest(run / 'last.pt'),
                            'count': 512, 'batch': 256, 'source_hashes': source_hashes(),
                            'evaluator_sha256': digest(ROOT / 'scripts/evaluate_regional_facts.py')}
        assert report['test_capabilities']['composition'] == report['composition_progress']
        record = selection['runs'][arm]
        for name in ('endpoint_export', 'selected_export'):
            assert record[name]['exact_logits_and_states']
            assert set(record[name]['removed_heads']) == {
                'source_reconstruction', 'regional_readout', 'memory_retention', 'fact_retention'}
        diagnostic = read(run / 'local_diagnostic.json')
        assert diagnostic['checkpoint_sha256'] == manifest['checkpoint_sha256']
        assert diagnostic['source_sha256'] == digest(ROOT / 'scripts/probe_regional_facts.py')
        assert diagnostic['checkpoint_step'] == 100 and diagnostic['count_per_task'] == 64
        for task in diagnostic['records'].values():
            assert len(task['identities']) == len(set(task['identities'])) == 64
            for fields in task['nodes'].values():
                for row in fields.values():
                    assert row['count'] == len(row['targets']) == len(row['predictions'])
                    assert len(row['write_predictions']) == len(row['write_positions']) == row['count']
                    assert all(position >= 0 for position in row['write_positions'])
                    assert all(value in range(65, 69) for value in row['targets'])
                    for prefix in ('', 'write_'):
                        correct = sum(a == b for a, b in zip(row[prefix + 'predictions'], row['targets']))
                        assert row[prefix + 'correct'] == correct
                        assert row[prefix + 'accuracy'] == correct / row['count']
                        diagnostic_groups += 1
        evaluated.append({'run': arm, 'completed_stages': len(stages),
                          'checkpoint_sha256': manifest['checkpoint_sha256'],
                          'evaluation_sha256': digest(run / 'evaluation.json'),
                          'local_diagnostic_sha256': digest(run / 'local_diagnostic.json')})
    recovery = read(root / 'recovery.json')
    assert recovery['control_step100_matches_preinterruption_log_exactly_except_timing']
    for arm, row in recovery['runs'].items():
        assert row['resumed_from_step'] == 90 and row['remaining_updates'] == 10
        assert row['source_hashes_verified']
        assert digest(root / arm / 'recovery_step0090.pt') == row['checkpoint_sha256']
    log = (root / 'all_tests.log').read_text()
    suites = [int(n) for n in re.findall(r'^Ran (\d+) tests', log, re.M)]
    assert suites == [10, 16, 14, 20, 295]
    assert 'All five experiment suites passed.' in log and '\nFAILED' not in log
    return {'all_passed': True, 'runs': evaluated,
            'regression_tests': {'total': sum(suites), 'suites': len(suites), 'all_passed': True,
                                 'timing': 'Completed before recovery; training source hashes remain unchanged',
                                 'log_sha256': digest(root / 'all_tests.log')},
            'recovery_checkpoints_verified': True,
            'all_stage_manifests_verified': True,
            'training_and_evaluation_audits_passed': True,
            'diagnostic_prediction_groups_recomputed': diagnostic_groups,
            'export_heads_removed_and_exact_logits_states_checked': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/regional_facts'))
    args = p.parse_args()
    report = verify(args.root)
    (args.root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
