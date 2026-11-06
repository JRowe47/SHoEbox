"""Run the preserved experiment suites in isolated processes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ('semantic_1m', ['test_model']),
    ('semantic_10m', ['test_model', 'test_scaled']),
    ('two_child', ['test_hierarchy']),
    ('hat_minkron', ['verify']),
]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())['files']
    changed = [name for name, sha in manifest.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha]
    if changed:
        print('Source differs from the recorded experiment: ' + ', '.join(changed), flush=True)
    failed = []
    for name, modules in SUITES:
        print(f'\nRunning {name}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v', *modules],
                                cwd=ROOT / 'experiments' / name)
        if result.returncode:
            failed.append(name)
    print('\nRunning text_regions', flush=True)
    modules = ['verify', 'test_activity', 'test_objectives', 'test_learning', 'test_probe', 'test_normalization',
               'test_view_data', 'test_view_eval', 'test_view_train', 'test_evidence_ports',
               'test_view_score', 'test_transfer_eval', 'test_evidence_compare', 'test_query_interference',
               'test_baseline_compare', 'test_observability_compare', 'test_source_schedule_data',
               'test_source_schedule_training', 'test_source_schedule_compare', 'test_readout_diagnostic',
               'test_emission_adapter', 'test_residual_adapter', 'test_shared_preservation',
               'test_audit_precision', 'test_cohort_diversity', 'test_full_rank_readout', 'test_composition',
               'test_composition_curriculum', 'test_composition_continue', 'test_integrated',
               'test_fact_supervision', 'test_gradient_horizon', 'test_publication', 'test_retained_composition',
               'test_joint_retained', 'test_multifact', 'test_joint_text', 'test_gradient_balance', 'test_runtime',
               'test_runtime_fusion']
    result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                             *['experiments.text_regions.' + name for name in modules]], cwd=ROOT)
    if result.returncode:
        failed.append('text_regions')
    if failed:
        print('Failed suites: ' + ', '.join(failed), file=sys.stderr)
        return 1
    print('\nAll five experiment suites passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
