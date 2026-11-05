"""Rebuild objectives, selections and prediction tables for all 432 cohort fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import statistics
import time

import torch

from .audit_precision import same_audit_report
from .cohort_diversity import (COUNTS, PENALTIES, SETTINGS, bundles_for_count, cohort_coverage,
    nested_prefix, restore_head, source_hashes, validate_splits, verify_resume)
from .emission_adapter import ACTIVATIONS, ADAPTER_SEEDS, load_adapters, same_protocol, score_adapters, sha
from .emission_compare import summarize
from .preservation_compare import audit_objective
from .readout_compare import require
from .readout_diagnostic import load_diagnostic
from .residual_adapter import (base_fingerprint, balanced_weights, initialize_residual,
    load_residuals, score_residuals, select_strength, write_json)
from .residual_compare import audit_cohort, audit_rows

ROOT = Path(__file__).resolve().parents[2]


def audit_legacy(run, legacy, validation, inherited):
    """Check 30 legacy residual and 30 emission prediction groups per graph."""
    reference = load_diagnostic(ROOT/run['reference'])
    adapters = load_adapters(ROOT/run['prior_run']/'adapters.pt')
    reproduced = 0
    for directory, fitted, scorer in [(ROOT/run['residual_run'], inherited, score_residuals),
            (ROOT/run['prior_run'], adapters, lambda data, heads: score_adapters(data, heads, reference))]:
        saved = [json.loads(s) for s in (directory/'predictions.jsonl').read_text().splitlines()]
        report = json.loads((directory/'results.json').read_text())
        for split in ('test', 'independent'):
            scores, rows = scorer(legacy[split], fitted)
            require(rows == [r for r in saved if r['split'] == split], 'Legacy predictions differ.')
            require(same_audit_report(scores, report['evaluations'][split]), 'Legacy losses differ.')
            reproduced += len(rows)
    return reproduced


def audit_run(directory, run, rebuilt):
    directory = Path(directory)
    report = json.loads((directory/'results.json').read_text())
    require(report['complete'], 'Incomplete cohort experiment.')
    invocation = json.loads((directory/'invocation.json').read_text())
    require(same_protocol(invocation, report['invocation']), 'Invocation differs.')
    require(same_protocol(invocation['run'], run), 'Frozen graph inventory differs.')
    require(invocation['source_sha256'] == source_hashes(), 'Measured runtime source changed.')
    require(same_protocol(invocation['settings'], SETTINGS), 'Fitting settings differ.')
    require(all(sha(directory/n) == h for n, h in report['output_sha256'].items()), 'Output artifact changed.')
    prior = ROOT/run['residual_run']; emission = ROOT/run['prior_run']
    inputs = {'checkpoint': ROOT/run['checkpoint'], 'reference': ROOT/run['reference'],
        'residuals': prior/'residuals.pt', 'validation': prior/'validation.pt', 'features': emission/'features.pt'}
    require(invocation['input_sha256'] == {k: sha(p) for k, p in inputs.items()}, 'Frozen input artifacts changed.')
    require(sha(inputs['checkpoint']) == run['checkpoint_sha256'], 'Checkpoint lineage differs.')
    inherited = load_residuals(inputs['residuals']); base = inherited['base']
    reference = load_diagnostic(inputs['reference'])
    require(base_fingerprint(base) == base_fingerprint(reference['answer_heads']['raw']) == report['base_head_sha256'],
        'Frozen base differs from the legacy readout.')
    legacy = torch.load(inputs['features'], map_location='cpu', weights_only=True)['datasets']
    validation = torch.load(inputs['validation'], map_location='cpu', weights_only=True)
    mass = balanced_weights(legacy['train']); x = legacy['train']['features']['final/raw'].double()
    mean = (mass[:, None]*x).sum(0); scale = ((x-mean).square()*mass[:, None]).sum(0).sqrt().clamp_min(1e-6)
    require(torch.allclose(mean, base.mean, atol=1e-12, rtol=1e-12) and
        torch.allclose(scale, base.scale, atol=1e-12, rtol=1e-12), 'Legacy normalization provenance differs.')
    capture = torch.load(directory/'training.pt', map_location='cpu', weights_only=True)
    require(same_protocol(capture['invocation'], invocation), 'Capture provenance differs.')
    large = capture['training']; training = {c: nested_prefix(large, c) for c in COUNTS}
    fitted = torch.load(directory/'fitted.pt', map_location='cpu', weights_only=True)
    verify_resume(fitted, invocation, sha(directory/'training.pt'), base)
    require(len(fitted['heads']) == 36 and report['fits'] == {k: v['correction']['fit'] for k, v in fitted['heads'].items()},
        'Fit inventory or recorded optimizer diagnostics differ.')
    require(report['additional_expert_executions'] == large['protocol']['expert_executions'] == 3840*25*84,
        'Additional expert budget differs.')
    for c, data in training.items():
        audit_cohort(data, rebuilt)
        require(report['coverage'][str(c)] == cohort_coverage(data['protocol']), 'Coverage differs.')
        validate_splits(data, validation, {s: legacy[s] for s in ('test', 'independent')})
    for data in [validation, legacy['test'], legacy['independent']]:
        audit_cohort(data, rebuilt)
    for small, big in zip(training[64]['groups'], large['groups']):
        require(torch.equal(training[64]['features']['final/raw'][small['start']:small['stop']],
            large['features']['final/raw'][big['start']:big['start']+64]), 'Cohorts are not nested exactly.')
    drift = float((training[64]['features']['final/raw']-legacy['train']['features']['final/raw']).abs().max())
    require(drift == report['legacy_training_max_abs_drift'], 'Recapture drift differs.')
    for key, record in fitted['heads'].items():
        count, penalty, activation, seed = key.split('/'); head = restore_head(record, base); fit = head.correction.fit
        require(fit['parameters'] == 5392 and fit['base_frozen_verified'] and
            fit['preservation']['margin_cap'] == 1., 'Correction dimensions or frozen base differ.')
        initial = initialize_residual(base, hidden=16, seed=int(seed), activation=activation)
        require(initial.correction.fit['initial_tensor_sha256'] == fit['initial_tensor_sha256'], 'Initialization differs.')
        audit_objective(training[int(count)], head)
        selected = select_strength(head, validation, fit_identities=training[int(count)]['identities'])
        require(head.strength == selected.strength and same_audit_report(head.selection, selected.selection),
            'Validation selection differs.')
    rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    require(len(rows) == 240, 'Expected 240 prediction groups per graph.')
    metrics = {}; by_head = {}; selection = {}
    for count in COUNTS:
        bundles = bundles_for_count(count, training[count], validation, base, fitted['heads'])
        lock = {family: {n: h.selection for n, h in b['heads'].items()} for family, b in bundles.items()}
        require(same_protocol(lock, report['selection'][str(count)]), 'Selected penalty or stored decision differs.')
        selection[str(count)] = lock
        for family, bundle in bundles.items():
            selected_rows = [{k: v for k, v in r.items() if k not in ('count_per_cell', 'family')}
                for r in rows if r['count_per_cell'] == count and r['family'] == family]
            for split in ('test', 'independent'):
                scores, predictions = score_residuals(legacy[split], bundle)
                require(same_audit_report(scores, report['evaluations'][str(count)][family][split]) and
                    predictions == [r for r in selected_rows if r['split'] == split], 'Predictions or losses differ.')
            scores = audit_rows({'fits': bundle['heads'], 'evaluations': report['evaluations'][str(count)][family]}, selected_rows)
            summary = summarize(scores, selected_rows[0]['predictions'])
            by_head[f'{count}/{family}'] = summary
            for activation in ACTIVATIONS:
                for mode in ('full', 'selected'):
                    names = [f'{activation}/{seed}/{mode}' for seed in ADAPTER_SEEDS]
                    label = f'{count}/{family}/{activation}/{mode}'
                    metrics[label] = {m: statistics.mean(summary[n][m] for n in names) for m in summary[names[0]]}
            metrics[f'{count}/{family}/linear/raw'] = summary['linear/raw']
    legacy_groups = audit_legacy(run, legacy, validation, inherited)
    return {**run, 'metrics': metrics, 'by_head': by_head, 'selection': selection,
        'fits': report['fits'], 'coverage': report['coverage'], 'seconds': report['seconds'],
        'legacy_training_max_abs_drift': drift, 'legacy_prediction_groups_exact': legacy_groups,
        'additional_expert_executions': report['additional_expert_executions'],
        'files': {n: sha(directory/n) for n in ('training.pt', 'fitted.pt', 'results.json', 'predictions.jsonl', 'invocation.json')}}


_REBUILT = {}


def _audit_worker(args):
    directory, run, threads, wait = args
    torch.set_num_threads(threads)
    if wait:
        while not (directory/'results.json').exists():
            status = json.loads((directory.parent/'status.json').read_text())
            row = next(r for r in status['runs'] if r['name'] == run['name'])
            require(row['status'] != 'failed', 'Fitting failed for '+run['name'])
            require(row['status'] != 'complete', 'Completed run is missing results: '+run['name'])
            time.sleep(1)
    return audit_run(directory, run, _REBUILT)


def compare(source, *, on_run=None, jobs=1, threads=2, wait=False):
    source = Path(source); status = json.loads((source/'status.json').read_text())
    require(len(status['runs']) == 12, 'Expected twelve graph runs.')
    if not wait:
        require(status['complete'] and all(r['status'] == 'complete' and r['returncode'] == 0
            for r in status['runs']), 'All twelve runs must finish.')
    expected = {(p, s, n) for p in ('original', 'gated') for s in ('dropout', 'balanced') for n in (20261106, 20261107, 20261108)}
    require({(r['publication_policy'], r['source_schedule'], r['training_seed']) for r in status['runs']} == expected,
        'Incomplete matched graph inventory.')
    manifest = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())
    original = {r['name']: r for r in manifest['runs']}; runs = []
    require(jobs > 0 and threads > 0, 'Audit jobs and threads must be positive.')
    tasks = [(source/row['name'], original[row['name']], threads, wait) for row in status['runs']]
    if jobs == 1:
        for task in tasks:
            runs.append(_audit_worker(task))
            if on_run: on_run({'verified': len(runs), 'models': 12, 'name': task[1]['name']})
    else:
        # Spawn avoids inheriting an initialized PyTorch thread pool through fork.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(_audit_worker, tasks):
                runs.append(result)
                if on_run: on_run({'verified': len(runs), 'models': 12, 'name': result['name']})
    status = json.loads((source/'status.json').read_text())
    require(status['complete'] and all(r['status'] == 'complete' and r['returncode'] == 0
        for r in status['runs']), 'All twelve fits must finish before the comparison is exported.')
    groups = {}
    for policy in ('original', 'gated'):
        for schedule in ('dropout', 'balanced'):
            members = [r for r in runs if (r['publication_policy'], r['source_schedule']) == (policy, schedule)]
            groups[policy+'/'+schedule] = {family: {metric: {
                'mean': statistics.mean(r['metrics'][family][metric] for r in members),
                'min_graph': min(r['metrics'][family][metric] for r in members),
                'max_graph': max(r['metrics'][family][metric] for r in members)}
                for metric in members[0]['metrics'][family]} for family in members[0]['metrics']}
    nonzero = {str(c): {a: sum(r['selection'][str(c)]['selected'][f'{a}/{s}']['selected_strength'] > 0
                              for r in runs for s in ADAPTER_SEEDS) for a in ACTIVATIONS} for c in COUNTS}
    return {'schema_version': 1, 'verified_models': 12, 'fitted_heads': 432, 'settings': SETTINGS,
        'groups': groups, 'runs': runs, 'selected_nonzero': nonzero,
        'legacy_prediction_groups_exact': sum(r['legacy_prediction_groups_exact'] for r in runs),
        'additional_expert_executions': sum(r['additional_expert_executions'] for r in runs),
        'integrity': {'nested_training_features_exact': True, 'source_and_artifact_hashes_verified': True,
            'frozen_linear_base_and_normalization_identical': True, 'training_objectives_reconstructed': True,
            'training_validation_evaluation_ids_disjoint': True, 'validation_choices_reproduced': True,
            'predictions_and_losses_reproduced': True, 'coverage_and_budgets_reconstructed': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1); parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--wait', action='store_true', help='Audit finished graphs while the remaining fit jobs run.')
    args = parser.parse_args(); torch.set_num_threads(2)
    result = compare(args.source, jobs=args.jobs, threads=args.threads, wait=args.wait,
                     on_run=lambda row: print(json.dumps(row), flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True); write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ('verified_models', 'fitted_heads', 'selected_nonzero',
        'legacy_prediction_groups_exact', 'additional_expert_executions')}), flush=True)


if __name__ == '__main__':
    main()
