"""Audit saved shared-preservation fits, validation choices, and every prediction."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .audit_precision import same_audit_report
from .emission_adapter import ACTIVATIONS, ADAPTER_SEEDS, same_protocol, sha
from .emission_compare import summarize
from .readout_compare import require
from .residual_adapter import (balanced_weights, base_fingerprint, initialize_residual,
    load_residuals, score_residuals, select_strength, write_json)
from .residual_compare import audit_run as audit_prior, audit_rows
from .shared_preservation import PENALTIES, MARGIN_CAP, JOINT_RULE

ROOT = Path(__file__).resolve().parents[2]


def audit_objective(training, head):
    """Recompute the augmented objective directly from saved tensors."""
    x = training['features']['final/raw'].double(); y = training['answers']
    mass = balanced_weights(training); mass = mass / mass.sum()
    base = head.base.logits(x); logits = base + head.correction.logits(x)
    index = torch.arange(len(y))
    other = torch.ones_like(base, dtype=torch.bool); other[index, y] = False
    base_margin = base[index, y] - base.masked_fill(~other, -torch.inf).max(-1).values
    final_margin = logits[index, y] - logits.masked_fill(~other, -torch.inf).max(-1).values
    fit = head.correction.fit; cfg = fit['preservation']
    anchor = torch.tensor([q == 'shared' for q in training['queries']]) & (base.argmax(-1) == y) & (base_margin > 0)
    anchor_mass = mass * anchor
    if anchor.any():
        anchor_mass /= anchor_mass.sum()
    target = base_margin.clamp(0, cfg['margin_cap'])
    loss = float(((target - final_margin).clamp_min(0).square() * anchor_mass).sum())
    nll = float((F.cross_entropy(logits, y, reduction='none') * mass).sum())
    regularizer = .5 * fit['l2'] * sum(float(getattr(head.correction, k).square().sum()) for k in ('w1', 'b1', 'w2', 'b2'))
    objective = nll + regularizer + cfg['penalty'] * loss
    require(all(math.isclose(actual, saved, abs_tol=1e-12, rel_tol=1e-12) for actual, saved in (
        (loss, cfg['unweighted_penalty']), (nll, fit['weighted_training_nll']), (objective, fit['objective']))),
        'Training preservation objective differs.')
    require(cfg['anchor_count'] == int(anchor.sum()) and cfg['shared_count'] == training['queries'].count('shared')
        and cfg['margin_violations'] == int(((final_margin < target - 1e-9) & anchor).sum())
        and cfg['baseline_correct_answers_lost'] == int(((logits.argmax(-1) != y) & anchor).sum()),
        'Training anchor inventory or outcome differs.')
    require(fit['weighted_training_accuracy'] == float(((logits.argmax(-1) == y).double() * mass).sum()),
        'Training accuracy differs.')


def audit_run(directory, run, rebuilt):
    directory = Path(directory)
    old = audit_prior(ROOT / run['residual_run'], run, rebuilt)
    report = json.loads((directory / 'results.json').read_text())
    require(report['complete'] and report['additional_expert_executions'] == 0, 'Incomplete cached-feature experiment.')
    require(all(sha(ROOT / p) == h for p, h in report['source_sha256'].items()), 'Measured source changed.')
    expected_files = {'selected.pt', 'penalty_1.pt', 'penalty_10.pt', 'selection_lock.json', 'predictions.jsonl'}
    require(set(report['output_sha256']) == expected_files, 'Incomplete output inventory.')
    require(all(sha(directory / p) == h for p, h in report['output_sha256'].items()), 'Output artifact changed.')
    prior = ROOT / run['residual_run']; emission = ROOT / run['prior_run']
    inputs = {'prior_results': prior / 'results.json', 'prior_residuals': prior / 'residuals.pt',
        'validation': prior / 'validation.pt', 'features': emission / 'features.pt'}
    require(report['input_sha256'] == {k: sha(p) for k, p in inputs.items()}, 'Input artifact changed.')
    inherited = load_residuals(inputs['prior_residuals'])
    bundles = {f'penalty_{p:g}': load_residuals(directory / f'penalty_{p:g}.pt') for p in PENALTIES}
    selected = load_residuals(directory / 'selected.pt')
    datasets = torch.load(inputs['features'], map_location='cpu', weights_only=True)['datasets']
    validation = torch.load(inputs['validation'], map_location='cpu', weights_only=True)
    training = datasets['train']
    require(same_protocol(training['protocol'], report['fit_protocol']) and
        same_protocol(validation['protocol'], report['validation_protocol']), 'Reported cohorts differ.')
    lock = json.loads((directory / 'selection_lock.json').read_text())
    require(lock == report['selection'] and lock['rule'] == JOINT_RULE and lock['penalties'] == list(PENALTIES)
        and lock['selection_inputs'] == ['training', 'validation']
        and lock['fit_inputs_sha256'] == training['protocol']['inputs_sha256']
        and lock['validation_inputs_sha256'] == validation['protocol']['inputs_sha256'], 'Invalid selection lock.')
    for family, bundle in {**bundles, 'selected': selected}.items():
        require(base_fingerprint(bundle['base']) == report['base_head_sha256'] == base_fingerprint(inherited['base']),
            'Frozen baseline differs.')
        for key in ('fit_protocol', 'validation_protocol', 'fit_identities', 'validation_identities', 'initialization_seeds'):
            require(same_protocol(bundle[key], inherited[key]), 'Fitting or validation lineage differs.')
        require(report['fits'][family] == {k: h.correction.fit for k, h in bundle['heads'].items()}, 'Fit metadata differs.')
        for name, head in bundle['heads'].items():
            fit = head.correction.fit
            require(fit['hidden'] == 16 and fit['max_iter'] == 400 and fit['l2'] == .001
                and fit['parameters'] == 5392 and fit['examples'] == 960 and fit['base_frozen_verified'], 'Fitting budget differs.')
            cfg = fit['preservation']
            require(cfg['margin_cap'] == MARGIN_CAP and cfg['penalty'] in PENALTIES, 'Preservation setting differs.')
            if family != 'selected':
                require(family == f"penalty_{cfg['penalty']:g}", 'Candidate label differs.')
            initial = initialize_residual(bundle['base'], hidden=16, seed=fit['seed'], activation=fit['activation'])
            require(initial.correction.fit['initial_tensor_sha256'] == fit['initial_tensor_sha256']
                and fit['initial_correction_exactly_zero'], 'Matched initialization differs.')
            audit_objective(training, head)
            if family != 'selected':
                decision = select_strength(head, validation, fit_identities=bundle['fit_identities'])
                require(decision.strength == head.strength and same_audit_report(decision.selection, head.selection),
                    'Validation strength decision differs.')
    for name, head in selected['heads'].items():
        choices = {p: bundles[f'penalty_{p:g}']['heads'][name] for p in PENALTIES}
        best = min(h.selection['selected_weighted_nll'] for h in choices.values())
        p, h = min(((p, h) for p, h in choices.items() if h.selection['selected_weighted_nll'] <= best + 1e-12),
            key=lambda item: (item[1].strength, item[0]))
        expected = {**h.selection, 'preservation_choice': {'rule': JOINT_RULE, 'selected_penalty': p,
            'candidates': {str(p): {'strength': h.strength, 'selection': h.selection} for p, h in choices.items()}}}
        require(head.strength == h.strength and same_protocol(head.selection, expected)
            and same_protocol(head.correction.fit, h.correction.fit)
            and all(torch.equal(getattr(head.correction, k), getattr(h.correction, k)) for k in ('mean', 'scale', 'w1', 'b1', 'w2', 'b2')),
            'Joint validation decision or selected tensors differ.')
        require(same_protocol(lock['heads'][name], expected), 'Selected head differs from lock.')
    rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
    require(len(rows) == 120 and set(r['family'] for r in rows) == {'prior', 'selected', 'penalty_1', 'penalty_10'},
        'Incomplete prediction rows.')
    metrics = {}; by_head = {}
    for family, bundle in {'prior': inherited, **bundles, 'selected': selected}.items():
        family_rows = [{k: v for k, v in r.items() if k != 'family'} for r in rows if r['family'] == family]
        for split in ('test', 'independent'):
            scores, predictions = score_residuals(datasets[split], bundle)
            require(same_audit_report(scores, report['evaluations'][family][split])
                and predictions == [r for r in family_rows if r['split'] == split], 'Saved predictions or losses differ.')
        summary = summarize(audit_rows({'fits': bundle['heads'], 'evaluations': report['evaluations'][family]}, family_rows),
            family_rows[0]['predictions'])
        by_head[family] = summary
        families = {'linear/raw': ['linear/raw']}
        families.update({a + '/' + mode: [f'{a}/{s}/{mode}' for s in ADAPTER_SEEDS]
            for a in ACTIVATIONS for mode in ('full', 'selected')})
        for label, names in families.items():
            metrics[family + '/' + label] = {m: statistics.mean(summary[n][m] for n in names) for m in summary['linear/raw']}
    for family in ('selected', 'penalty_1', 'penalty_10'):
        require(metrics[family + '/linear/raw'] == old['metrics']['linear/raw'], 'Prior raw baseline changed.')
    for a in ACTIVATIONS:
        for mode in ('selected', 'full'):
            require(metrics['prior/' + a + '/' + mode] == old['metrics'][a + '/' + mode], 'Prior residual scores changed.')
    return {k: run[k] for k in ('name', 'publication_policy', 'source_schedule', 'training_seed')} | {
        'metrics': metrics, 'by_head': by_head, 'fits': report['fits'], 'selection': lock,
        'seconds': report['seconds'], 'files': [{'name': p.name, 'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(directory.iterdir()) if p.is_file()]}


def compare(source):
    source = Path(source); status = json.loads((source / 'status.json').read_text())
    require(status['complete'] and all(r['status'] == 'complete' and r['returncode'] == 0 for r in status['runs']),
        'All runs must complete.')
    expected = {(p, s, n) for p in ('original', 'gated') for s in ('dropout', 'balanced') for n in (20261106, 20261107, 20261108)}
    require(len(status['runs']) == 12 and {(r['publication_policy'], r['source_schedule'], r['training_seed'])
        for r in status['runs']} == expected, 'All twelve frozen graphs required.')
    require(status['settings']['penalties'] == list(PENALTIES) and status['settings']['margin_cap'] == MARGIN_CAP,
        'Study settings differ.')
    rebuilt = {}; runs = [audit_run(source / r['name'], r, rebuilt) for r in status['runs']]
    groups = {}
    for policy in ('original', 'gated'):
        for schedule in ('dropout', 'balanced'):
            cohort = [r for r in runs if (r['publication_policy'], r['source_schedule']) == (policy, schedule)]
            groups[policy + '/' + schedule] = {family: {metric: {
                'mean': statistics.mean(r['metrics'][family][metric] for r in cohort),
                'min_graph': min(r['metrics'][family][metric] for r in cohort),
                'max_graph': max(r['metrics'][family][metric] for r in cohort)}
                for metric in cohort[0]['metrics'][family]} for family in cohort[0]['metrics']}
    choices = {a: dict(sorted(Counter(f"{r['selection']['heads'][f'{a}/{s}']['preservation_choice']['selected_penalty']:g}/"
        f"{r['selection']['heads'][f'{a}/{s}']['selected_strength']:g}" for r in runs for s in ADAPTER_SEEDS).items()))
        for a in ACTIVATIONS}
    return {'schema_version': 1, 'verified_models': 12, 'new_fitted_heads': 144,
        'prior_control_heads': 72, 'settings': status['settings'], 'groups': groups, 'runs': runs,
        'selected_penalty_and_strength': choices, 'additional_expert_executions': 0,
        'integrity': {'prior_study_fully_reaudited': True, 'training_objectives_reconstructed': True,
            'validation_selection_reproduced': True, 'predictions_and_losses_reproduced': True,
            'selected_tensors_verified': True, 'baseline_predictions_unchanged': True,
            'synthetic_cohorts_reconstructed': True, 'split_identities_disjoint': True}}


def audit_convergence(source, directory):
    """Reconstruct training-only case selection and the four longer-fit results."""
    source, directory = Path(source), Path(directory)
    status = json.loads((source / 'status.json').read_text())
    reports = {r['name']: json.loads((source / r['name'] / 'results.json').read_text()) for r in status['runs']}
    plan = json.loads((directory / 'plan.json').read_text())
    require(plan['main_study_unchanged'] and len(plan['cases']) == 4, 'Invalid convergence plan.')
    cases = []
    for family in ('penalty_1', 'penalty_10'):
        for activation in ACTIVATIONS:
            options = [(r, n, f) for r in status['runs'] for n, f in reports[r['name']]['fits'][family].items()
                if n.startswith(activation + '/')]
            run, name, prior_fit = max(options, key=lambda item: item[2]['gradient_max'])
            label = family + '_' + activation; folder = directory / label
            report = json.loads((folder / 'results.json').read_text()); case = report['case']
            require(case in plan['cases'] and case['run'] == run and case['head'] == name
                and case['family'] == family and case['max_iter'] == 1600
                and case['training_gradient_max'] == prior_fit['gradient_max'], 'Convergence case selection differs.')
            require(all(sha(ROOT / p) == h for p, h in report['source_sha256'].items()), 'Convergence source changed.')
            require(all(sha(ROOT / p) == h for p, h in report['input_sha256'].items()), 'Convergence input changed.')
            require(set(report['output_sha256']) == {'extended.pt', 'selection_lock.json', 'predictions.jsonl'}
                and all(sha(folder / p) == h for p, h in report['output_sha256'].items()), 'Convergence output changed.')
            bundle = load_residuals(folder / 'extended.pt'); head = bundle['heads'][name]
            original = load_residuals(source / run['name'] / (family + '.pt'))
            require(base_fingerprint(bundle['base']) == base_fingerprint(original['base']), 'Convergence base changed.')
            require(report['prior_fit'] == prior_fit and report['extended_fit'] == head.correction.fit
                and report['prior_strength'] == original['heads'][name].strength
                and report['extended_strength'] == head.strength and head.correction.fit['max_iter'] == 1600,
                'Convergence metadata differs.')
            for key in ('fit_protocol', 'validation_protocol', 'fit_identities', 'validation_identities'):
                require(same_protocol(bundle[key], original[key]), 'Convergence cohort differs.')
            datasets = torch.load(ROOT / run['prior_run'] / 'features.pt', map_location='cpu', weights_only=True)['datasets']
            validation = torch.load(ROOT / run['residual_run'] / 'validation.pt', map_location='cpu', weights_only=True)
            audit_objective(datasets['train'], head)
            decision = select_strength(head, validation, fit_identities=bundle['fit_identities'])
            require(head.strength == decision.strength and same_audit_report(head.selection, decision.selection)
                and same_protocol(json.loads((folder / 'selection_lock.json').read_text()),
                    {'head': name, 'selection': head.selection}), 'Convergence validation selection differs.')
            rows = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text().splitlines()]
            require(len(rows) == 30, 'Incomplete convergence prediction export.')
            for split in ('test', 'independent'):
                scores, predictions = score_residuals(datasets[split], bundle)
                expected = {'protocol': scores['protocol'], 'groups': [{**g, 'decoders': {
                    n: m for n, m in g['decoders'].items() if n.startswith(name + '/')}} for g in scores['groups']]}
                expected_rows = [{**r, 'predictions': {n: p for n, p in r['predictions'].items()
                    if n.startswith(name + '/')}} for r in predictions]
                require(same_audit_report(expected, report['evaluations'][split])
                    and expected_rows == [r for r in rows if r['split'] == split], 'Convergence predictions differ.')
            cases.append({'family': family, 'activation': activation, 'run': run['name'], 'head': name,
                'prior_fit': prior_fit, 'extended_fit': head.correction.fit,
                'prior_strength': report['prior_strength'], 'extended_strength': head.strength,
                'validation_selection': head.selection})
    return {'verified_fits': 4, 'selection': plan['selection'], 'main_study_unchanged': True,
        'cases': cases, 'objectives_selection_and_predictions_reproduced': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--convergence', type=Path)
    args = parser.parse_args(); torch.set_num_threads(2)
    result = compare(args.source)
    if args.convergence:
        result['convergence_diagnostic'] = audit_convergence(args.source, args.convergence)
    args.out.parent.mkdir(parents=True, exist_ok=True); write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ('verified_models', 'new_fitted_heads', 'selected_penalty_and_strength')}))


if __name__ == '__main__':
    main()
