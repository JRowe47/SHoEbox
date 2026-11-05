"""Audit every full-rank fit and compare against retained identity/GELU heads."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .audit_precision import same_audit_report
from .cohort_diversity import bundles_for_count
from .compiled_readout import compile_readout, restore_compiled
from .diversity_compare import audit_run as audit_prior_run
from .emission_adapter import ACTIVATIONS, ADAPTER_SEEDS, same_protocol, sha
from .emission_compare import summarize
from .full_rank_readout import (COUNTS, PENALTIES, ROOT, SETTINGS, bundle_for_count,
    load_inputs, restore_head, source_hashes, verify_resume)
from .readout_compare import require
from .residual_adapter import (balanced_weights, save_tensors, score_residuals, select_strength, write_json)
from .residual_compare import audit_rows
from .shared_preservation import answer_margins, preservation_loss, shared_anchors


def audit_objective(training, head):
    """Recompute the full-weight objective and gradient from saved tensors."""
    fit = head.refit.fit; x = training['features']['final/raw'].double(); y = training['answers']
    mass = balanced_weights(training)
    z = torch.cat(((x-head.base.mean)/head.base.scale, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    mask, target, anchors = shared_anchors(head.base.logits(x), y, training['queries'], mass,
        fit['preservation']['margin_cap'])
    weight = head.refit.weight.clone().requires_grad_(True)
    logits = z @ weight
    nll = (F.cross_entropy(logits, y, reduction='none')*mass).sum()
    penalty = preservation_loss(logits, y, target, anchors)
    objective = nll+.5*fit['l2']*weight.square().sum()+fit['preservation']['penalty']*penalty
    objective.backward()
    for actual, expected in ((float(objective.detach()), fit['objective']),
            (float(nll.detach()), fit['weighted_training_nll']),
            (float(penalty.detach()), fit['preservation']['unweighted_penalty']),
            (float(weight.grad.abs().max()), fit['gradient_max'])):
        require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), 'Objective or gradient differs.')
    require(int(mask.sum()) == fit['preservation']['anchor_count'], 'Anchor count differs.')
    require(int(((logits.argmax(-1) != y) & mask).sum()) == fit['preservation']['baseline_correct_answers_lost'],
        'Shared-answer losses differ.')
    require(int(((answer_margins(logits, y) < target-1e-9) & mask).sum()) == fit['preservation']['margin_violations'],
        'Training margin violations differ.')
    require(float(((logits.argmax(-1) == y).double()*mass).sum()) == fit['weighted_training_accuracy'],
        'Training accuracy differs.')


def deployment_heads(base, training, validation, records, previous):
    result = {'base': base}
    for count in COUNTS:
        full = bundle_for_count(count, base, training[count], validation, records)
        prior = bundles_for_count(count, training[count], validation, base, previous['heads'])['selected']
        result[f'{count}/linear/preserved'] = full['heads']['preserved']
        result[f'{count}/linear/penalty_0'] = full['heads']['penalty_0']
        for name, h in prior['heads'].items():
            result[f'{count}/{name}'] = h
    return result


def audit_run(source, prior_source, row, rebuilt, *, export_compiled=False):
    directory = source/row['name']; report = json.loads((directory/'results.json').read_text())
    require(report['complete'], 'Incomplete full-rank run.')
    require(all(sha(directory/n) == h for n, h in report['output_sha256'].items()), 'Changed full-rank artifacts.')
    invocation = json.loads((directory/'invocation.json').read_text())
    require(same_protocol(invocation, report['invocation']) and invocation['run'] == row, 'Invocation differs.')
    require(invocation['source_sha256'] == source_hashes() and invocation['settings'] == SETTINGS,
        'Measured source or settings changed.')
    # The inherited corrections are rebuilt from tensors, not copied aggregate scores.
    inherited = audit_prior_run(prior_source/row['name'], row, rebuilt)
    base, training, validation, evaluations, previous, hashes = load_inputs(prior_source, row)
    require(hashes == invocation['input_sha256'], 'Retained inputs changed.')
    fitted = torch.load(directory/'fitted.pt', map_location='cpu', weights_only=True)
    verify_resume(fitted, invocation, base)
    require(set(fitted['heads']) == {f'{c}/{p:g}' for c in COUNTS for p in PENALTIES}, 'Incomplete fit inventory.')
    require(report['fits'] == {k: r['refit']['fit'] for k, r in fitted['heads'].items()}, 'Fit metadata differs.')
    require(report['additional_expert_executions'] == 0, 'Unexpected new expert executions.')
    for key, record in fitted['heads'].items():
        count = int(key.split('/')[0]); head = restore_head(record, base)
        audit_objective(training[count], head)
        selected = select_strength(head, validation, fit_identities=training[count]['identities'])
        require(head.strength == selected.strength and same_audit_report(head.selection, selected.selection),
            'Validation scale selection differs.')
    saved_rows = [json.loads(s) for s in (directory/'predictions.jsonl').read_text().splitlines()]
    require(len(saved_rows) == 60, 'Incomplete prediction groups.')
    metrics = {}; by_head = {}; selections = {}
    for count in COUNTS:
        bundle = bundle_for_count(count, base, training[count], validation, fitted['heads'])
        selections[str(count)] = {n: h.selection for n, h in bundle['heads'].items()}
        rows = [{k: v for k, v in r.items() if k != 'count_per_cell'}
            for r in saved_rows if r['count_per_cell'] == count]
        for split, data in evaluations.items():
            scores, predictions = score_residuals(data, bundle)
            require(predictions == [r for r in rows if r['split'] == split], 'Predictions differ.')
            require(same_audit_report(scores, report['evaluations'][str(count)][split]), 'Evaluation losses differ.')
        scores = audit_rows({'fits': bundle['heads'], 'evaluations': report['evaluations'][str(count)]}, rows)
        means = summarize(scores, rows[0]['predictions']); by_head[str(count)] = means
        metrics.update({f'{count}/linear/{n}': m for n, m in means.items() if n != 'linear/raw'})
        metrics['base'] = means['linear/raw']
        for activation in ACTIVATIONS:
            for family in ('selected', 'penalty_0'):
                for mode in ('full', 'selected'):
                    metrics[f'{count}/{activation}/{family}/{mode}'] = inherited['metrics'][f'{count}/{family}/{activation}/{mode}']
    require(same_protocol(selections, report['selection']), 'Penalty selection differs.')
    lock = json.loads((directory/'selection_lock.json').read_text())
    require(lock == {'selection_inputs': ['train', 'validation'], 'selection': selections}, 'Selection lock differs.')
    heads = deployment_heads(base, training, validation, fitted['heads'], previous)
    compiled = {}; equivalence = {}; costs = {}
    with torch.inference_mode():
        for name, h in heads.items():
            c = compile_readout(h); compiled[name] = c.state_dict(); costs[name] = c.cost()
            maximum = 0.; cases = 0
            for data in [*training.values(), validation, *evaluations.values()]:
                x = data['features']['final/raw']; expected = h.logits(x); actual = c(x)
                maximum = max(maximum, float((expected-actual).abs().max())); cases += len(x)
                require(torch.allclose(expected, actual, atol=1e-9, rtol=1e-11), 'Compiled logits differ.')
                require(torch.equal(expected.argmax(-1), actual.argmax(-1)), 'Compiled predictions differ.')
            equivalence[name] = {'prediction_cases_exact': cases, 'max_abs_logit_difference': maximum}
    compiled_path = directory/'compiled.pt'
    if export_compiled:
        save_tensors(compiled_path, {'schema_version': 1, 'heads': compiled,
            'input_sha256': hashes, 'source_sha256': invocation['source_sha256']})
    require(compiled_path.exists(), 'Export compiled heads first with --export-compiled.')
    exported = torch.load(compiled_path, map_location='cpu', weights_only=True)
    require(exported['schema_version'] == 1 and exported['input_sha256'] == hashes
        and exported['source_sha256'] == invocation['source_sha256'] and set(exported['heads']) == set(heads),
        'Compiled export provenance differs.')
    for name, record in exported['heads'].items():
        restored = restore_compiled(record)
        require(set(record) == set(compiled[name]) and all(torch.equal(v, compiled[name][k]) for k, v in record.items()),
            'Compiled export tensor differs.')
        require(restored.cost() == costs[name], 'Compiled cost differs.')
    files = {n: sha(directory/n) for n in ('fitted.pt', 'results.json', 'invocation.json',
        'selection_lock.json', 'predictions.jsonl', 'compiled.pt')}
    return {**row, 'metrics': metrics, 'by_head': by_head, 'selection': selections,
        'fits': report['fits'], 'compiled_equivalence': equivalence, 'compiled_costs': costs,
        'files': files, 'seconds': report['seconds'], 'prediction_groups_exact': 60,
        'inherited_prediction_groups_exact': 240+inherited['legacy_prediction_groups_exact']}


def compare(source, prior_source, *, export_compiled=False, progress=None):
    source = Path(source); prior_source = Path(prior_source)
    status = json.loads((source/'status.json').read_text())
    expected = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())['runs']
    require(status['complete'] and len(status['runs']) == 12 and
        all(r['status'] == 'complete' and r['returncode'] == 0 for r in status['runs']), 'All twelve runs must finish.')
    require([r['name'] for r in status['runs']] == [r['name'] for r in expected], 'Graph inventory changed.')
    runs = []; rebuilt = {}
    for row in expected:
        runs.append(audit_run(source, prior_source, row, rebuilt, export_compiled=export_compiled))
        if progress: progress({'audited': len(runs), 'total': 12, 'name': row['name']})
    groups = {}
    for policy in ('original', 'gated'):
        for schedule in ('dropout', 'balanced'):
            members = [r for r in runs if (r['publication_policy'], r['source_schedule']) == (policy, schedule)]
            groups[policy+'/'+schedule] = {name: {metric: {
                'mean': statistics.mean(r['metrics'][name][metric] for r in members),
                'min_graph': min(r['metrics'][name][metric] for r in members),
                'max_graph': max(r['metrics'][name][metric] for r in members)}
                for metric in members[0]['metrics'][name]} for name in members[0]['metrics']}
    return {'schema_version': 1, 'verified_models': 12, 'fitted_heads': 72,
        'inherited_fitted_heads_verified': 432, 'settings': SETTINGS, 'groups': groups, 'runs': runs,
        'additional_expert_executions': 0,
        'selected_nonzero': {str(c): {family: sum(r['selection'][str(c)][family]['selected_strength'] > 0 for r in runs)
            for family in ('penalty_0', 'preserved')} for c in COUNTS},
        'compiled_prediction_cases_exact': sum(v['prediction_cases_exact'] for r in runs for v in r['compiled_equivalence'].values()),
        'prediction_groups_exact': sum(r['prediction_groups_exact'] for r in runs),
        'inherited_prediction_groups_exact': sum(r['inherited_prediction_groups_exact'] for r in runs),
        'integrity': {'saved_objectives_and_gradients_rebuilt': True, 'validation_choices_reproduced': True,
            'all_evaluation_predictions_exact': True, 'compiled_export_predictions_exact': True,
            'frozen_graphs_inputs_and_normalization_verified': True, 'nested_cohort_and_split_ids_checked': True}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('artifacts/full_rank_v1'))
    p.add_argument('--prior-source', type=Path, default=Path('artifacts/cohort_reproduction_v1'))
    p.add_argument('--out', type=Path, required=True); p.add_argument('--export-compiled', action='store_true')
    p.add_argument('--threads', type=int, default=2); args = p.parse_args()
    if args.threads < 1: p.error('Positive thread count required.')
    torch.set_num_threads(args.threads)
    result = compare(ROOT/args.source, ROOT/args.prior_source, export_compiled=args.export_compiled,
        progress=lambda r: print(json.dumps(r), flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True); write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'runs')}), flush=True)


if __name__ == '__main__':
    main()
