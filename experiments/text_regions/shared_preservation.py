"""Train residual byte heads with a shared-answer margin preservation loss.

Query metadata is used only to define training anchors. Prediction remains a
single feature-only 256-byte decoder. All tensors in the baseline stay frozen.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .emission_adapter import ACTIVATIONS, ADAPTER_SEEDS, same_protocol, sha
from .residual_adapter import (
    SELECTION_RULE, balanced_weights, base_fingerprint, initialize_residual,
    load_residuals, save_residuals, score_residuals, select_strength,
    write_bytes, write_json,
)

PENALTIES = (1., 10.)
MARGIN_CAP = 1.
JOINT_RULE = ('minimum query-balanced validation NLL under unchanged all-source/shared '
              'and B-only/shared correct-count guards; ties prefer smaller strength, '
              'then smaller penalty')
ROOT = Path(__file__).resolve().parents[2]


def answer_margins(logits, answers):
    """True-byte logit minus the largest other-byte logit, over all 256 bytes."""
    indices = torch.arange(len(answers), device=logits.device)
    rivals = logits.scatter(1, answers[:, None], -torch.inf).amax(-1)
    return logits[indices, answers] - rivals


def shared_anchors(baseline, answers, queries, mass, cap=MARGIN_CAP):
    if (not math.isfinite(cap) or cap <= 0 or len(queries) != len(answers)
            or any(q not in ('shared', 'left', 'right') for q in queries)):
        raise ValueError('Invalid shared training metadata or margin cap.')
    shared = torch.tensor([q == 'shared' for q in queries])
    margin = answer_margins(baseline, answers).detach()
    mask = shared & (baseline.argmax(-1) == answers) & (margin > 0)
    weights = mass * mask
    if mask.any():
        weights = weights / weights.sum()
    target = margin.clamp(min=0., max=cap)
    return mask, target, weights


def preservation_loss(logits, answers, target, anchor_weights):
    deficit = (target - answer_margins(logits, answers)).clamp_min(0)
    return (anchor_weights * deficit.square()).sum()


def fit_preserving(training, base, *, penalty=10., margin_cap=MARGIN_CAP,
                   hidden=16, seed=20261121, activation='gelu', l2=.001,
                   max_iter=400):
    if training['protocol']['split'] != 'train':
        raise ValueError('Preservation anchors require the training split.')
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Penalty must be finite and nonnegative.')
    x = training['features']['final/raw'].detach().cpu().double()
    y = training['answers'].detach().cpu().long()
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(base.mean)
            or y.shape != (len(x),) or not torch.isfinite(x).all()
            or ((y < 0) | (y > 255)).any() or not math.isfinite(l2)
            or l2 <= 0 or max_iter < 1):
        raise ValueError('Invalid preserving fit inputs.')
    mass = balanced_weights(training)
    mass = mass / mass.sum()
    head = initialize_residual(base, hidden=hidden, seed=seed, activation=activation)
    baseline = head.base.logits(x).detach()
    before = base_fingerprint(base)
    mask, target, anchor_weights = shared_anchors(baseline, y, training['queries'], mass, margin_cap)
    z = (x - head.base.mean) / head.base.scale
    names = ('w1', 'b1', 'w2', 'b2')
    parameters = [getattr(head.correction, n).clone().requires_grad_(True) for n in names]
    w1, b1, w2, b2 = parameters

    def logits():
        h = z @ w1 + b1
        return baseline + (F.gelu(h) if activation == 'gelu' else h) @ w2 + b2

    if not torch.equal(logits().detach(), baseline):
        raise RuntimeError('Residual initialization changed the baseline.')
    optimizer = torch.optim.LBFGS(parameters, max_iter=max_iter, history_size=30,
        tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        predicted = logits()
        loss = (F.cross_entropy(predicted, y, reduction='none') * mass).sum()
        loss = loss + .5 * l2 * sum(p.square().sum() for p in parameters)
        if penalty:
            loss = loss + penalty * preservation_loss(predicted, y, target, anchor_weights)
        loss.backward()
        return loss

    started = time.perf_counter()
    with torch.enable_grad():
        optimizer.step(closure)
        objective = float(closure().detach())
    with torch.no_grad():
        result = logits()
        nll = float((F.cross_entropy(result, y, reduction='none') * mass).sum())
        anchor_loss = float(preservation_loss(result, y, target, anchor_weights))
        accuracy = float(((result.argmax(-1) == y).double() * mass).sum())
        violations = int(((answer_margins(result, y) < target - 1e-9) & mask).sum())
        lost = int(((result.argmax(-1) != y) & mask).sum())
    if not math.isfinite(objective) or any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError('Nonfinite preserving fit.')
    if base_fingerprint(base) != before or base_fingerprint(head.base) != before:
        raise RuntimeError('Frozen linear head changed.')
    metadata = {**head.correction.fit, 'examples': len(x), 'l2': l2,
        'max_iter': max_iter, 'iterations': optimizer.state[w1]['n_iter'],
        'gradient_max': max(float(p.grad.abs().max()) for p in parameters),
        'objective': objective, 'weighted_training_nll': nll,
        'weighted_training_accuracy': accuracy, 'seconds': time.perf_counter() - started,
        'optimizer': 'full-batch L-BFGS, strong Wolfe',
        'normalization': 'frozen linear head training mean/scale',
        'base_frozen_verified': True,
        'regularization': '0.5 * l2 * squared sum of correction weights and biases only',
        'preservation': {'penalty': penalty, 'margin_cap': margin_cap,
            'anchor_rule': 'training shared cases with positive, correct frozen-baseline margin',
            'anchor_count': int(mask.sum()), 'shared_count': training['queries'].count('shared'),
            'unweighted_penalty': anchor_loss, 'margin_violations': violations,
            'baseline_correct_answers_lost': lost}}
    correction = replace(head.correction, fit=metadata,
        **dict(zip(names, (p.detach().clone() for p in parameters))))
    return replace(head, correction=correction)


def choose_penalty(candidates):
    """Choose among already validation-selected heads without looking at test data."""
    if not candidates:
        raise ValueError('At least one preservation candidate required.')
    for penalty, head in candidates.items():
        if (not math.isfinite(penalty) or penalty <= 0
                or head.correction.fit['preservation']['penalty'] != penalty
                or head.selection.get('split') != 'validation'
                or head.selection.get('rule') != SELECTION_RULE):
            raise ValueError('Invalid preservation candidate or selection provenance.')
    best = min(h.selection['selected_weighted_nll'] for h in candidates.values())
    eligible = [(p, h) for p, h in candidates.items()
                if h.selection['selected_weighted_nll'] <= best + 1e-12]
    penalty, head = min(eligible, key=lambda item: (item[1].strength, item[0]))
    choice = {'rule': JOINT_RULE, 'selected_penalty': penalty,
        'candidates': {str(p): {'strength': h.strength, 'selection': h.selection}
                       for p, h in candidates.items()}}
    return replace(head, selection={**head.selection, 'preservation_choice': choice})


def fit_candidates(training, validation, base, *, penalties=PENALTIES,
                   seeds=ADAPTER_SEEDS, hidden=16, l2=.001, max_iter=400,
                   margin_cap=MARGIN_CAP, progress=None):
    if training['protocol']['split'] != 'train' or validation['protocol']['split'] != 'validation':
        raise ValueError('Training and validation splits required.')
    if (not penalties or len(set(penalties)) != len(penalties)
            or any(not math.isfinite(p) or p <= 0 for p in penalties)
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError('Distinct positive penalties and initialization seeds required.')
    fit_ids = sorted(set(training['identities']))
    valid_ids = sorted(set(validation['identities']))
    if set(fit_ids) & set(valid_ids):
        raise ValueError('Training and validation overlap.')
    for key in ('model_tensor_sha256', 'policy', 'gap', 'query_every'):
        if not same_protocol(training['protocol'][key], validation['protocol'][key]):
            raise ValueError('Validation graph or protocol differs.')
    mass = balanced_weights(training)
    x = training['features']['final/raw'].double()
    mean = (x * mass[:, None]).sum(0)
    scale = ((x - mean).square() * mass[:, None]).sum(0).sqrt().clamp_min(1e-6)
    if not torch.equal(mean, base.mean) or not torch.equal(scale, base.scale):
        raise ValueError('Training data differs from frozen normalization.')
    common = {'schema_version': 1, 'base': base, 'fit_protocol': training['protocol'],
        'validation_protocol': validation['protocol'], 'fit_identities': fit_ids,
        'validation_identities': valid_ids, 'initialization_seeds': list(seeds),
        'selection_rule': SELECTION_RULE}
    bundles = {p: {**common, 'heads': {}} for p in penalties}
    selected = {**common, 'heads': {}}
    for seed in seeds:
        for activation in ACTIVATIONS:
            name = f'{activation}/{seed}'
            choices = {}
            for penalty in penalties:
                head = fit_preserving(training, base, penalty=penalty, margin_cap=margin_cap,
                    hidden=hidden, seed=seed, activation=activation, l2=l2, max_iter=max_iter)
                choices[penalty] = select_strength(head, validation, fit_identities=fit_ids)
                bundles[penalty]['heads'][name] = choices[penalty]
                if progress:
                    progress({'fit': name, 'penalty': penalty,
                        'strength': choices[penalty].strength, **head.correction.fit})
            selected['heads'][name] = choose_penalty(choices)
    return bundles, selected


def run(prior_run, emission_run, out, *, threads=2):
    if out.exists() or threads < 1:
        raise ValueError('Use a new output directory and positive thread count.')
    torch.set_num_threads(threads)
    prior = json.loads((prior_run / 'results.json').read_text())
    inputs = {'prior_results': prior_run / 'results.json',
        'prior_residuals': prior_run / 'residuals.pt',
        'validation': prior_run / 'validation.pt',
        'features': emission_run / 'features.pt'}
    hashes = {k: sha(p) for k, p in inputs.items()}
    if (hashes['prior_residuals'] != prior['residuals_sha256']
            or hashes['validation'] != prior['validation_sha256']
            or hashes['features'] != prior['prior_features_sha256']):
        raise ValueError('Restored input artifact differs from prior study.')
    inherited = load_residuals(inputs['prior_residuals'])
    datasets = torch.load(inputs['features'], map_location='cpu', weights_only=True)['datasets']
    validation = torch.load(inputs['validation'], map_location='cpu', weights_only=True)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), Path(__file__).with_name('residual_adapter.py'),
        Path(__file__).with_name('emission_adapter.py'))}
    out.mkdir(parents=True)
    started = time.perf_counter()
    bundles, selected = fit_candidates(datasets['train'], validation, inherited['base'],
        progress=lambda row: print(json.dumps(row), flush=True))
    fitted = {}
    for name, bundle in [(f'penalty_{p:g}', b) for p, b in bundles.items()] + [('selected', selected)]:
        save_residuals(out / (name + '.pt'), bundle)
        fitted[name] = load_residuals(out / (name + '.pt'))
    lock = {'rule': JOINT_RULE, 'selection_inputs': ['training', 'validation'],
        'penalties': list(PENALTIES), 'fit_inputs_sha256': datasets['train']['protocol']['inputs_sha256'],
        'validation_inputs_sha256': validation['protocol']['inputs_sha256'],
        'heads': {n: h.selection for n, h in fitted['selected']['heads'].items()}}
    write_json(out / 'selection_lock.json', lock)
    report = {'schema_version': 1, 'complete': False, 'input_sha256': hashes,
        'source_sha256': sources, 'base_head_sha256': base_fingerprint(inherited['base']),
        'fit_protocol': datasets['train']['protocol'], 'validation_protocol': validation['protocol'],
        'selection': lock, 'additional_expert_executions': 0,
        'fits': {n: {k: h.correction.fit for k, h in b['heads'].items()} for n, b in fitted.items()},
        'evaluations': {}, 'runtime': {'torch': str(torch.__version__), 'threads': threads}}
    rows = []
    for family, bundle in {'prior': inherited, **fitted}.items():
        report['evaluations'][family] = {}
        for split in ('test', 'independent'):
            scores, predictions = score_residuals(datasets[split], bundle)
            report['evaluations'][family][split] = scores
            rows.extend({'family': family, **row} for row in predictions)
    write_bytes(out / 'predictions.jsonl', ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows).encode())
    if sources != {p: sha(ROOT / p) for p in sources} or hashes != {k: sha(p) for k, p in inputs.items()}:
        raise RuntimeError('Measured source or inputs changed during execution.')
    report.update(complete=True, seconds=time.perf_counter() - started,
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out / 'results.json', report)
    print(json.dumps({'complete': True, 'out': str(out), 'seconds': report['seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, required=True)
    parser.add_argument('--emission-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.prior_run, args.emission_run, args.out, threads=args.threads)


if __name__ == '__main__':
    main()
