"""Deterministic full-rank byte refits on the retained nested cohorts.

The graph, feature normalization and old decoder are immutable. A fitted head
uses all 256 bytes; query/source metadata is used only for training and scoring.
Validation selects interpolation strength and preservation penalty before test.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .cohort_diversity import COUNTS, nested_prefix, validate_splits
from .emission_adapter import same_protocol, sha
from .readout_diagnostic import ByteDecoder
from .residual_adapter import (SELECTION_RULE, STRENGTHS, balanced_weights,
    base_fingerprint, save_tensors, score_residuals, select_strength, write_bytes, write_json)
from .shared_preservation import (answer_margins, choose_penalty, preservation_loss, shared_anchors)

ROOT = Path(__file__).resolve().parents[2]
PENALTIES = (0., 1., 10.)
SETTINGS = {'counts_per_cell': list(COUNTS), 'penalties': list(PENALTIES),
    'l2': .001, 'max_iter': 1000, 'margin_cap': 1.,
    'normalization': 'unchanged legacy training mean/scale',
    'regularization': 'entire refitted matrix including bias',
    'initialization': 'frozen baseline; deterministic, no RNG use'}


@dataclass(frozen=True)
class FullRankHead:
    base: ByteDecoder
    refit: ByteDecoder
    strength: float = 1.
    selection: dict = None

    @property
    def correction(self):
        return replace(self.refit, weight=self.refit.weight-self.base.weight)

    def logits(self, features):
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError('Refit strength must be in [0,1].')
        baseline = self.base.logits(features)
        return baseline if self.strength == 0 else baseline+self.strength*self.correction.logits(features)

    def predict(self, features):
        return self.logits(features).argmax(-1)


def validate_decoder(head):
    d = len(head.mean)
    values = (head.mean, head.scale, head.weight)
    if (head.mean.shape != (d,) or head.scale.shape != (d,) or head.weight.shape != (d+1, 256)
            or not (head.scale > 0).all() or any(not t.is_floating_point() or
            t.requires_grad or not torch.isfinite(t).all() for t in values)):
        raise ValueError('Invalid frozen byte decoder.')


def fit_full_rank(training, base, *, penalty=0., l2=.001, max_iter=1000, margin_cap=1.):
    validate_decoder(base)
    if training['protocol']['split'] != 'train':
        raise ValueError('Full-rank fitting requires the training split.')
    if (not math.isfinite(penalty) or penalty < 0 or not math.isfinite(l2) or l2 <= 0
            or type(max_iter) is not int or max_iter < 1):
        raise ValueError('Invalid full-rank fitting settings.')
    x = training['features']['final/raw'].detach().cpu().double()
    y = training['answers'].detach().cpu().long()
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(base.mean) or
            y.shape != (len(x),) or not torch.isfinite(x).all() or ((y < 0) | (y > 255)).any()):
        raise ValueError('Invalid full-rank training inputs.')
    before = base_fingerprint(base)
    frozen = ByteDecoder(*(t.detach().clone() for t in (base.mean, base.scale, base.weight)), dict(base.fit))
    mass = balanced_weights(training)
    design = torch.cat(((x-frozen.mean)/frozen.scale, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    baseline = design @ frozen.weight
    mask, target, anchors = shared_anchors(baseline, y, training['queries'], mass, margin_cap)
    weight = frozen.weight.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([weight], max_iter=max_iter, history_size=30,
        tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        z = design @ weight
        loss = (F.cross_entropy(z, y, reduction='none')*mass).sum() + .5*l2*weight.square().sum()
        if penalty:
            loss = loss + penalty*preservation_loss(z, y, target, anchors)
        loss.backward()
        return loss

    started = time.perf_counter()
    with torch.enable_grad():
        optimizer.step(closure)
        objective = float(closure().detach())
    with torch.no_grad():
        z = design @ weight
        nll = float((F.cross_entropy(z, y, reduction='none')*mass).sum())
        accuracy = float(((z.argmax(-1) == y).double()*mass).sum())
        anchor_loss = float(preservation_loss(z, y, target, anchors))
        violations = int(((answer_margins(z, y) < target-1e-9) & mask).sum())
        lost = int(((z.argmax(-1) != y) & mask).sum())
    if not math.isfinite(objective) or not torch.isfinite(weight).all():
        raise RuntimeError('Nonfinite full-rank fit.')
    if before != base_fingerprint(base) or before != base_fingerprint(frozen):
        raise RuntimeError('Frozen baseline changed.')
    fit = {'examples': len(x), 'width': x.shape[1], 'output_bytes': 256,
        'parameters': weight.numel(), 'l2': l2, 'max_iter': max_iter,
        'iterations': optimizer.state[weight]['n_iter'],
        'gradient_max': float(weight.grad.abs().max()), 'objective': objective,
        'weighted_training_nll': nll, 'weighted_training_accuracy': accuracy,
        'seconds': time.perf_counter()-started, 'base_head_sha256': before,
        'base_frozen_verified': True, 'optimizer': 'full-batch L-BFGS, strong Wolfe',
        'normalization': SETTINGS['normalization'], 'initialization': SETTINGS['initialization'],
        'regularization': '0.5 * l2 * squared sum of entire refitted matrix including bias',
        'preservation': {'penalty': penalty, 'margin_cap': margin_cap,
            'anchor_count': int(mask.sum()), 'shared_count': training['queries'].count('shared'),
            'unweighted_penalty': anchor_loss, 'margin_violations': violations,
            'baseline_correct_answers_lost': lost}}
    return FullRankHead(frozen, ByteDecoder(frozen.mean.clone(), frozen.scale.clone(),
        weight.detach().clone(), fit), 1., {})


def source_hashes():
    names = ('full_rank_readout.py', 'compiled_readout.py', 'cohort_diversity.py',
        'readout_diagnostic.py', 'residual_adapter.py', 'shared_preservation.py', 'emission_adapter.py')
    return {str(Path(__file__).with_name(n).relative_to(ROOT)): sha(Path(__file__).with_name(n)) for n in names}


def record_head(head):
    return {'refit': asdict(head.refit), 'strength': head.strength, 'selection': head.selection}


def restore_head(record, base):
    refit = ByteDecoder(**record['refit']); validate_decoder(refit)
    h = FullRankHead(base, refit, record['strength'], record['selection'])
    if (refit.fit['base_head_sha256'] != base_fingerprint(base)
            or not torch.equal(base.mean, refit.mean) or not torch.equal(base.scale, refit.scale)
            or h.strength not in STRENGTHS or h.selection.get('selected_strength') != h.strength
            or h.selection.get('split') != 'validation' or h.selection.get('rule') != SELECTION_RULE):
        raise ValueError('Invalid refit provenance, normalization or validation selection.')
    return h


def load_inputs(source, row):
    source = Path(source); old = source/row['name']
    report = json.loads((old/'results.json').read_text())
    if not report['complete'] or report['invocation']['run'] != row:
        raise ValueError('Incomplete or mismatched cohort source.')
    if any(sha(old/n) != digest for n, digest in report['output_sha256'].items()):
        raise ValueError('Retained cohort output changed.')
    paths = {'training': old/'training.pt', 'cohort_fits': old/'fitted.pt',
        'cohort_results': old/'results.json', 'checkpoint': ROOT/row['checkpoint'],
        'reference': ROOT/row['reference'], 'residuals': ROOT/row['residual_run']/'residuals.pt',
        'validation': ROOT/row['residual_run']/'validation.pt', 'features': ROOT/row['prior_run']/'features.pt'}
    hashes = {k: sha(p) for k, p in paths.items()}
    if any(hashes[k] != v for k, v in report['invocation']['input_sha256'].items()):
        raise ValueError('Frozen input changed since cohort study.')
    saved = torch.load(paths['cohort_fits'], map_location='cpu', weights_only=True)
    base = ByteDecoder(**saved['base']); validate_decoder(base)
    if base_fingerprint(base) != report['base_head_sha256']:
        raise ValueError('Frozen baseline fingerprint differs.')
    capture = torch.load(paths['training'], map_location='cpu', weights_only=True)
    if not same_protocol(capture['invocation'], report['invocation']):
        raise ValueError('Training capture provenance differs.')
    large = capture['training']
    training = {c: nested_prefix(large, c) for c in COUNTS}
    validation = torch.load(paths['validation'], map_location='cpu', weights_only=True)
    legacy = torch.load(paths['features'], map_location='cpu', weights_only=True)['datasets']
    evaluation = {s: legacy[s] for s in ('test', 'independent')}
    for data in training.values():
        validate_splits(data, validation, evaluation)
    return base, training, validation, evaluation, saved, hashes


def verify_resume(saved, invocation, base):
    if (saved.get('schema_version') != 1 or not same_protocol(saved['invocation'], invocation)
            or base_fingerprint(ByteDecoder(**saved['base'])) != base_fingerprint(base)):
        raise ValueError('Resume source, configuration, inputs or baseline changed.')
    allowed = {f'{c}/{p:g}' for c in COUNTS for p in PENALTIES}
    if not set(saved['heads']) <= allowed:
        raise ValueError('Unexpected resumed refit.')
    for key, value in saved['heads'].items():
        h = restore_head(value, base); count, penalty = key.split('/'); fit = h.refit.fit
        if (fit['examples'] != int(count)*15 or fit['preservation']['penalty'] != float(penalty)
                or fit['l2'] != SETTINGS['l2'] or fit['max_iter'] != SETTINGS['max_iter']
                or fit['preservation']['margin_cap'] != SETTINGS['margin_cap']
                or fit['parameters'] != base.weight.numel()):
            raise ValueError('Resumed fit settings differ.')


def bundle_for_count(count, base, training, validation, records):
    heads = {f'penalty_{p:g}': restore_head(records[f'{count}/{p:g}'], base) for p in PENALTIES}
    heads['preserved'] = choose_penalty({p: heads[f'penalty_{p:g}'] for p in PENALTIES if p})
    return {'base': base, 'heads': heads, 'fit_protocol': training['protocol'],
        'fit_identities': training['identities'], 'validation_identities': validation['identities']}


def run(source, row, out, *, threads=2, resume=False):
    out = Path(out)
    if threads < 1 or (out.exists() and not resume):
        raise ValueError('Use a new directory or --resume, and positive threads.')
    torch.set_num_threads(threads)
    base, training, validation, evaluations, _, hashes = load_inputs(source, row)
    invocation = {'schema_version': 1, 'run': row, 'settings': SETTINGS,
        'source_sha256': source_hashes(), 'input_sha256': hashes,
        'runtime': {'torch': str(torch.__version__), 'threads': threads}}
    out.mkdir(parents=True, exist_ok=True)
    if (out/'invocation.json').exists() and not same_protocol(json.loads((out/'invocation.json').read_text()), invocation):
        raise ValueError('Cannot resume a changed invocation.')
    write_json(out/'invocation.json', invocation)
    saved_path = out/'fitted.pt'
    saved = torch.load(saved_path, map_location='cpu', weights_only=True) if saved_path.exists() else {
        'schema_version': 1, 'invocation': invocation, 'base': asdict(base), 'heads': {}}
    verify_resume(saved, invocation, base)
    if (out/'results.json').exists():
        result = json.loads((out/'results.json').read_text())
        if not result['complete'] or any(sha(out/n) != h for n, h in result['output_sha256'].items()):
            raise ValueError('Completed refit output changed.')
        return
    started = time.perf_counter()
    for count in COUNTS:
        for penalty in PENALTIES:
            key = f'{count}/{penalty:g}'
            if key in saved['heads']:
                continue
            h = fit_full_rank(training[count], base, penalty=penalty,
                **{k: SETTINGS[k] for k in ('l2', 'max_iter', 'margin_cap')})
            h = select_strength(h, validation, fit_identities=training[count]['identities'])
            saved['heads'][key] = record_head(h); save_tensors(saved_path, saved)
            print(json.dumps({'fit': key, 'completed': len(saved['heads']), 'total': 6,
                'strength': h.strength, 'iterations': h.refit.fit['iterations'],
                'seconds': h.refit.fit['seconds']}), flush=True)
    result = {'schema_version': 1, 'complete': False, 'invocation': invocation,
        'base_head_sha256': base_fingerprint(base), 'additional_expert_executions': 0,
        'fits': {k: v['refit']['fit'] for k, v in saved['heads'].items()},
        'selection': {}, 'evaluations': {}}
    bundles = {c: bundle_for_count(c, base, training[c], validation, saved['heads']) for c in COUNTS}
    result['selection'] = {str(c): {n: h.selection for n, h in b['heads'].items()} for c, b in bundles.items()}
    write_json(out/'selection_lock.json', {'selection_inputs': ['train', 'validation'], 'selection': result['selection']})
    rows = []
    for count, bundle in bundles.items():
        result['evaluations'][str(count)] = {}
        for split, data in evaluations.items():
            scores, predictions = score_residuals(data, bundle)
            result['evaluations'][str(count)][split] = scores
            rows.extend({'count_per_cell': count, **r} for r in predictions)
    write_bytes(out/'predictions.jsonl', ''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows).encode())
    if invocation['source_sha256'] != source_hashes():
        raise RuntimeError('Measured source changed during fitting.')
    result.update(complete=True, seconds=time.perf_counter()-started,
        output_sha256={n: sha(out/n) for n in ('fitted.pt', 'invocation.json', 'selection_lock.json', 'predictions.jsonl')})
    write_json(out/'results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/cohort_reproduction_v1'))
    parser.add_argument('--run', required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rows = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())['runs']
    row = next((r for r in rows if r['name'] == args.run), None)
    if row is None:
        parser.error('Unknown frozen graph.')
    run(ROOT/args.source, row, ROOT/args.out, threads=args.threads, resume=args.resume)


if __name__ == '__main__':
    main()
