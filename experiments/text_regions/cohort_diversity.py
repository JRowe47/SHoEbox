"""Reproduce the missing 960-versus-3840 decoder-cohort experiment.

Both arms use fresh nested training features, one unchanged legacy linear base,
and the same retained validation/evaluation caches. Completed fits are saved
atomically and can be resumed without fitting them again.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .emission_adapter import ACTIVATIONS, ADAPTER_SEEDS, compact_capture, same_protocol, sha
from .readout_compare import expected_cohort
from .readout_diagnostic import ByteDecoder, SUBSETS, collect_features, controlled_batch, visible
from .residual_adapter import (SELECTION_RULE, ResidualHead, base_fingerprint, load_residuals,
    STRENGTHS, save_tensors, score_residuals, select_strength, write_bytes, write_json)
from .emission_adapter import EmissionAdapter
from .shared_preservation import choose_penalty, fit_preserving
from .view_data import make_view_episode
from .view_eval import FACTORS
from .view_train import load_view_checkpoint, tensor_fingerprint

ROOT = Path(__file__).resolve().parents[2]
COUNTS = (64, 256)
PENALTIES = (0., 1., 10.)
SETTINGS = {'counts_per_cell': list(COUNTS), 'penalties': list(PENALTIES),
    'seeds': list(ADAPTER_SEEDS), 'activations': list(ACTIVATIONS),
    'hidden': 16, 'l2': .001, 'max_iter': 400, 'margin_cap': 1.}
SOURCE_NAMES = ('cohort_diversity.py', 'shared_preservation.py', 'residual_adapter.py',
    'emission_adapter.py', 'readout_diagnostic.py', 'view_data.py', 'view_eval.py',
    'view_train.py', 'transfer_eval.py', 'model.py', 'node.py', 'graph.py', 'ports.py',
    'state.py', 'learning.py', 'config.py')


def source_hashes():
    files = [Path(__file__).with_name(n) for n in SOURCE_NAMES]
    files += sorted((ROOT / 'experiments/hat_minkron').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def cohort_coverage(protocol):
    """Count observed templates, full facts and traces without ID headers."""
    if protocol['split'] != 'train':
        raise ValueError('Training protocol required.')
    count = protocol['count_per_cell']; identities = set(); facts = set()
    signatures = set(); traces = set(); queries = Counter()
    for control, views in SUBSETS.items():
        for query in FACTORS:
            if not visible(control, query):
                continue
            batch = controlled_batch('train', 0, count, control=control, query=query,
                **{k: protocol[k] for k in ('seed', 'gap', 'query_every')})
            for i in range(count):
                episode = make_view_episode('train', i, query_type=query,
                    **{k: protocol[k] for k in ('seed', 'gap', 'query_every')})
                f = tuple(episode.factors[q] for q in FACTORS)
                identities.add(episode.identity); facts.add(f); queries[query] += 1
                signatures.add((control, query, f,
                    tuple((v, episode.template_ids[v]) for v in views)))
                trace = [(control, query)]
                for view in views:
                    mask = batch.availability[view][i, 7:]
                    observed = batch.inputs[view][i, 7:] * mask
                    trace.append((view, observed.tolist(), mask.tolist()))
                traces.add(hashlib.sha256(json.dumps(trace).encode()).hexdigest())
    return {'cases': sum(queries.values()), 'count_per_cell': count,
        'distinct_episode_ids': len(identities), 'full_fact_combinations': len(facts),
        'observed_template_signatures': len(signatures), 'traces_without_id_headers': len(traces),
        'query_counts': dict(queries),
        'signature_definition': 'source subset, query, all three facts, templates of retained sources'}


def nested_prefix(data, count):
    """Select each source/query cell's first count rows without graph recapture."""
    protocol = data['protocol']; old_count = protocol['count_per_cell']
    if protocol['split'] != 'train' or type(count) is not int or count < 32 or count % 32 or count > old_count:
        raise ValueError('Expected a complete nested training cohort.')
    indices = []; groups = []; cursor = 0
    for g in data['groups']:
        if g['stop'] - g['start'] != old_count:
            raise ValueError('Unbalanced source/query groups.')
        indices.extend(range(g['start'], g['start'] + count))
        groups.append({**g, 'start': cursor, 'stop': cursor + count,
            'expert_executions': g['expert_executions'] * count // old_count,
            'external_events': g['external_events'] * count // old_count})
        cursor += count
    take = torch.tensor(indices)
    new_protocol = {**protocol, 'count_per_cell': count, 'episodes': cursor,
        'expert_executions': sum(g['expert_executions'] for g in groups),
        'external_events': sum(g['external_events'] for g in groups)}
    expected = expected_cohort(new_protocol)
    new_protocol.update({k: expected[k] for k in ('case_sha256', 'inputs_sha256')})
    return {'answers': data['answers'][take].clone(),
        **{k: [data[k][i] for i in indices] for k in ('queries', 'controls', 'identities')},
        'features': {k: v[take].clone() for k, v in data['features'].items()},
        **{k: data[k][take].clone() for k in ('direct_predictions', 'direct_nll_per_case')},
        'groups': groups, 'protocol': new_protocol}


def validate_splits(training, validation, evaluations):
    if training['protocol']['split'] != 'train' or validation['protocol']['split'] != 'validation':
        raise ValueError('Training and validation splits required.')
    cohorts = {'train': training, 'validation': validation, **evaluations}
    if set(cohorts) != {'train', 'validation', 'test', 'independent'}:
        raise ValueError('Both evaluation cohorts are required.')
    seen = set()
    for split, data in cohorts.items():
        if data['protocol']['split'] != split:
            raise ValueError('Incorrect split label.')
        ids = set(data['identities'])
        if seen & ids:
            raise ValueError('Split identities overlap.')
        seen |= ids
        for key in ('model_tensor_sha256', 'policy', 'gap', 'query_every'):
            if not same_protocol(data['protocol'][key], training['protocol'][key]):
                raise ValueError('Frozen graph or capture policy differs.')


def fit_key(count, penalty, activation, seed):
    return f'{count}/{penalty:g}/{activation}/{seed}'


def head_record(head):
    return {'correction': asdict(head.correction), 'strength': head.strength, 'selection': head.selection}


def restore_head(record, base):
    head = ResidualHead(base, EmissionAdapter(**record['correction']), record['strength'], record['selection'])
    if head.correction.fit['base_head_sha256'] != base_fingerprint(base):
        raise ValueError('Checkpoint belongs to another frozen base.')
    if not torch.equal(head.correction.mean, base.mean) or not torch.equal(head.correction.scale, base.scale):
        raise ValueError('Correction normalization differs from the frozen base.')
    width, hidden = len(base.mean), head.correction.fit['hidden']
    if ([tuple(getattr(head.correction, k).shape) for k in ('w1', 'b1', 'w2', 'b2')]
            != [(width, hidden), (hidden,), (hidden, 256), (256,)]
            or head.strength not in STRENGTHS or head.selection.get('selected_strength') != head.strength
            or head.selection.get('split') != 'validation' or head.selection.get('rule') != SELECTION_RULE):
        raise ValueError('Invalid saved correction shape or validation selection.')
    for name in ('w1', 'b1', 'w2', 'b2'):
        value = getattr(head.correction, name)
        if not torch.isfinite(value).all() or value.requires_grad:
            raise ValueError('Invalid saved correction tensor.')
    return head


def bundles_for_count(count, training, validation, base, records):
    common = {'schema_version': 1, 'base': base, 'fit_protocol': training['protocol'],
        'validation_protocol': validation['protocol'], 'fit_identities': sorted(set(training['identities'])),
        'validation_identities': sorted(set(validation['identities'])),
        'initialization_seeds': list(ADAPTER_SEEDS), 'selection_rule': SELECTION_RULE}
    bundles = {f'penalty_{p:g}': {**common, 'heads': {}} for p in PENALTIES}
    selected = {**common, 'heads': {}}
    for seed in ADAPTER_SEEDS:
        for activation in ACTIVATIONS:
            name = f'{activation}/{seed}'; choices = {}
            for p in PENALTIES:
                head = restore_head(records[fit_key(count, p, activation, seed)], base)
                bundles[f'penalty_{p:g}']['heads'][name] = head
                if p:
                    choices[p] = head
            selected['heads'][name] = choose_penalty(choices)
    return {**bundles, 'selected': selected}


def verify_resume(saved, invocation, capture_hash, base):
    if (not same_protocol(saved['invocation'], invocation) or saved['capture_sha256'] != capture_hash
            or base_fingerprint(ByteDecoder(**saved['base'])) != base_fingerprint(base)):
        raise ValueError('Resume source, settings, inputs or frozen base changed.')
    expected = {fit_key(c, p, a, s) for c in COUNTS for p in PENALTIES
                for a in ACTIVATIONS for s in ADAPTER_SEEDS}
    if not set(saved['heads']) <= expected:
        raise ValueError('Unexpected resumed fit.')
    for key, record in saved['heads'].items():
        count, penalty, activation, seed = key.split('/')
        head = restore_head(record, base); fit = head.correction.fit
        if (fit['examples'] != int(count)*15 or fit['seed'] != int(seed)
                or fit['activation'] != activation or fit['preservation']['penalty'] != float(penalty)
                or fit['max_iter'] != SETTINGS['max_iter'] or fit['hidden'] != SETTINGS['hidden']
                or fit['l2'] != SETTINGS['l2']):
            raise ValueError('Resumed fit metadata differs.')


def run(run_row, out, *, threads=2, resume=False):
    torch.set_num_threads(threads)
    if threads < 1 or (out.exists() and not resume):
        raise ValueError('Use a new output directory or explicitly resume.')
    out.mkdir(parents=True, exist_ok=True)
    prior_dir = ROOT / run_row['residual_run']; emission = ROOT / run_row['prior_run']
    inputs = {'checkpoint': ROOT / run_row['checkpoint'], 'reference': ROOT / run_row['reference'],
        'residuals': prior_dir / 'residuals.pt', 'validation': prior_dir / 'validation.pt',
        'features': emission / 'features.pt'}
    input_hashes = {k: sha(p) for k, p in inputs.items()}
    if input_hashes['checkpoint'] != run_row['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    invocation = {'schema_version': 1, 'run': run_row, 'settings': SETTINGS,
        'source_sha256': source_hashes(), 'input_sha256': input_hashes,
        'runtime': {'torch': str(torch.__version__), 'threads': threads}}
    invocation_path = out / 'invocation.json'
    if invocation_path.exists() and not same_protocol(json.loads(invocation_path.read_text()), invocation):
        raise ValueError('Cannot resume a changed invocation.')
    write_json(invocation_path, invocation)
    if (out / 'results.json').exists():
        old = json.loads((out / 'results.json').read_text())
        if old['complete'] and all(sha(out / n) == h for n, h in old['output_sha256'].items()):
            return
        raise ValueError('Completed run failed output verification.')
    started = time.perf_counter()

    def progress(row):
        write_json(out / 'progress.json', {'run': run_row['name'], **row})
        print(json.dumps(row), flush=True)

    inherited = load_residuals(inputs['residuals']); base = inherited['base']
    legacy = torch.load(inputs['features'], map_location='cpu', weights_only=True)['datasets']
    validation = torch.load(inputs['validation'], map_location='cpu', weights_only=True)
    capture_path = out / 'training.pt'
    if capture_path.exists():
        captured = torch.load(capture_path, map_location='cpu', weights_only=True)
        if not same_protocol(captured['invocation'], invocation):
            raise ValueError('Captured training features have different provenance.')
        large = captured['training']
    else:
        model, _, _, saved = load_view_checkpoint(inputs['checkpoint'], restore_rng=False)
        if model.cfg.rounds != 1 or tensor_fingerprint(model) != legacy['train']['protocol']['model_tensor_sha256']:
            raise ValueError('Expected the frozen K=1 reference graph.')
        protocol = legacy['train']['protocol']
        large = compact_capture(collect_features(model, split='train', count_per_cell=256,
            **{k: protocol[k] for k in ('batch_size', 'segment', 'seed', 'gap', 'query_every')},
            on_cell=lambda row: progress({'phase': 'capture', **row})))
        if tensor_fingerprint(model) != protocol['model_tensor_sha256']:
            raise ValueError('Frozen graph changed during capture.')
        save_tensors(capture_path, {'schema_version': 1, 'invocation': invocation, 'training': large})
        del model
    training = {count: nested_prefix(large, count) for count in COUNTS}
    evaluations = {k: legacy[k] for k in ('test', 'independent')}
    for data in training.values():
        validate_splits(data, validation, evaluations)
    fitted_path = out / 'fitted.pt'; capture_hash = sha(capture_path)
    if fitted_path.exists():
        fitted = torch.load(fitted_path, map_location='cpu', weights_only=True)
        verify_resume(fitted, invocation, capture_hash, base)
    else:
        fitted = {'schema_version': 1, 'invocation': invocation, 'capture_sha256': capture_hash,
            'base': asdict(base), 'heads': {}}
    for count in COUNTS:
        for seed in ADAPTER_SEEDS:
            for activation in ACTIVATIONS:
                for penalty in PENALTIES:
                    key = fit_key(count, penalty, activation, seed)
                    if key in fitted['heads']:
                        continue
                    head = fit_preserving(training[count], base, penalty=penalty, activation=activation,
                        seed=seed, hidden=16, l2=.001, max_iter=400, margin_cap=1.)
                    head = select_strength(head, validation, fit_identities=training[count]['identities'])
                    fitted['heads'][key] = head_record(head)
                    save_tensors(fitted_path, fitted)
                    progress({'phase': 'fit', 'completed_fits': len(fitted['heads']), 'total_fits': 36,
                        'key': key, 'strength': head.strength, 'fit_seconds': head.correction.fit['seconds']})
    fitted = torch.load(fitted_path, map_location='cpu', weights_only=True)
    verify_resume(fitted, invocation, capture_hash, base)
    report = {'schema_version': 1, 'complete': False, 'invocation': invocation,
        'base_head_sha256': base_fingerprint(base), 'additional_expert_executions': large['protocol']['expert_executions'],
        'coverage': {str(c): cohort_coverage(training[c]['protocol']) for c in COUNTS},
        'legacy_training_max_abs_drift': float((training[64]['features']['final/raw'] - legacy['train']['features']['final/raw']).abs().max()),
        'fits': {k: v['correction']['fit'] for k, v in fitted['heads'].items()},
        'selection': {}, 'evaluations': {}}
    rows = []
    for count in COUNTS:
        bundles = bundles_for_count(count, training[count], validation, base, fitted['heads'])
        report['selection'][str(count)] = {k: {n: h.selection for n, h in b['heads'].items()} for k, b in bundles.items()}
        report['evaluations'][str(count)] = {}
        for family, bundle in bundles.items():
            report['evaluations'][str(count)][family] = {}
            for split, data in evaluations.items():
                scores, predictions = score_residuals(data, bundle)
                report['evaluations'][str(count)][family][split] = scores
                rows.extend({'count_per_cell': count, 'family': family, **r} for r in predictions)
    write_bytes(out / 'predictions.jsonl', ''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows).encode())
    if invocation['source_sha256'] != source_hashes() or input_hashes != {k: sha(p) for k, p in inputs.items()}:
        raise ValueError('Measured source or inputs changed during the run.')
    report.update(complete=True, seconds=time.perf_counter()-started,
        output_sha256={n: sha(out/n) for n in ('training.pt', 'fitted.pt', 'predictions.jsonl', 'invocation.json')})
    write_json(out / 'results.json', report)
    progress({'phase': 'complete', 'completed_fits': 36, 'total_fits': 36, 'seconds': report['seconds']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())
    row = next((r for r in manifest['runs'] if r['name'] == args.run), None)
    if row is None:
        parser.error('Unknown frozen graph.')
    run(row, args.out, threads=args.threads, resume=args.resume)


if __name__ == '__main__':
    main()
