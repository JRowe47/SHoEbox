"""Reconstruct exposure, checkpoint selection, exports and held-out predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.publication_data import make_batch, batch_digest
from experiments.text_regions.publication_train import build_producer, load, evaluate, forward_batch, validation
from experiments.text_regions.composition_train import source_hashes
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from calibrate_publication import calibrate

SEEDS = (20261110, 20261111, 20261112)
COUNTERS = ('expert_executions', 'external_events', 'source_occurrences')


def equal_weights(first, second):
    assert first.keys() == second.keys()
    for key in first: torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def audit(root, evaluations=False, seeds=SEEDS):
    lock = json.loads((root/'protocol_lock.json').read_text())
    assert lock['seeds'] == list(SEEDS)
    assert hashlib.sha256((ROOT/'docs/SHoEbox_Selective_Publication_Protocol.md').read_bytes()).hexdigest() == lock['protocol_sha256']
    assert hashlib.sha256((root/'parent.pt').read_bytes()).hexdigest() == lock['parent_sha256']
    runs, pairs, group_count = {}, [], 0
    for seed in seeds:
        for policy in ('learned', 'always'):
            name = f'{policy}_{seed}'; path = root/name
            initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=True)
            final = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
            best = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            protocol = final['protocol']; history = final['history']
            assert protocol == initial['protocol'] == best['protocol']
            assert initial['step'] == 0 and not initial['optimizer']['state']
            assert protocol['source_hashes'] == source_hashes()
            assert protocol['parent_sha256'] == lock['parent_sha256']
            assert protocol['steps'] == final['step'] == len(history) == lock['steps']
            assert protocol['seed'] == seed and protocol['policy'] == policy
            equal_weights(initial['weights'], build_producer(root/'parent.pt', seed).state_dict())
            totals = dict.fromkeys(COUNTERS, 0); hashes = []
            choices = []
            for i, row in enumerate(history, 1):
                assert row['step'] == i
                kind = 'recall' if i % 2 else 'correction'
                gap = (0, 4, 8, 16)[((i-1)//2)%4]
                assert (row['kind'], row['gap']) == (kind, gap)
                batch = make_batch('train', (i-1)*protocol['batch'], protocol['batch'], seed, gap, kind)
                assert batch_digest(batch) == row['batch_sha256']
                hashes.append(row['batch_sha256'])
                events = batch.inputs.numel()
                assert row['external_events'] == row['source_occurrences'] == events
                assert row['expert_executions'] == events*12
                for key in COUNTERS: totals[key] += row[key]
                if 'validation' in row:
                    scores = row['validation']
                    key = (min(s['accuracy'] for s in scores), sum(s['accuracy'] for s in scores)/3,
                           -sum(s['nats'] for s in scores)/3)
                    choices.append((key, -i))
            chosen = max(choices)
            assert best['step'] == -chosen[1]
            assert best['best'] == list(chosen[0]) == final['best']
            assert all(final['totals'][k] == totals[k] for k in COUNTERS)
            assert final['totals']['training_seconds'] == sum(r['training_seconds'] for r in history)
            assert history == json.loads((path/'history.json').read_text())
            model, _, _ = load(path/'best.pt')
            key, scores = validation(model, policy, seed, protocol['validation_count'])
            assert list(key) == best['best']
            assert scores == best['history'][-1]['validation']
            runs[name] = dict(selected_step=best['step'], validation_key=list(key), **totals,
                              training_seconds=final['totals']['training_seconds'],
                              parameters=model.parameter_summary(), training_hashes=hashes)
            if evaluations:
                records = json.loads((path/'evaluation.json').read_text())
                assert records['complete']
                expected_count = 24 if policy == 'learned' else 12
                assert len(records['groups']) == expected_count
                for checkpoint in ('best', 'last'):
                    assert records[checkpoint]['sha256'] == hashlib.sha256((path/(checkpoint+'.pt')).read_bytes()).hexdigest()
                    candidate, _, _ = load(path/(checkpoint+'.pt')); candidate.eval()
                    for label, record in records['groups'].items():
                        stage, runtime, gap, kind, control = label.split('/')
                        if stage != checkpoint: continue
                        assert record['count'] == lock['test_episodes'] and record['split'] == 'test'
                        recomputed = evaluate(candidate, runtime, 'test', seed, record['count'], int(gap), kind, control)
                        assert record == recomputed, label
                        assert record['accuracy'] == sum(a == b for a, b in zip(record['predictions'], record['targets']))/record['count']
                        assert record['hold_pairs'] == record['exact_hold_pairs']
                        group_count += 1
                exported = torch.load(path/'inference.pt', map_location='cpu', weights_only=True)
                assert exported['publication_policy'] == policy and exported['selected_step'] == best['step']
                inference = RegionLanguageModel(GraphConfig.from_dict(exported['config']))
                inference.load_state_dict(exported['weights'])
                equal_weights(exported['weights'], best['weights'])
                batch = make_batch('test', 10000, 8, seed)
                with torch.no_grad():
                    a = forward_batch(model, batch, policy)
                    b = forward_batch(inference, batch, policy)
                torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
                for x, y in zip(a['state'].tensors(), b['state'].tensors()):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                if policy == 'learned':
                    calibration, calibrated = calibrate(path, write=False)
                    assert calibration == json.loads((path/'calibration.json').read_text())
                    exported = torch.load(path/'calibrated_inference.pt',map_location='cpu',weights_only=True)
                    equal_weights(exported['weights'],calibrated.state_dict())
                    for parameter,value in best['weights'].items():
                        expected = value-calibration['selected']['offset'] if parameter == 'nodes.A.publication_gate.bias' else value
                        torch.testing.assert_close(exported['weights'][parameter],expected,rtol=0,atol=0)
                    assert records['calibrated']['offset'] == calibration['selected']['offset']
                    assert records['calibrated']['sha256'] == hashlib.sha256((path/'calibrated_inference.pt').read_bytes()).hexdigest()
                    for label, record in records['groups'].items():
                        stage,runtime,gap,kind,control = label.split('/')
                        if stage != 'calibrated': continue
                        assert runtime == 'learned' and record['count'] == lock['test_episodes']
                        assert record == evaluate(calibrated,runtime,'test',seed,record['count'],int(gap),kind,control)
                        assert record['hold_pairs'] == record['exact_hold_pairs']
                        group_count += 1
            print(json.dumps({'audited': name, 'selected_step': best['step'], 'evaluations': evaluations}), flush=True)
        first, second = [runs[f'{policy}_{seed}'] for policy in ('learned', 'always')]
        equal_weights(torch.load(root/f'learned_{seed}'/'initial.pt', weights_only=True)['weights'],
                      torch.load(root/f'always_{seed}'/'initial.pt', weights_only=True)['weights'])
        assert first['training_hashes'] == second['training_hashes']
        assert all(first[key] == second[key] for key in COUNTERS)
        pairs.append({'seed': seed, 'initial_weights_equal': True, 'examples_equal': True, 'budgets_equal': True})
    for run in runs.values(): run.pop('training_hashes')
    result = dict(all_passed=True, evaluations_recomputed=evaluations, prediction_groups=group_count, runs=runs, pairs=pairs)
    suffix = f'_{seeds[0]}' if len(seeds) == 1 else ''
    destination = root/(('evaluation_audit' if evaluations else 'training_audit')+suffix+'.json')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/selective_publication')
    parser.add_argument('--evaluations', action='store_true')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.seed:
        result = audit(args.root,args.evaluations,(args.seed,))
    elif args.workers == 1:
        result = audit(args.root,args.evaluations)
    else:
        if args.workers < 1: parser.error('Worker count must be positive.')
        prefix = 'evaluation_audit' if args.evaluations else 'training_audit'
        def run(seed):
            command=[sys.executable,__file__,'--root',str(args.root),'--seed',str(seed)]
            if args.evaluations: command.append('--evaluations')
            with (args.root/f'{prefix}_{seed}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            return json.loads((args.root/f'{prefix}_{seed}.json').read_text())
        with ThreadPoolExecutor(max_workers=min(args.workers,3)) as pool:
            parts=list(pool.map(run,SEEDS))
        result=dict(all_passed=all(p['all_passed'] for p in parts),evaluations_recomputed=args.evaluations,
                    prediction_groups=sum(p['prediction_groups'] for p in parts),
                    runs={k:v for p in parts for k,v in p['runs'].items()},
                    pairs=[pair for p in parts for pair in p['pairs']])
        (args.root/(prefix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_passed': result['all_passed'], 'prediction_groups': result['prediction_groups']}))


if __name__ == '__main__': main()
