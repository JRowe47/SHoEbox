"""Audit joint artifacts and export one inference-only graph for all tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.composition import encode_sources, answer_query
from experiments.text_regions.composition_data import batch_from_keys, evaluation_keys
from experiments.text_regions.composition_train import source_hashes, atomic_save
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.episode_data import make_batch
from experiments.text_regions.integrated_eval import selection_key
from experiments.text_regions.integrated_train import load_integrated
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.view_data import make_view_batch


def assert_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else: assert a == b


@torch.no_grad()
def export_inference(path, out):
    model, _, _, _, saved = load_integrated(path)
    model.eval()
    inference = RegionLanguageModel(GraphConfig.from_dict(saved['config'])).eval()
    weights = {k: v for k, v in saved['weights'].items()
               if not k.startswith(('source_reconstruction.', 'regional_readout.'))}
    inference.load_state_dict(weights, strict=True)
    batch = batch_from_keys(evaluation_keys('test')[:64])
    a = answer_query(model, batch, encode_sources(model, batch)['state'])
    b = answer_query(inference, batch, encode_sources(inference, batch)['state'])
    assert torch.equal(a['logits'], b['logits'])
    cases = {'composition': batch.size}
    for name, batch in [('correction', make_batch('correction', 'test', 0, 8, gap=16)),
                        ('views', make_view_batch('test', 0, 8, gap=8))]:
        a_state = b_state = None
        for pos in range(0, batch.length, 16):
            chunk = batch.slice(pos, min(pos+16, batch.length))
            args = dict(reset=chunk.reset, availability=chunk.availability, rounds=2, feedback=True)
            a = model(chunk.inputs, state=a_state, **args)
            b = inference(chunk.inputs, state=b_state, **args)
            assert torch.equal(a['logits'], b['logits'])
            assert_equal(a['state'].to_dict(), b['state'].to_dict())
            a_state, b_state = a['state'], b['state']
        cases[name] = 8
    atomic_save(out, {'schema': 'shoebox-integrated-inference-v1', 'config': saved['config'],
                      'weights': weights, 'parent_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    return {'exact_logits_and_states': True, 'cases': cases,
            'parameters': inference.parameter_summary(),
            'removed_heads': ['source_reconstruction', 'regional_readout'],
            'inference_sha256': hashlib.sha256(Path(out).read_bytes()).hexdigest()}


def audit_training(parent, directory):
    load = lambda name: torch.load(directory/name, map_location='cpu', weights_only=True)
    initial, final, best, candidate = map(load, ('initial.pt', 'last.pt', 'best.pt', 'best_candidate.pt'))
    original = torch.load(parent, map_location='cpu', weights_only=True)
    for key in ('weights', 'optimizer', 'rng', 'torch_rng'): assert_equal(original[key], initial[key])
    assert final['protocol']['source_hashes'] == source_hashes()
    assert final['protocol']['parent_sha256'] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert final['step'] == final['protocol']['steps']
    history = final['history']
    updates = [row for row in history if row['step']]
    assert [row['step'] for row in updates] == list(range(1, final['step']+1))
    checks = [row for row in history if 'validation' in row]
    expected = max(checks, key=lambda row: selection_key(row['validation']))
    expected_candidate = max(checks, key=lambda row: selection_key(row['validation'])[2:])
    assert best['step'] == expected['step'] == final['totals']['best_step']
    assert candidate['step'] == expected_candidate['step'] == final['totals']['candidate_step']
    counter_keys = ('expert_executions', 'source_occurrences', 'external_events', 'composition_episodes',
                    'correction_final_queries', 'views_final_queries', 'correction_episodes_started',
                    'views_episodes_started', 'correction_supervised', 'views_supervised')
    for key in counter_keys: assert sum(row[key] for row in updates) == final['totals'][key]
    for row in updates:
        assert row['composition_episodes'] == final['protocol']['composition_batch']
        assert row['lr'] == final['protocol']['lr']
        assert row['expert_executions'] >= 264*row['composition_episodes']
    return {'parent_weights_optimizer_rng_exact': True, 'source_hashes_match': True,
            'completed_updates': final['step'], 'selected_step': best['step'],
            'candidate_step': candidate['step'], 'selection_recomputed': True,
            'counters_recomputed': {key: final['totals'][key] for key in counter_keys},
            'checkpoint_sha256': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                  for name in ('initial.pt', 'last.pt', 'best.pt', 'best_candidate.pt')},
            'selected_scores': expected['validation']['scores'],
            'selected_failures': expected['validation']['gate_failures'],
            'candidate_scores': expected_candidate['validation']['scores'],
            'candidate_failures': expected_candidate['validation']['gate_failures']}


def audit_predictions(report):
    groups = 0
    for row in report['composition_controls'].values():
        assert len(row['predictions']) == len(row['targets']) == row['episodes']
        correct = sum(a == b for a, b in zip(row['predictions'], row['targets']))
        assert correct == row['correct'] and correct/row['episodes'] == row['accuracy']
        groups += 1
    for cohort in ('heldout_views', 'independent_views'):
        for rows in report[cohort].values():
            for row in rows.values():
                correct = sum(a == b for a, b in zip(row['predictions'], row['targets']))
                assert len(row['predictions']) == len(row['targets']) == row['episodes']
                assert correct == row['exact_correct'] and correct/row['episodes'] == row['exact_accuracy']
                assert len(set(row['episode_ids'])) == row['episodes']
                groups += 1
    for record in report['memory_controls'].values():
        for kind, evaluation in record.items():
            for control, row in evaluation['controls'].items():
                batch = make_batch(kind, evaluation['split'], 0, row['episodes'],
                                   seed=evaluation['seed'], gap=evaluation['gap'],
                                   query_every=evaluation['query_every'], control=control)
                assert list(batch.identities) == row['episode_ids']
                predictions = torch.tensor(row['predictions'])
                correct = int(predictions.eq(batch.answers).sum())
                old_correct = int(predictions.eq(batch.old_answers).sum())
                assert len(predictions) == row['episodes']
                assert correct == row['exact_correct']
                assert correct/row['episodes'] == row['exact_accuracy']
                assert old_correct/row['episodes'] == row['old_value_accuracy']
                groups += 1
            assert len({r['expert_executions'] for r in evaluation['controls'].values()}) == 1
    for kind in ('shared', 'left', 'right'):
        rows = [control[kind] for control in report['independent_views'].values()]
        assert len({row['expert_executions'] for row in rows}) == 1
        assert len({tuple(row['episode_ids']) for row in rows}) == 1
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', default='best.pt', choices=('best.pt', 'best_candidate.pt', 'last.pt'))
    p.add_argument('--evaluation', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    report = audit_training(args.parent, args.run)
    report['inference_export'] = export_inference(args.run/args.checkpoint, args.run/'inference.pt')
    if args.evaluation:
        report['prediction_groups_recomputed'] = audit_predictions(json.loads(args.evaluation.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
