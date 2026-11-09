"""Reload a finished hierarchy endpoint and reproduce its complete held-out score."""
import argparse
import json
from pathlib import Path
import torch
from shoebox.checkpoint import load
from shoebox.train_hierarchy import evaluate, load_data
from shoebox.train_fineweb import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--data', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    result = json.loads((args.run/'results.json').read_text())
    if not result['complete']:
        raise ValueError('Audit requires a completed run.')
    model, state, _, checkpoint = load(args.run/'last.pt')
    model.set_execution_backend('cpu')
    if checkpoint['metadata']['protocol'] != result['protocol'] or checkpoint['step'] != result['updates']:
        raise ValueError('Checkpoint/result protocol mismatch.')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite checkpoint parameter.')
    finite_state = []
    state._map(lambda value: finite_state.append(bool(torch.isfinite(value).all())) or value)
    if not all(finite_state):
        raise ValueError('Nonfinite checkpoint state.')
    _, docs, manifest = load_data(args.data)
    if manifest['files']['validation']['sha256'] != result['protocol']['validation_sha256']:
        raise ValueError('Validation corpus mismatch.')
    context = result['protocol'].get('eval_context', result['protocol']['context'])
    score = evaluate(model, docs, context=context, span_metrics=True)
    expected = result['evaluation'][-1]
    for field in ('nll', 'accuracy', 'targets', 'span_nll', 'span_targets'):
        if score[field] != expected[field]:
            raise ValueError(f'Checkpoint does not exactly reproduce {field}.')
    monitoring = evaluate(model, docs, context=context, limit=8192, span_metrics=True)
    audit = dict(exact_score_reproduction=True, finite_parameters=True, finite_state=True,
                 checkpoint_step=checkpoint['step'], evaluation=score,
                 monitoring_endpoint=dict(updates=result['updates'], full=False, **monitoring))
    atomic_json(args.run/'checkpoint_audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
