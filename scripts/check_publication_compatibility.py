"""Compare the original seven-node checkpoint in two source checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def snapshot(source, parent, out):
    sys.path.insert(0, str(source))
    import torch
    from experiments.text_regions.integrated_train import load_integrated
    from experiments.text_regions.composition import evaluate, encode_sources, answer_query
    from experiments.text_regions.composition_data import evaluation_keys, batch_from_keys
    torch.set_num_threads(1)
    model, *_ = load_integrated(parent)
    model.eval()
    evidence = {'parameters': sum(p.numel() for p in model.parameters()), 'batches': []}
    def digest(tensors):
        h = hashlib.sha256()
        for value in tensors:
            h.update(str((tuple(value.shape), value.dtype)).encode())
            h.update(value.detach().contiguous().numpy().tobytes())
        return h.hexdigest()
    with torch.no_grad():
        evidence['composition'] = evaluate(model, split='test', count=None, rounds=(1, 2, 4), controls=True)
        keys = evaluation_keys('test', None)
        for start in range(0, len(keys), 64):
            batch = batch_from_keys(keys[start:start+64])
            prefix = encode_sources(model, batch)
            result = answer_query(model, batch, prefix['state'], 2)
            evidence['batches'].append({'logits': digest([result['logits']]),
                'state': digest(result['state'].tensors()), 'prefix_state': digest(prefix['state'].tensors()),
                'expert_executions': prefix['expert_executions']+result['expert_executions']})
    out.write_text(json.dumps(evidence, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--parent', type=Path, default=ROOT/'artifacts/selective_publication/parent.pt')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/selective_publication/compatibility.json')
    args = p.parse_args()
    if args.snapshot:
        snapshot(args.snapshot.resolve(), args.parent.resolve(), args.out.resolve()); return
    if not args.previous: p.error('--previous must identify the parent source checkout.')
    paths = [args.out.with_name('previous_snapshot.json'), args.out.with_name('current_snapshot.json')]
    for source, path in zip((args.previous.resolve(), ROOT), paths):
        subprocess.run([sys.executable, __file__, '--snapshot', str(source), '--parent', str(args.parent),
                        '--out', str(path)], check=True)
    old, new = [json.loads(path.read_text()) for path in paths]
    if old != new: raise AssertionError('Default behavior changed.')
    result = dict(all_passed=True, exact_logits_states_costs=True,
                  previous_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.previous).decode().strip(),
                  parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                  composition={name: {'accuracy': record['accuracy'], 'episodes': record['episodes']}
                               for name, record in new['composition'].items()}, batches=len(new['batches']))
    args.out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__': main()
