"""Measure each task gradient on the next recorded training segment, without an update."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.integrated_train import load_integrated, stream_loss
from experiments.text_regions.composition import training_loss
from experiments.text_regions.composition_data import training_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _, rng, (streams, states), saved = load_integrated(args.checkpoint)
    rows = {}
    for name in ('composition', 'correction', 'views'):
        model.zero_grad(set_to_none=True)
        if name == 'composition':
            batch = training_batch(rng, saved['protocol']['composition_batch'])
            loss, *_ = training_loss(model, batch, 2, True, .5, 1.)
        else:
            chunk = streams[name].next_chunk(saved['protocol']['segment'])
            loss, _ = stream_loss(model, chunk, states[name])
        loss.backward()
        norms = {key: float(parameter.grad.norm()) for key, parameter in model.named_parameters()
                 if parameter.grad is not None}
        rows[name] = {'loss': float(loss.detach()), 'norm': sum(v*v for v in norms.values())**.5,
                      'largest': sorted(norms.items(), key=lambda x: x[1], reverse=True)[:5]}
        print(name, json.dumps(rows[name]), flush=True)
    result = {'checkpoint_step': saved['step'],
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), 'tasks': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
