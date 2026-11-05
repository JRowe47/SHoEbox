"""Score a frozen composition checkpoint from identical independent source states."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .composition import evaluate
from .composition_data import coverage
from .composition_train import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=('validation','test','all'), default='test')
    p.add_argument('--count', type=int)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--rounds', type=int, nargs='+', default=[1,2,4])
    p.add_argument('--controls', action='store_true')
    args = p.parse_args()
    if min(args.batch,args.threads,*args.rounds)<1:
        p.error('Counts and rounds must be positive.')
    torch.set_num_threads(args.threads)
    model, _, _, saved = load_checkpoint(args.checkpoint)
    start = time.perf_counter()
    metrics = evaluate(model,args.split,args.count,args.batch,tuple(args.rounds),args.controls)
    result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  checkpoint_step=saved['step'],config=model.cfg.to_dict(),protocol=saved['protocol'],
                  parameters=model.parameter_summary(),split=args.split,coverage=coverage(),
                  scoring='argmax over all 256 byte logits',controls=metrics,
                  same_prefix='Each K starts from the same functional source state; query bytes ingested once.',
                  seconds=time.perf_counter()-start)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ('episodes','accuracy','nats_per_answer','expert_executions')}
                      for k,v in metrics.items()},indent=2))


if __name__ == '__main__':
    main()
