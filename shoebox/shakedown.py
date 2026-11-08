"""Small deterministic learning/streaming run, plus portable checkpoints.

The default fixed byte fixture measures optimization plumbing, not language
quality. --data instead advances independent contiguous byte streams, carrying
ordinary recurrent state across truncated-backprop windows.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
from .config import SkeletonConfig, REGIONS
from .model import CorticalModel
from .checkpoint import save, load


FIXTURE = b'red blue red.\nblue red blue.\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/cortical_skeleton'))
    parser.add_argument('--steps', type=int, default=24, help='Additional optimizer steps.')
    parser.add_argument('--context', type=int, default=4)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--data', type=Path, help='Optional ordinary UTF-8/raw-byte corpus.')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--seed', type=int, default=20261108)
    args = parser.parse_args(argv)
    if min(args.context, args.batch, args.threads, args.rounds) < 1 or args.steps < 0 or args.lr <= 0:
        parser.error('Dimensions, threads and LR must be positive; steps cannot be negative.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    raw = args.data.read_bytes() if args.data else FIXTURE
    if len(raw) < 2:
        parser.error('Corpus must contain at least two bytes.')
    digest = hashlib.sha256(raw).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'last.pt').exists():
        parser.error('Choose a new output directory; existing checkpoints are preserved.')
    optimizer_factory = lambda p: torch.optim.AdamW(p, lr=args.lr, weight_decay=.01)
    cursor = 0
    if args.resume:
        model, state, optimizer, checkpoint = load(args.resume, optimizer_factory=optimizer_factory)
        start_step = checkpoint['step']
        metadata = checkpoint['metadata']
        expected = (digest, args.batch, args.context, bool(args.data))
        previous = (metadata.get('data_sha256'), metadata.get('batch'), metadata.get('context'), metadata.get('streaming'))
        if expected != previous:
            parser.error('Resume requires identical data, batch, context and streaming mode.')
        if args.rounds != model.cfg.rounds:
            parser.error('Resume rounds must match the checkpoint.')
        cursor = metadata.get('cursor', 0)
    else:
        model = CorticalModel(SkeletonConfig(rounds=args.rounds, seed=args.seed))
        optimizer = optimizer_factory(model.parameters())
        state, start_step = model.initial_state(args.batch), 0
    if args.data:
        row_length = len(raw)//args.batch
        if row_length < args.context+1:
            parser.error('Corpus is too short for batch * (context+1) bytes.')
        data = torch.tensor(list(raw[:row_length*args.batch]), dtype=torch.long).reshape(args.batch, row_length)
        fixture = data[:, :args.context+1]
    else:
        repeated = raw*(1+(args.context+1+7*args.batch)//len(raw))
        fixture = torch.tensor([list(repeated[7*b:7*b+args.context+1]) for b in range(args.batch)], dtype=torch.long)
        data = fixture
    metadata = {'data_sha256': digest, 'batch': args.batch, 'context': args.context,
                'streaming': bool(args.data), 'cursor': cursor}
    save(args.out/'initial.pt', model, state, optimizer, step=start_step, metadata=metadata)

    @torch.no_grad()
    def evaluate():
        model.eval()
        result = model(fixture[:, :-1], targets=fixture[:, 1:])
        value = {'nll': float(result['ce']), 'accuracy': float((result['logits'].argmax(-1) == fixture[:, 1:]).float().mean())}
        model.train()
        return value

    initial = evaluate()
    history, grad_coverage = [], {}
    begin = time.perf_counter()
    for offset in range(args.steps):
        if args.data:
            if cursor+args.context+1 > data.shape[1]:
                cursor, state = 0, model.initial_state(args.batch)
            segment = data[:, cursor:cursor+args.context+1]
            cursor += args.context
        else:
            segment, state = fixture, model.initial_state(args.batch)
        optimizer.zero_grad(set_to_none=True)
        result = model(segment[:, :-1], targets=segment[:, 1:], state=state)
        if not bool(torch.isfinite(result['loss'])):
            raise RuntimeError('Nonfinite loss.')
        result['loss'].backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError('Nonfinite gradient.')
        if offset == 0:
            grad_coverage = {name: sum(float(p.grad.abs().sum()) for p in region.parameters() if p.grad is not None)
                             for name, region in model.regions.items()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        state = result['state'].detach()
        record = {'step': start_step+offset+1, 'nll': float(result['ce'].detach()), 'gradient_norm': float(norm)}
        history.append(record)
        if offset == 0 or (offset+1) % 8 == 0 or offset+1 == args.steps:
            print(json.dumps(record), flush=True)
    elapsed = time.perf_counter()-begin
    final = evaluate()
    metadata['cursor'] = cursor
    save(args.out/'last.pt', model, state, optimizer, step=start_step+args.steps, metadata=metadata)
    with torch.no_grad():
        sample = model(fixture[:, :2], trace=True)
    (args.out/'trace.json').write_text(json.dumps(sample['trace'], indent=2)+'\n')
    (args.out/'topology.json').write_text(json.dumps([asdict(r) for r in REGIONS], indent=2)+'\n')
    params = sum(p.numel() for p in model.parameters())
    total_bytes = args.steps*args.context*args.batch
    summary = {'format': 'cortical-skeleton-shakedown-v1', 'config': model.cfg.to_dict(),
               'parameters': params, 'regions': len(REGIONS), 'experts': len(REGIONS)*model.cfg.experts,
               'initial': initial, 'final': final, 'steps': args.steps, 'start_step': start_step,
               'batch': args.batch, 'context': args.context, 'streaming': bool(args.data),
               'data_sha256': digest, 'training_seconds': elapsed,
               'training_bytes_per_second': total_bytes/elapsed if elapsed else 0,
               'runtime_state_bytes': state.storage_bytes(),
               'region_gradient_sums_first_step': grad_coverage,
               'all_region_gradients_nonzero': bool(grad_coverage) and all(v > 0 for v in grad_coverage.values()),
               'history': history, 'torch': torch.__version__, 'python': platform.python_version(),
               'threads': args.threads}
    (args.out/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ('parameters', 'initial', 'final', 'training_seconds', 'training_bytes_per_second')}), flush=True)


if __name__ == '__main__':
    main()
