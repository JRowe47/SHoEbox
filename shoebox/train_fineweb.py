"""One-pass cortical byte training with exact budgets and resumable state.

Run with: python -m shoebox.train_fineweb --data data/cortical_fineweb \
  --out artifacts/cortical_fineweb_10m
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
import torch
from . import CorticalModel, SkeletonConfig, cortical_10m
from .checkpoint import save, load
from .config import REGIONS


def source_digest():
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root/'shoebox').glob('*.py')),
             *sorted((root/'experiments/hat_minkron').glob('*.py'))]
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def load_stream(path, batch, targets, checksum):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError('Corpus checksum mismatch.')
    if targets < 1 or batch < 1 or targets % batch or len(raw) != targets+batch:
        raise ValueError('Need exactly targets+batch bytes and a divisible target budget.')
    return torch.tensor(list(raw), dtype=torch.long).reshape(batch, targets//batch+1)


def optimizer_for(model, lr, weight_decay):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim < 2 or getattr(parameter, '_no_weight_decay', False)
         else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                              {'params': no_decay, 'weight_decay': 0.}],
                             lr=lr, betas=(.9, .95), foreach=True)


@torch.no_grad()
def evaluate(model, stream, *, context=8, memory_enabled=None):
    training = model.training
    model.eval()
    state, nll, correct, count = None, 0., 0, 0
    start = time.perf_counter()
    for cursor in range(0, stream.shape[1]-1, context):
        length = min(context, stream.shape[1]-1-cursor)
        x, y = stream[:, cursor:cursor+length], stream[:, cursor+1:cursor+length+1]
        out = model(x, targets=y, state=state, memory_enabled=memory_enabled)
        state = out['state'].detach()
        nll += float(out['ce'])*y.numel()
        correct += int((out['logits'].argmax(-1) == y).sum())
        count += y.numel()
    model.train(training)
    nll /= count
    return {'nll': nll, 'bits_per_byte': nll/math.log(2), 'accuracy': correct/count,
            'targets': count, 'seconds': time.perf_counter()-start}


@torch.no_grad()
def sample(model, prompt, *, count=96, temperature=.8, seed=731):
    """External autoregressive text loop; no internal simulation mechanism."""
    training = model.training
    model.eval()
    prefix = prompt.encode('utf-8')
    if not prefix or count < 1 or temperature <= 0:
        raise ValueError('Need a nonempty prompt, positive length and temperature.')
    result = model(torch.tensor([list(prefix)], dtype=torch.long))
    state, logits = result['state'], result['logits'][:, -1]
    rng = torch.Generator().manual_seed(seed)
    generated = []
    for i in range(count):
        token = torch.multinomial((logits/temperature).softmax(-1), 1, generator=rng)
        generated.append(int(token))
        if i+1 < count:
            result = model(token, state=state, continuation=token)
            state, logits = result['state'], result['logits'][:, -1]
    model.train(training)
    return {'prompt': prompt, 'generated': bytes(generated).decode('utf-8', errors='replace'),
            'generated_hex': bytes(generated).hex(), 'temperature': temperature, 'seed': seed}


def task_coverage(model, loss):
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    groups = {name: 0. for name in model.regions}
    hippocampus = 0.
    for (name, _), gradient in zip(named, gradients):
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('Nonfinite pure task gradient in '+name)
        magnitude = float(gradient.abs().sum())
        if name.startswith('regions.'):
            groups[name.split('.')[1]] += magnitude
        elif name.startswith('hippocampus.'):
            hippocampus += magnitude
    return {'regions': groups, 'hippocampus': hippocampus,
            'all_regions_nonzero': all(value > 0 for value in groups.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--preset', choices=('10m', 'skeleton'), default='10m')
    p.add_argument('--context', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--warmup-updates', type=int, default=10)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--checkpoint-every', type=int, default=50)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--max-updates', type=int, help='Stop early with a resumable checkpoint; budget remains unchanged.')
    p.add_argument('--sample-bytes', type=int, default=96)
    args = p.parse_args()
    if args.out.exists():
        p.error('Choose a new output directory; resume writes to a new directory.')
    if min(args.context, args.threads, args.checkpoint_every, args.log_every) < 1:
        p.error('Context, threads and intervals must be positive.')
    if args.lr <= 0 or args.warmup_updates < 0 or args.weight_decay < 0 or args.sample_bytes < 0:
        p.error('Invalid optimizer or sampling setting.')
    if args.max_updates is not None and args.max_updates < 1:
        p.error('max-updates must be positive.')
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.data/'manifest.json').read_text())
    batch, budget = manifest['batch'], manifest['train_targets']
    train = load_stream(args.data/'train.bin', batch, budget, manifest['files']['train']['sha256'])
    validation = load_stream(args.data/'validation.bin', manifest['validation_batch'],
                             manifest['validation_targets'], manifest['files']['validation']['sha256'])
    protocol = {'batch': batch, 'context': args.context, 'budget': budget,
                'train_sha256': manifest['files']['train']['sha256'],
                'validation_sha256': manifest['files']['validation']['sha256'],
                'lr': args.lr, 'warmup_updates': args.warmup_updates,
                'weight_decay': args.weight_decay, 'threads': args.threads,
                'source_sha256': source_digest()}
    coverage, history, validations = None, [], []
    cursor, step, training_seconds, initial = 0, 0, 0., None
    if args.resume:
        model, state, _, checkpoint = load(args.resume)
        old = checkpoint['metadata']
        if old.get('protocol') != protocol:
            raise ValueError('Resume protocol/source differs; use the original arguments and source.')
        optimizer = optimizer_for(model, args.lr, args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'])
        cursor, step = old['cursor'], checkpoint['step']
        training_seconds, initial = old['training_seconds'], old['initial']
        coverage, history, validations = old['task_gradient_coverage'], old['history'], old['validation']
        if cursor*batch != old['trained_targets'] or state.event != cursor:
            raise ValueError('Checkpoint cursor, byte count and state disagree.')
    else:
        cfg = cortical_10m() if args.preset == '10m' else SkeletonConfig()
        torch.manual_seed(cfg.seed)
        model = CorticalModel(cfg)
        optimizer = optimizer_for(model, args.lr, args.weight_decay)
        state = model.initial_state(batch)
    parameters = sum(p.numel() for p in model.parameters())
    args.out.mkdir(parents=True)
    atomic_json(args.out/'config.json', model.cfg.to_dict())
    atomic_json(args.out/'data_manifest.json', manifest)
    atomic_json(args.out/'topology.json', [asdict(spec) for spec in REGIONS])
    if initial is None:
        initial = evaluate(model, validation)
        validations.append({'trained_targets': 0, **initial})
    print(json.dumps({'event': 'start', 'parameters': parameters, 'trained_targets': cursor*batch,
                      'budget': budget, 'initial_validation': initial}), flush=True)
    start_step = step
    started = time.perf_counter()

    def snapshot(complete):
        metadata = {'protocol': protocol, 'cursor': cursor, 'trained_targets': cursor*batch,
                    'training_seconds': training_seconds, 'initial': initial,
                    'task_gradient_coverage': coverage, 'history': history, 'validation': validations}
        save(args.out/'last.pt', model, state, optimizer, step=step, metadata=metadata)
        summary = {'format': 'shoebox-cortical-fineweb-v1', 'complete': complete,
                   'parameters': parameters, 'config': model.cfg.to_dict(), 'protocol': protocol,
                   'trained_targets': cursor*batch, 'optimizer_updates': step,
                   'training_seconds': training_seconds,
                   'training_bytes_per_second': cursor*batch/training_seconds if training_seconds else 0.,
                   'runtime_state_bytes': state.storage_bytes(),
                   'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   'initial_validation': initial, 'validation': validations,
                   'task_gradient_coverage': coverage, 'history': history,
                   'torch': str(torch.__version__), 'python': platform.python_version(),
                   'session_wall_seconds': time.perf_counter()-started}
        atomic_json(args.out/'results.json', summary)
        return summary

    model.train()
    while cursor < train.shape[1]-1:
        if args.max_updates is not None and step-start_step >= args.max_updates:
            break
        tick = time.perf_counter()
        length = min(args.context, train.shape[1]-1-cursor)
        x, y = train[:, cursor:cursor+length], train[:, cursor+1:cursor+length+1]
        lr = args.lr*min(1., (step+1)/max(1, args.warmup_updates))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        result = model(x, targets=y, state=state)
        if not bool(torch.isfinite(result['loss'])):
            raise RuntimeError('Nonfinite training loss.')
        if coverage is None:
            coverage = task_coverage(model, result['ce'])
            if not coverage['all_regions_nonzero'] or coverage['hippocampus'] <= 0:
                raise RuntimeError('Pure task loss did not reach all regions and hippocampus.')
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
        optimizer.step()
        state = result['state'].detach()
        step += 1
        cursor += length
        duration = time.perf_counter()-tick
        training_seconds += duration
        record = {'step': step, 'trained_targets': cursor*batch, 'targets': y.numel(),
                  'nll': float(result['ce'].detach()), 'aux': float(result['aux'].detach()),
                  'accuracy': float((result['logits'].detach().argmax(-1) == y).float().mean()),
                  'gradient_norm': float(norm), 'lr': lr, 'seconds': duration}
        history.append(record)
        # Release the large event graph before validation or checkpoint I/O.
        del result
        with (args.out/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if step % args.log_every == 0 or step == 1 or cursor*batch == budget:
            recent = history[-args.log_every:]
            print(json.dumps({'event': 'progress', **record,
                              'recent_nll': sum(r['nll']*r['targets'] for r in recent)/sum(r['targets'] for r in recent),
                              'bytes_per_second': cursor*batch/training_seconds}), flush=True)
        if step % args.checkpoint_every == 0 or cursor*batch == budget:
            snapshot(False)
            score = evaluate(model, validation)
            validations.append({'trained_targets': cursor*batch, **score})
            snapshot(cursor*batch == budget)
            print(json.dumps({'event': 'validation', 'trained_targets': cursor*batch, **score}), flush=True)
    complete = cursor*batch == budget
    summary = snapshot(complete)
    if complete:
        final = validations[-1]
        if final['trained_targets'] != budget:
            raise RuntimeError('Final validation is missing.')
        # A fixed prefix of each training stream allows initial/final-style
        # interpretation without presenting moving online losses as a holdout.
        summary['final_train_probe'] = evaluate(model, train[:, :min(17, train.shape[1])])
        summary['final_validation'] = final
        if args.sample_bytes:
            samples = [sample(model, prompt, count=args.sample_bytes, seed=731+i)
                       for i, prompt in enumerate(('The ', 'In the '))]
            atomic_json(args.out/'samples.json', samples)
            summary['samples'] = samples
        summary['session_wall_seconds'] = time.perf_counter()-started
        atomic_json(args.out/'results.json', summary)
    print(json.dumps({'event': 'complete' if complete else 'paused', 'trained_targets': cursor*batch,
                      'parameters': parameters, 'training_seconds': training_seconds}), flush=True)


if __name__ == '__main__':
    main()
