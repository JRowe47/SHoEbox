"""CPU streaming trainer with full-state, optimizer, cursor and RNG checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import time
import torch
from ..hat_minkron.train import optimizer_for
from .config import BASELINE_REVISION, GraphConfig
from .data import OrderedByteStreams, split_bytes
from .model import RegionLanguageModel
from .state import GraphState

CHECKPOINT_VERSION = 1
FIXTURE = b'ABCD\n'*1024


def learning_rate(step, total_steps, base_lr):
    warmup = min(40, max(1, total_steps//10))
    scale = min(1, step/warmup)
    if step > .8*total_steps:
        scale *= 1-.8*(step-.8*total_steps)/(.2*total_steps)
    return base_lr*scale


def save_checkpoint(path, model, optimizer, state, streams, step, protocol):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save({'schema_version': CHECKPOINT_VERSION, 'config': model.cfg.to_dict(),
                'weights': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'graph_state': state.to_dict(), 'streams': streams.state_dict(), 'step': step,
                'rng': {'torch': torch.get_rng_state(), 'python': random.getstate()}, 'protocol': protocol}, temporary)
    temporary.replace(path)


def load_checkpoint(path, streams=None, restore_rng=True):
    # Only tensors, primitives, lists, tuples and dicts are serialized.
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['schema_version'] != CHECKPOINT_VERSION:
        raise ValueError('Unsupported checkpoint version.')
    model = RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights'])
    optimizer = optimizer_for(model, saved['protocol']['lr'])
    optimizer.load_state_dict(saved['optimizer'])
    if streams is not None:
        streams.load_state_dict(saved['streams'])
    if restore_rng:
        torch.set_rng_state(saved['rng']['torch'])
        random.setstate(saved['rng']['python'])
    return model, optimizer, GraphState.from_dict(saved['graph_state']), saved


def optimizer_bytes(optimizer):
    return sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if torch.is_tensor(t))


@torch.no_grad()
def evaluate(model, raw, batch_size=4, context=128, chunks=2):
    was_training = model.training
    # Validation owns fresh state/cursors and an isolated RNG context.
    streams = OrderedByteStreams.from_bytes(raw, batch_size)
    state, losses = None, []
    try:
        model.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(333)
            for _ in range(chunks):
                chunk = streams.next_chunk(context)
                result = model(chunk.values, chunk.targets, state=state, reset=chunk.reset)
                state = result['state']
                losses.append(float(result['ce']))
        return sum(losses)/len(losses)
    finally:
        model.train(was_training)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path, default=Path('artifacts/text_regions'))
    p.add_argument('--steps', type=int, default=20, help='Total schedule horizon, retained on resume.')
    p.add_argument('--stop-after', type=int, help='Checkpoint early without changing the schedule horizon.')
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--seed', type=int, default=20261104)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--recall-slots', type=int, default=0)
    p.add_argument('--recall-nodes', nargs='*', default=[])
    p.add_argument('--recall-layout', choices=('distinct', 'pooled'), default='distinct')
    p.add_argument('--recall-feedback', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--recall-hard', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--tiny', action='store_true', help='Reduced-capacity engineering fixture only.')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if min(args.steps, args.batch, args.context, args.rounds, args.threads, args.eval_every) < 1 or args.lr <= 0:
        p.error('Steps, batch, context, rounds, threads, eval-every and lr must be positive.')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        p.error('--stop-after must be between 1 and --steps.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    raw = args.data.read_bytes() if args.data else FIXTURE
    train, val, test = split_bytes(raw)
    streams = OrderedByteStreams.from_bytes(train, args.batch)
    protocol = {'baseline_revision': BASELINE_REVISION, 'steps': args.steps, 'lr': args.lr,
                'batch': args.batch, 'context': args.context, 'seed': args.seed,
                'data_sha256': hashlib.sha256(raw).hexdigest(), 'data_bytes': len(raw),
                'train_bytes': len(train), 'validation_bytes': len(val), 'untouched_test_bytes': len(test),
                'data_kind': 'file' if args.data else 'repeating five-byte engineering fixture',
                'optimizer': 'AdamW (.9,.95), matrix decay .01; inherited exclusions; clip 1',
                'schedule': 'inherited warmup then linear taper over final 20%; horizon retained on resume',
                'state_policy': 'ordered rows, explicit episode boundaries, detach every segment',
                'torch': str(torch.__version__), 'python': platform.python_version(), 'threads': args.threads}
    state, start = None, 0
    if args.resume:
        model, optimizer, state, saved = load_checkpoint(args.resume, streams)
        for key in ('steps', 'lr', 'batch', 'context', 'data_sha256', 'threads'):
            if saved['protocol'][key] != protocol[key]:
                raise ValueError(f'Resume must preserve {key} from the saved protocol.')
        if model.cfg.rounds != args.rounds:
            raise ValueError('Resume must preserve the round count.')
        for key in ('recall_slots', 'recall_layout', 'recall_feedback', 'recall_hard'):
            if getattr(model.cfg, key) != getattr(args, key):
                raise ValueError(f'Resume must preserve {key}.')
        if model.cfg.recall_nodes != tuple(args.recall_nodes):
            raise ValueError('Resume must preserve recall nodes.')
        protocol, start = saved['protocol'], saved['step']
    else:
        factory = GraphConfig.tiny if args.tiny else GraphConfig
        model = RegionLanguageModel(factory(context=args.context, rounds=args.rounds,
                                             recurrent_feedback=args.rounds > 1, seed=args.seed,
                                             recall_slots=args.recall_slots, recall_nodes=tuple(args.recall_nodes),
                                             recall_layout=args.recall_layout, recall_feedback=args.recall_feedback,
                                             recall_hard=args.recall_hard))
        optimizer = optimizer_for(model, args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'protocol.json').write_text(json.dumps({**protocol, 'config': model.cfg.to_dict(),
                                                   'parameters': model.parameter_summary()}, indent=2))
    initial = evaluate(model, val, args.batch, args.context, chunks=1)
    print(json.dumps({'initial_validation_nats_per_byte': initial, **model.parameter_summary()}), flush=True)
    seconds, executed, seen, recent = 0., 0, 0, []
    stop = args.stop_after or args.steps
    if stop <= start:
        raise ValueError('Checkpoint already reached the requested stopping step.')
    with (args.out/'history.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(start+1, stop+1):
            model.train()
            chunk = streams.next_chunk(args.context)
            lr = learning_rate(step, args.steps, args.lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result = model(chunk.values, chunk.targets, state=state, reset=chunk.reset)
            if not bool(torch.isfinite(result['loss'])):
                raise RuntimeError('Nonfinite training loss.')
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('Nonfinite gradient.')
            optimizer.step()
            state = result['state'].detach()
            seconds += time.perf_counter()-tick
            executed += result['expert_executions']
            seen += result['external_events']
            recent.append(float(result['ce'].detach()))
            if step % args.eval_every == 0 or step == stop:
                validation = evaluate(model, val, args.batch, args.context, chunks=1)
                row = {'step': step, 'external_bytes_seen_total': step*args.batch*args.context,
                       'training_nats_per_byte_recent': sum(recent)/len(recent),
                       'validation_nats_per_byte': validation, 'validation_bits_per_byte': validation/math.log(2),
                       'training_seconds_this_invocation': seconds, 'gradient_norm': float(norm), 'lr': lr,
                       'state_bytes': state.storage_bytes(), 'optimizer_state_bytes': optimizer_bytes(optimizer),
                       'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
                recent = []
            # Release the previous segment's autograd roots before the next forward.
            del result
    save_checkpoint(args.out/'checkpoint.pt', model, optimizer, state, streams, stop, protocol)
    generated, _ = model.generate('AB' if not args.data else 'ROMEO:\n', count=40)
    sample = bytes(generated[0].tolist())
    (args.out/'sample.bin').write_bytes(sample)
    (args.out/'sample.txt').write_text(sample.decode('utf-8', errors='replace'))
    summary = {'schema_version': 1, 'initial_validation_nats_per_byte': initial,
               'final_validation_nats_per_byte': validation, 'external_bytes_this_invocation': seen,
               'expert_executions_this_invocation': executed, 'training_seconds_this_invocation': seconds,
               'bytes_per_second': seen/seconds, 'state_bytes': state.storage_bytes(),
               'optimizer_state_bytes': optimizer_bytes(optimizer),
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               'final_step': stop, 'parameters': model.parameter_summary()}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
