"""Train the regional graph itself on distributed relation composition."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from ..hat_minkron.train import optimizer_for
from .composition import evaluate, training_loss
from .composition_data import coverage, training_batch
from .config import GraphConfig
from .model import RegionLanguageModel


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'text_regions').glob('*.py')) + sorted((root/'hat_minkron').glob('*.py'))
    files = [p for p in files if not p.name.startswith('test_') and p.name != 'verify.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def save_checkpoint(path, model, optimizer, rng, step, protocol, best_validation, counters, history):
    atomic_save(path, dict(schema='shoebox-composition-v1', config=model.cfg.to_dict(),
                          weights=model.state_dict(), optimizer=optimizer.state_dict(),
                          rng=rng.getstate(), torch_rng=torch.get_rng_state(), step=step,
                          protocol=protocol, best_validation=best_validation,
                          counters=counters, history=history))


def load_checkpoint(path, expected_protocol=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['schema'] != 'shoebox-composition-v1':
        raise ValueError('Unknown composition checkpoint schema.')
    if expected_protocol is not None and saved['protocol'] != expected_protocol:
        raise ValueError('Resume requires unchanged source and training protocol.')
    model = RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights'])
    optimizer = optimizer_for(model, saved['protocol']['lr'])
    optimizer.load_state_dict(saved['optimizer'])
    rng = random.Random(); rng.setstate(saved['rng'])
    torch.set_rng_state(saved['torch_rng'])
    return model, optimizer, rng, saved


def update(model, optimizer, batch, step, protocol):
    lr = protocol['lr'] * min(1., (step+1)/protocol['warmup'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    loss, ce, prefix, query = training_loss(model, batch, rounds=protocol['rounds'],
                                           feedback=protocol['feedback'])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite composition training loss.')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'step':step+1, 'loss':float(loss.detach()), 'nats':float(ce.detach()),
            'accuracy':float(query['logits'][:, -1].argmax(-1).eq(batch.answers).float().mean()),
            'gradient_norm':float(gradient), 'lr':lr,
            'expert_executions':prefix['expert_executions']+query['expert_executions'],
            'source_occurrences':prefix['source_occurrences']+query['source_occurrences']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--lookup-warmup', type=int, default=200)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--no-feedback', action='store_true')
    p.add_argument('--active-groups', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--validation-count', type=int, default=128)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    if min(args.steps,args.batch,args.warmup,args.rounds,args.threads,args.validate_every,args.save_every) < 1 or args.lookup_warmup < 0 or args.batch % 4 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error('Invalid positive training count or learning rate.')
    torch.set_num_threads(args.threads)
    protocol = {k:v for k,v in vars(args).items() if k not in ('out','resume','stop_after','no_feedback')}
    protocol.update(feedback=not args.no_feedback, source_hashes=source_hashes(), coverage=coverage(),
                    optimizer='AdamW beta=(0.9,0.95), weight decay 0.01 on matrices, clip 1',
                    schedule='linear warmup then constant', task='g(f(x)), B=f, C=g, A=query',
                    selection='lowest validation query NLL at the trained number of rounds',
                    prefix='B and C independently encode four table bytes; other nodes inactive')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        model, optimizer, rng, saved = load_checkpoint(args.out/'last.pt', protocol)
        start, best, counters, history = saved['step'], saved['best_validation'], saved['counters'], saved['history']
    else:
        if (args.out/'last.pt').exists():
            p.error('Output already contains a checkpoint; use --resume or a new output directory.')
        torch.manual_seed(args.seed)
        cfg = GraphConfig.tiny(readout_mode='concat', rounds=args.rounds,
                               recurrent_feedback=not args.no_feedback, evidence_gated_ports=True,
                               active_groups=args.active_groups, feedback_gate_init=.1, seed=args.seed)
        model = RegionLanguageModel(cfg)
        optimizer = optimizer_for(model, args.lr)
        rng = random.Random(args.seed)
        start, best, history = 0, float('inf'), []
        counters = dict(expert_executions=0, source_occurrences=0, episodes=0, training_seconds=0.)
        save_checkpoint(args.out/'initial.pt',model,optimizer,rng,0,protocol,best,counters,history)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'config':model.cfg.to_dict(),'parameters':model.parameter_summary(),'start':start}),flush=True)
    stop = min(args.steps,args.stop_after) if args.stop_after is not None else args.steps
    model.train()
    for step in range(start, stop):
        tick = time.perf_counter()
        batch = training_batch(rng,args.batch,lookup_warmup=step < args.lookup_warmup)
        row = update(model,optimizer,batch,step,protocol)
        row['seconds'] = time.perf_counter()-tick
        row['lookup_warmup'] = step < args.lookup_warmup
        counters['training_seconds'] += row['seconds']
        counters['episodes'] += args.batch
        for key in ('expert_executions','source_occurrences'):
            counters[key] += row[key]
        history.append(row)
        if step == start or (step+1)%25 == 0:
            print(json.dumps(row),flush=True)
        if (step+1)%args.validate_every == 0 or step+1 == args.steps:
            metrics = evaluate(model,count=args.validation_count,rounds=(args.rounds,))
            selected = metrics[f'k{args.rounds}']
            row['validation_nats'] = selected['nats_per_answer']
            row['validation_accuracy'] = selected['accuracy']
            print(json.dumps({'step':step+1,'validation_nats':row['validation_nats'],
                              'validation_accuracy':row['validation_accuracy']}),flush=True)
            if step+1 >= args.lookup_warmup and row['validation_nats'] < best:
                best = row['validation_nats']
                save_checkpoint(args.out/'best.pt',model,optimizer,rng,step+1,protocol,best,counters,history)
        if (step+1)%args.save_every == 0 or step+1 == stop:
            save_checkpoint(args.out/'last.pt',model,optimizer,rng,step+1,protocol,best,counters,history)
            (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps({'finished_step':stop,'best_validation_nats':best,**counters}),flush=True)


if __name__ == '__main__':
    main()
