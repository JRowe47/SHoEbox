"""Master lookup before advancing; preserve it with freshly generated examples."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from ..hat_minkron.train import optimizer_for
from .composition import accumulate, answer_query, empty_metrics, encode_sources, finish
from .composition_data import batch_from_keys, pairs
from .composition_train import build_model, load_checkpoint, save_checkpoint, source_hashes, update
from .config import GraphConfig


def skill_pairs(split, skill):
    """The two lookup orientations have separate gates; identity/identity excluded."""
    predicates = {'lookup_b': lambda a, b: a != 0 and b == 0,
                  'lookup_c': lambda a, b: a == 0 and b != 0,
                  'composition': lambda a, b: a != 0 and b != 0}
    if skill not in predicates:
        raise ValueError('Unknown skill.')
    return tuple((a, b) for a, b in pairs(split) if predicates[skill](a, b))


def curriculum_batch(rng, batch_size, stage, lookup_fraction=.5):
    if batch_size < 8 or batch_size % 8 or stage not in ('lookup', 'composition'):
        raise ValueError('Batch must be a positive multiple of eight; invalid stage.')
    if not 0 <= lookup_fraction <= 1:
        raise ValueError('Lookup fraction must be in [0, 1].')
    groups = batch_size // 4
    lookup_groups = groups if stage == 'lookup' else 2*round(groups*lookup_fraction/2)
    # Each table pair contributes all four queries, preserving answer balance.
    skills = ['lookup_b', 'lookup_c']*(lookup_groups//2)
    skills += ['composition']*(groups-lookup_groups)
    keys = []
    for skill in skills:
        a, b = rng.choice(skill_pairs('train', skill))
        keys.extend((a, b, q) for q in range(4))
    rng.shuffle(keys)
    return batch_from_keys(keys)


@torch.no_grad()
def evaluate_skills(model, split='validation', batch_size=64, rounds=2, feedback=True):
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    was_training = model.training
    model.eval()
    metrics = {}
    try:
        for skill in ('lookup_b', 'lookup_c', 'composition'):
            keys = [(a, b, q) for a, b in skill_pairs(split, skill) for q in range(4)]
            record = empty_metrics()
            for start in range(0, len(keys), batch_size):
                batch = batch_from_keys(keys[start:start+batch_size])
                prefix = encode_sources(model, batch)
                result = answer_query(model, batch, prefix['state'], rounds, feedback)
                accumulate(record, batch, prefix, result)
            metrics[skill] = finish(record)
        return metrics
    finally:
        model.train(was_training)


@dataclass
class Mastery:
    threshold: float = .9
    patience: int = 2
    stage: str = 'lookup'
    streak: int = 0
    transition_step: int | None = None

    def observe(self, metrics, step):
        passed = all(metrics[s]['accuracy'] >= self.threshold for s in ('lookup_b', 'lookup_c'))
        if self.stage == 'lookup':
            self.streak = self.streak+1 if passed else 0
            if self.streak >= self.patience:
                self.stage = 'composition'
                self.transition_step = step
        return passed


def metric_summary(metrics):
    return {k: {n: v[n] for n in ('episodes', 'correct', 'accuracy', 'nats_per_answer')}
            for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--initialize', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--no-feedback', action='store_true')
    p.add_argument('--active-groups', type=int, default=2)
    p.add_argument('--source-reconstruction', type=float, default=.5)
    p.add_argument('--lookup-fraction', type=float, default=.5)
    p.add_argument('--threshold', type=float, default=.9)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--validate-every', type=int, default=50)
    p.add_argument('--save-every', type=int, default=50)
    args = p.parse_args()
    if min(args.steps,args.batch,args.warmup,args.rounds,args.threads,args.validate_every,
           args.save_every,args.patience) < 1 or args.batch % 8:
        p.error('Positive counts and a batch divisible by eight are required.')
    if not (0 < args.threshold <= 1 and 0 <= args.lookup_fraction <= 1 and
            math.isfinite(args.lr) and args.lr > 0 and
            math.isfinite(args.source_reconstruction) and args.source_reconstruction >= 0):
        p.error('Invalid learning rate, threshold, mixture or reconstruction weight.')
    torch.set_num_threads(args.threads)
    protocol = {k:v for k,v in vars(args).items() if k not in
                ('out','resume','stop_after','no_feedback','initialize')}
    protocol.update(feedback=not args.no_feedback, source_hashes=source_hashes(),
                    initialize_sha256=(hashlib.sha256(args.initialize.read_bytes()).hexdigest()
                                       if args.initialize else None),
                    selection='Validation NLL; composition requires both lookup accuracies >= threshold',
                    curriculum='Two consecutive separate lookup gates, then fresh lookup/composition mixture')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        model, optimizer, rng, saved = load_checkpoint(args.out/'last.pt', protocol)
        start, best, counters, history = saved['step'],saved['best_validation'],saved['counters'],saved['history']
        mastery = Mastery(**counters['mastery'])
    else:
        if (args.out/'last.pt').exists():
            p.error('Checkpoint exists: use --resume or a new directory.')
        torch.manual_seed(args.seed)
        if args.initialize:
            model, _, _, saved = load_checkpoint(args.initialize)
            if (model.cfg.active_groups != args.active_groups or
                bool(hasattr(model,'source_reconstruction')) != bool(args.source_reconstruction)):
                p.error('Initialization must match active groups and reconstruction-head presence.')
            torch.manual_seed(args.seed)
        else:
            cfg = GraphConfig.tiny(readout_mode='concat',rounds=args.rounds,
                                  recurrent_feedback=not args.no_feedback,evidence_gated_ports=True,
                                  active_groups=args.active_groups,feedback_gate_init=.1,seed=args.seed)
            model = build_model(cfg,args.source_reconstruction)
        optimizer = optimizer_for(model,args.lr)
        rng = random.Random(args.seed)
        mastery = Mastery(args.threshold,args.patience)
        start, best, history = 0,float('inf'),[]
        counters = dict(episodes=0,expert_executions=0,source_occurrences=0,training_seconds=0.,
                        best_lookup_nats=float('inf'),mastery=asdict(mastery))
        save_checkpoint(args.out/'initial.pt',model,optimizer,rng,0,protocol,best,counters,history)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'start':start,'parameters':model.parameter_summary(),'mastery':asdict(mastery)}),flush=True)
    stop = min(args.steps,args.stop_after) if args.stop_after is not None else args.steps
    model.train()
    for step in range(start,stop):
        tick = time.perf_counter()
        stage = mastery.stage
        batch = curriculum_batch(rng,args.batch,stage,args.lookup_fraction)
        row = update(model,optimizer,batch,step,protocol)
        row.update(stage=stage,seconds=time.perf_counter()-tick)
        history.append(row)
        counters['episodes'] += args.batch
        counters['training_seconds'] += row['seconds']
        for name in ('expert_executions','source_occurrences'):
            counters[name] += row[name]
        if step == start or (step+1)%25 == 0:
            print(json.dumps(row),flush=True)
        if (step+1)%args.validate_every == 0 or step+1 == args.steps:
            metrics = evaluate_skills(model,rounds=args.rounds,feedback=not args.no_feedback)
            row['validation'] = metric_summary(metrics)
            passed = mastery.observe(metrics,step+1)
            counters['mastery'] = asdict(mastery)
            lookup_nats = sum(metrics[s]['nats_per_answer'] for s in ('lookup_b','lookup_c'))/2
            if lookup_nats < counters['best_lookup_nats']:
                counters['best_lookup_nats'] = lookup_nats
                save_checkpoint(args.out/'best_lookup.pt',model,optimizer,rng,step+1,protocol,best,counters,history)
            if mastery.transition_step == step+1:
                save_checkpoint(args.out/'lookup_mastered.pt',model,optimizer,rng,step+1,protocol,best,counters,history)
            # A gate-transition checkpoint has not trained on composition yet.
            if stage == 'composition' and passed and metrics['composition']['nats_per_answer'] < best:
                best = metrics['composition']['nats_per_answer']
                save_checkpoint(args.out/'best.pt',model,optimizer,rng,step+1,protocol,best,counters,history)
            print(json.dumps({'step':step+1,'validation':row['validation'],'mastery':asdict(mastery)}),flush=True)
        if (step+1)%args.save_every == 0 or step+1 == stop:
            save_checkpoint(args.out/'last.pt',model,optimizer,rng,step+1,protocol,best,counters,history)
            (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps({'finished_step':stop,'best_composition_nats':best,**counters}),flush=True)


if __name__ == '__main__':
    main()
