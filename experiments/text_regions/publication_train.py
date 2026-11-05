"""Learn acquisition and exact publication retention in an isolated SHoE node."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from ..hat_minkron.train import optimizer_for
from .composition_train import atomic_save, source_hashes
from .config import GraphConfig, NodeSpec
from .model import RegionLanguageModel
from .publication_data import make_batch, batch_digest


def build_producer(parent, seed):
    """Reuse the parent's A expert, ingress and shared banks; initialize a byte head."""
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    cfg = replace(GraphConfig.from_dict(saved['config']), nodes=(NodeSpec('A', 'R0', 'A'),),
                  edges=(), readout=('A',), rounds=1, recurrent_feedback=False,
                  readout_mode='concat', selective_publication=('A',), seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RegionLanguageModel(cfg)
    current = model.state_dict()
    for name in current:
        if name in saved['weights'] and not name.startswith('readout.'):
            current[name] = saved['weights'][name].clone()
    model.load_state_dict(current)
    return model


def forward_batch(model, batch, policy, segment=16, capture=True):
    if policy not in ('learned', 'always') or segment < 1:
        raise ValueError('Invalid publication policy or segment length.')
    state, parts = None, []
    for start in range(0, batch.inputs.shape[1], segment):
        end = start+segment
        result = model({'A': batch.inputs[:, start:end]}, state=state,
                       reset=batch.reset[:, start:end], availability={'A': batch.available[:, start:end]},
                       return_node_activity=('A',) if capture else (),
                       return_publication_activity=capture,
                       publication_overrides={'A': 'always'} if policy == 'always' else {})
        parts.append(result)
        state = result['state'].detach()
    output = {'logits': torch.cat([p['logits'] for p in parts], 1), 'state': state,
              **{key: sum(p[key] for p in parts) for key in
                 ('expert_executions', 'external_events', 'source_occurrences')}}
    # Weight router regularizers by events rather than by unequal chunk counts.
    total = output['external_events']
    for key in ('aux', 'hat_aux'):
        output[key] = sum(p[key]*p['external_events']/total for p in parts)
    if capture:
        output['payload'] = torch.cat([p['node_activity']['A']['payload'] for p in parts], 1)
        for key in ('candidate', 'logit', 'written'):
            output[key] = torch.cat([p['publication_activity']['A'][key][:, :, 0] for p in parts], 1)
    return output


def update(model, optimizer, batch, policy):
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(model, batch, policy)
    known = batch.retained.ne(-100)
    retain_ce = F.cross_entropy(result['logits'][known], batch.retained[known])
    candidate_ce = F.cross_entropy(model.readout(result['candidate'][batch.writes]), batch.inputs[batch.writes])
    # Balanced binary loss avoids learning "hold everything" from rare writes.
    gate_ce = .5*(F.softplus(-result['logit'][batch.writes]).mean() +
                 F.softplus(result['logit'][~batch.writes]).mean())
    loss = retain_ce + candidate_ce + gate_ce
    loss = loss + model.cfg.auxiliary_loss_weight*result['aux'] + model.cfg.hat_balance_weight*result['hat_aux']
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), retained_nats=float(retain_ce.detach()),
                candidate_nats=float(candidate_ce.detach()), gate_nats=float(gate_ce.detach()),
                accuracy=float(result['logits'][:, -1].argmax(-1).eq(batch.answers).float().mean()),
                gradient_norm=float(norm), batch_sha256=batch_digest(batch),
                **{key: result[key] for key in ('expert_executions', 'external_events', 'source_occurrences')})


@torch.no_grad()
def evaluate(model, policy, split, seed, count=128, gap=16, kind='correction', control='carried', batch_size=128):
    predictions, targets, confidences = [], [], []
    totals = dict(true_positive=0, false_positive=0, false_negative=0, true_negative=0,
                  hold_pairs=0, exact_hold_pairs=0, expert_executions=0, external_events=0, source_occurrences=0)
    nll = 0.
    for start in range(0, count, batch_size):
        batch = make_batch(split, start, min(batch_size, count-start), seed, gap, kind, control)
        result = forward_batch(model, batch, policy)
        logits = result['logits'][:, -1]
        predictions.extend(logits.argmax(-1).tolist()); targets.extend(batch.answers.tolist())
        confidences.extend(logits.softmax(-1).amax(-1).tolist())
        nll += float(F.cross_entropy(logits, batch.answers, reduction='sum'))
        predicted = result['logit'].ge(0) & batch.available
        truth = batch.writes
        for key, mask in (('true_positive', predicted & truth), ('false_positive', predicted & ~truth),
                          ('false_negative', ~predicted & truth), ('true_negative', ~predicted & ~truth)):
            totals[key] += int(mask.sum())
        held = ~result['written'][:, 1:] & ~batch.reset[:, 1:]
        equal = result['payload'][:, 1:].eq(result['payload'][:, :-1]).all(-1)
        totals['hold_pairs'] += int(held.sum()); totals['exact_hold_pairs'] += int((held & equal).sum())
        for key in ('expert_executions', 'external_events', 'source_occurrences'): totals[key] += result[key]
    tp, fp, fn = totals['true_positive'], totals['false_positive'], totals['false_negative']
    return dict(split=split, count=count, gap=gap, kind=kind, control=control,
                accuracy=sum(a == b for a, b in zip(predictions, targets))/count, nats=nll/count,
                write_precision=tp/max(1, tp+fp), write_recall=tp/max(1, tp+fn),
                predictions=predictions, targets=targets, confidence=confidences, **totals)


def validation(model, policy, seed, count):
    scores = [evaluate(model, policy, 'validation', seed, count, 16, kind, control)
              for kind, control in (('recall', 'carried'), ('correction', 'carried'), ('correction', 'withheld'))]
    key = (min(s['accuracy'] for s in scores), sum(s['accuracy'] for s in scores)/3,
           -sum(s['nats'] for s in scores)/3)
    return key, [{k: v for k, v in score.items() if k not in ('predictions', 'targets', 'confidence')} for score in scores]


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != 'shoebox-publication-v1':
        raise ValueError('Unknown publication checkpoint schema.')
    model = RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights'])
    optimizer = optimizer_for(model, saved['protocol']['lr'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    return model, optimizer, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy', choices=('learned', 'always'), default='learned')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--validate-every', type=int, default=50)
    parser.add_argument('--validation-count', type=int, default=128)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.validate_every, args.validation_count) < 1 or not 0 < args.lr < 1:
        parser.error('Invalid training dimensions or rate.')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error('Invalid stop step.')
    torch.set_num_threads(1)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('parent', 'out', 'resume', 'stop_after')}
    protocol.update(parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                    source_hashes=source_hashes(), segment=16, gaps=[0, 4, 8, 16],
                    task='Alternating recall/correction; matched acquisition, retention and write-classification losses',
                    selection='Validation minimum accuracy, mean accuracy, then lower NLL; no test selection',
                    gate='Hard threshold at zero; balanced BCE trains decision, no straight-through estimator',
                    optimizer='Fresh AdamW beta=(0.9,0.95), decay .01 on matrices, clip 1')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, optimizer, saved = load(args.out/'last.pt')
        if saved['protocol'] != protocol: raise ValueError('Resume requires unchanged source and protocol.')
        step, history, totals, best = saved['step'], saved['history'], saved['totals'], saved['best']
    else:
        if (args.out/'last.pt').exists(): raise ValueError('Use --resume or a fresh output directory.')
        torch.manual_seed(args.seed)
        model = build_producer(args.parent, args.seed)
        optimizer = optimizer_for(model, args.lr)
        step, history, best = 0, [], [-1., -1., -1e30]
        totals = dict(training_seconds=0., expert_executions=0, external_events=0, source_occurrences=0)
    def save(name):
        atomic_save(args.out/name, dict(schema='shoebox-publication-v1', config=model.cfg.to_dict(),
                    weights=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    step=step, history=history, totals=totals, best=best, protocol=protocol))
    if not args.resume: save('initial.pt')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for step in range(step+1, (args.stop_after or args.steps)+1):
        # Every gap occurs with both task kinds in each eight-update cycle.
        kind = 'recall' if step % 2 else 'correction'
        gap = protocol['gaps'][((step-1)//2) % 4]
        batch = make_batch('train', (step-1)*args.batch, args.batch, args.seed, gap, kind)
        tick = time.perf_counter(); row = update(model, optimizer, batch, args.policy)
        row.update(step=step, kind=kind, gap=gap, training_seconds=time.perf_counter()-tick)
        for key in totals: totals[key] += row[key]
        history.append(row)
        if step % args.validate_every == 0 or step == args.steps:
            key, scores = validation(model, args.policy, args.seed, args.validation_count)
            row['validation'] = scores
            if key > tuple(best): best = list(key); save('best.pt')
            print(json.dumps({'step': step, 'validation': scores}), flush=True)
        if step == 1 or step % 10 == 0: print(json.dumps({k:v for k,v in row.items() if k != 'validation'}), flush=True)
        if step % 10 == 0 or step == (args.stop_after or args.steps):
            save('last.pt'); (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    print(json.dumps({'complete': step == args.steps, 'step': step, 'best': best, **totals}), flush=True)


if __name__ == '__main__': main()
