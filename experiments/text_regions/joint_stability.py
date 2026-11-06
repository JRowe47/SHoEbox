"""Matched continued learning with broad retention validation and gradient logs."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from . import gradient_balance
from .composition_train import atomic_save, source_hashes
from .data import OrderedByteStreams, split_bytes
from .joint_text import load as load_parent, optimizer_for, evaluate_text, selection
from .config import GraphConfig
from .model import RegionLanguageModel
from .multifact import table_batches, COUNTERS
from .multifact_data import make_batch, KINDS
from .multifact_eval import evaluate
from .state import GraphState


def primary_conditions():
    return [(kind, kind, 'intact') for kind in KINDS] + [
        ('withheld_'+kind, kind, 'withheld') for kind in KINDS[1:]]


@torch.no_grad()
def validation(model, seed, count=512):
    was_training = model.training
    model.eval()
    try:
        return [dict(condition=name, **evaluate(model, 'validation', seed, count, 16, kind,
                    source_control=control, batch_size=count))
                for name, kind, control in primary_conditions()]
    finally:
        model.train(was_training)


def load(path, streams=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['schema'] != 'shoebox-joint-stability-v1':
        raise ValueError('Unknown joint-stability checkpoint.')
    model = RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights'])
    p = saved['protocol']
    optimizer = optimizer_for(model, p['core_lr'], p['readout_lr'], p['producer_lr'])
    optimizer.load_state_dict(saved['optimizer'])
    if streams is not None:
        streams.load_state_dict(saved['streams'])
    state = GraphState.from_dict(saved['graph_state']) if saved['graph_state'] is not None else None
    torch.set_rng_state(saved['torch_rng'])
    return model, optimizer, state, saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--mode', choices=('sum', 'group_clip'), required=True)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--validate-every', type=int, default=20)
    p.add_argument('--validation-count', type=int, default=512)
    p.add_argument('--validation-chunks', type=int, default=8)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    if min(a.steps, a.validate_every, a.validation_count, a.validation_chunks) < 1:
        p.error('Positive counts required.')
    if a.stop_after is not None and not 1 <= a.stop_after <= a.steps:
        p.error('Invalid stopping update.')
    raw = a.data.read_bytes()
    train, val, _ = split_bytes(raw)
    parent = torch.load(a.parent, map_location='cpu', weights_only=True)
    streams = OrderedByteStreams.from_bytes(train, parent['protocol']['batch'])
    protocol = dict(seed=a.seed, mode=a.mode, steps=a.steps, validate_every=a.validate_every,
        validation_count=a.validation_count, validation_chunks=a.validation_chunks,
        batch=parent['protocol']['batch'], context=parent['protocol']['context'], memory_batch=16,
        core_lr=.0003, readout_lr=.002, producer_lr=.0001, gradient_cap=1., memory_weight=1.,
        gaps=[0,4,8,16], parent_step=parent['step'], parent_text_bytes=parent['totals']['text_bytes'],
        parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(raw).hexdigest(), source_hashes=source_hashes(),
        selection='Every field in nine validation conditions >= .90, then minimum text NLL; ties retain earlier',
        initialization='Full parent weights, Adam moments, text cursor/state and fixed alignment targets; five-update LR warmup',
        memory_data='Fresh procedural episodes in new seed namespace; no replay',
        gradient_rule='One text backward and segmented memory backward, separately captured; task/group cap then sum for group_clip, direct sum for control; final global cap 1')
    a.out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        model, optimizer, state, saved = load(a.out/'last.pt', streams)
        if saved['protocol'] != protocol:
            raise ValueError('Resume requires identical source and protocol.')
        step, history, best, totals, reference = [saved[k] for k in ('step','history','best','totals','reference')]
    else:
        if (a.out/'last.pt').exists():
            raise ValueError('Use --resume or a new directory.')
        model, optimizer, state, saved = load_parent(a.parent, streams)
        reference = saved['reference']
        step, history, best = 0, [], [-1., -1e30, -1e30]
        totals = dict.fromkeys((*COUNTERS, 'text_bytes', 'training_seconds'), 0)
    del parent

    def save(name):
        atomic_save(a.out/name, dict(schema='shoebox-joint-stability-v1', config=model.cfg.to_dict(),
            weights=model.state_dict(), optimizer=optimizer.state_dict(), graph_state=state.to_dict() if state else None,
            streams=streams.state_dict(), torch_rng=torch.get_rng_state(), reference=reference,
            protocol=protocol, step=step, history=history, best=best, totals=totals))

    def validate(row):
        text = evaluate_text(model, val, protocol['batch'], protocol['context'], a.validation_chunks)
        scores = validation(model, a.seed, a.validation_count)
        row.update(text_validation={k:v for k,v in text.items() if k != 'rows'}, memory_validation=scores)
        candidate = list(selection(text, scores))
        print(json.dumps(dict(step=step, mode=a.mode, text_nats=text['nats'],
            memory_min=min(f['accuracy'] for s in scores for f in s['fields']), eligible=bool(candidate[0]))), flush=True)
        return candidate

    if not a.resume:
        row = dict(step=0)
        best = validate(row)
        history.append(row)
        save('initial.pt'); save('best.pt'); save('last.pt')
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for step in range(step+1, (a.stop_after or a.steps)+1):
        model.train()
        kind = KINDS[(step-1)%5]
        gap = protocol['gaps'][((step-1)//5)%4]
        offset = (step-1)*protocol['memory_batch']
        batch = make_batch('train', offset, protocol['memory_batch'], a.seed, gap, kind)
        tables = table_batches(batch, 'train', offset, a.seed)
        chunk = streams.next_chunk(protocol['context'])
        for group in optimizer.param_groups:
            group['lr'] = protocol[group['name']+'_lr']*min(1., step/5)
        tick = time.perf_counter()
        state, row = gradient_balance.update(model, optimizer, chunk, state, batch, tables, reference,
                                             a.mode, protocol['gradient_cap'], protocol['memory_weight'])
        row.update(step=step, kind=kind, gap=gap, training_seconds=time.perf_counter()-tick,
            state_bytes=state.storage_bytes(), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        history.append(row)
        for key in totals:
            totals[key] += row[key]
        if step%a.validate_every == 0 or step == a.steps:
            candidate = validate(row)
            if tuple(candidate) > tuple(best):
                best = candidate
                save('best.pt')
        if step == 1 or step%5 == 0:
            print(json.dumps({k:row[k] for k in ('step','text_nats','gradient_norm','training_seconds','gradient_groups')}), flush=True)
        if step%5 == 0 or step == (a.stop_after or a.steps):
            save('last.pt')
            (a.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    print(json.dumps(dict(complete=step==a.steps, step=step, best=best, **totals)), flush=True)


if __name__ == '__main__':
    main()
