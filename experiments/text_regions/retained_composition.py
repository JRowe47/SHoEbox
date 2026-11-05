"""One retained producer feeding the existing recurrent composition graph.

F receives fact bytes. B/C receive fresh relation tables. F's continuous
publication reaches A through an ordinary directed edge, without byte decoding.
Only the new edge's value matrix is fitted in this integration experiment.
"""
from dataclasses import replace
import hashlib
import random
import torch
from torch.nn import functional as F
from ..hat_minkron.events import ByteBatch
from .config import GraphConfig, NodeSpec, EdgeSpec
from .model import RegionLanguageModel
from .publication_data import make_batch, batch_digest
from .composition_data import pairs, batch_from_keys

EDGE = 'F__A__forward'


def build(composer_path, producer_path, seed):
    composer = torch.load(composer_path, map_location='cpu', weights_only=True)
    producer = torch.load(producer_path, map_location='cpu', weights_only=True)
    base = GraphConfig.from_dict(composer['config'])
    pc = GraphConfig.from_dict(producer['config'])
    # The imported components use identical latent and expert dimensions.
    for name in ('width', 'experts_per_block', 'shared_per_block', 'groups', 'expert_hidden',
                 'active_groups', 'experts_per_group', 'address_dim', 'laces', 'active_laces',
                 'hat_levels', 'hat_symbols', 'hat_active_symbols', 'kron_heads', 'kron_key', 'kron_value'):
        if getattr(base, name) != getattr(pc, name): raise ValueError('Incompatible component dimension: '+name)
    cfg = replace(base, nodes=base.nodes+(NodeSpec('F', 'memory', 'F'),),
                  edges=base.edges+(EdgeSpec('F', 'A'),), selective_publication=('F',),
                  independent_sources=('F',), seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model = RegionLanguageModel(cfg)
    weights = model.state_dict()
    for name in weights:
        if name.startswith('nodes.F.'):
            weights[name] = producer['weights'][name.replace('nodes.F.', 'nodes.A.', 1)].clone()
        elif name.startswith('independent_hats.F.'):
            weights[name] = producer['weights'][name.replace('independent_hats.F.', 'hat.', 1)].clone()
        elif name.startswith('independent_banks.F.'):
            weights[name] = producer['weights'][name.replace('independent_banks.F.', 'shared_banks.', 1)].clone()
        elif name.startswith('connections.ingress.F.'):
            weights[name] = producer['weights'][name.replace('connections.ingress.F.', 'connections.ingress.A.', 1)].clone()
        elif name in composer['weights']:
            weights[name] = composer['weights'][name].clone()
    model.load_state_dict(weights)
    edge = model.connections.edges[EDGE]
    with torch.no_grad():
        edge.context.weight.zero_(); edge.gate.weight.zero_(); edge.gate.bias.zero_()
    freeze(model)
    return model


def freeze(model):
    for p in model.parameters(): p.requires_grad_(False)
    model.connections.edges[EDGE].value.weight.requires_grad_(True)


def empty_inputs(model, batch, length):
    return {v: torch.zeros(batch, length, dtype=torch.long) for v in model.cfg.views}


def masks(model, batch, length, active):
    return {n: torch.full((batch, length), n in active, dtype=torch.bool) for n in model.nodes}


def ingest(model, batch, state=None, policy='learned', segment=16):
    if policy not in ('learned', 'always'): raise ValueError('Unknown publication policy.')
    if segment < 1: raise ValueError('Positive segment length required.')
    costs = dict(expert_executions=0, source_occurrences=0, external_events=0)
    for start in range(0, batch.inputs.shape[1], segment):
        data = batch.inputs[:, start:start+segment]; n, t = data.shape
        inputs = empty_inputs(model, n, t); inputs['F'] = data
        available = {v: batch.available[:, start:start+t] if v == 'F' else
                     torch.zeros(n, t, dtype=torch.bool) for v in inputs}
        result = model(inputs, state=state, reset=batch.reset[:, start:start+t], availability=available,
                       node_masks=masks(model, n, t, ('F',)), rounds=1, feedback=False,
                       publication_overrides={'F':'always'} if policy == 'always' else {})
        state = result['state'].detach()
        for key in costs: costs[key] += result[key]
    return state, costs


def query_state(state):
    """Clear the query workspace; preserve F and its independent byte context.

    A new epoch disambiguates fresh workspace versions. F retains its original
    publication epoch, version and provenance until a new observation replaces it.
    """
    fresh = state.reset(torch.ones_like(state.epochs, dtype=torch.bool))
    return replace(fresh, ingress={**fresh.ingress, 'F':state.ingress['F']},
                   nodes={**fresh.nodes, 'F':state.nodes['F']},
                   ports=fresh.ports.commit({n:state.ports.latest[n] for n in ('F','@F')}),
                   events_seen=state.events_seen, rounds_seen=state.rounds_seen)


def query(model, first, second, state, rounds=2, feedback=True, control='intact', trace=False):
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 4:
        raise ValueError('Two aligned four-byte tables required.')
    if control not in ('intact','no_edge','reset_memory','shuffle_memory'): raise ValueError('Unknown query control.')
    state = query_state(state)
    n = first.shape[0]
    if control == 'reset_memory': state = state.reset(torch.ones(n, dtype=torch.bool))
    if control == 'shuffle_memory':
        from .ports import map_tensors
        shuffled = map_tensors(state.ports.latest['F'], lambda x:x.roll(1,0))
        state = replace(state, ports=state.ports.commit({'F':shuffled}))
    inputs = empty_inputs(model,n,4); inputs.update(B=first,C=second)
    available = {v:torch.full((n,4),v in ('B','C'),dtype=torch.bool) for v in inputs}
    prefix = model(inputs,state=state,availability=available,node_masks=masks(model,n,4,('B','C')),
                   rounds=1,feedback=False)
    inputs = empty_inputs(model,n,1)
    for v in ('B','C'): inputs[v].fill_(ord('?'))
    available = {v:torch.full((n,1),v in ('B','C'),dtype=torch.bool) for v in inputs}
    result = model(inputs,state=prefix['state'],availability=available,
                   node_masks=masks(model,n,1,tuple(n for n in model.nodes if n != 'F')),
                   edge_masks={EDGE:torch.zeros(n,1,dtype=torch.bool)} if control == 'no_edge' else {},
                   rounds=rounds,feedback=feedback,trace=trace)
    for key in ('expert_executions','source_occurrences','external_events'): result[key] += prefix[key]
    return result


def tables(split, start, count, seed, answers, request=0):
    choices = pairs(split)
    keys = []
    for i, answer in enumerate(answers.tolist()):
        digest = hashlib.sha256(f'retained-query/{split}/{seed}/{start+i}/{request}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        a,b = choices[rng.randrange(len(choices))]
        keys.append((a,b,answer-65))
    return batch_from_keys(keys)


@torch.no_grad()
def teacher_inputs(model, answers):
    # Loss-only local alignment with the observed fact byte; no labels at inference.
    result = model.hat(ByteBatch.static(answers[:,None]), backend='step')
    return model.connections.ingress['A'](result['payload'][:,0])


def update(model, optimizer, batch, table_batch):
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): state, costs = ingest(model,batch)
    pub = state.ports.latest['F']
    actual, _ = model.connections.edges[EDGE](pub.payload,pub.header.memberships)
    target = teacher_inputs(model,batch.answers)
    alignment = F.mse_loss(actual,target)/target.square().mean().clamp_min(1e-8)
    result = query(model,table_batch.first,table_batch.second,state)
    ce = F.cross_entropy(result['logits'][:,-1],table_batch.answers)
    loss = alignment + ce
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), alignment=float(alignment.detach()), nats=float(ce.detach()),
                accuracy=float(result['logits'][:,-1].argmax(-1).eq(table_batch.answers).float().mean()),
                gradient_norm=float(norm), batch_sha256=batch_digest(batch), keys=table_batch.keys,
                **{key:costs[key]+result[key] for key in costs})


@torch.no_grad()
def evaluate(model, split, seed, count=512, gap=16, kind='correction', memory_control='carried',
             query_control='intact', rounds=2, feedback=True, policy='learned', requests=1, batch_size=64):
    predictions, targets, keys, valid, counterfactual_targets = [],[],[],[],[]
    costs = dict(expert_executions=0, source_occurrences=0, external_events=0)
    nll=0.; unchanged=True
    from .ports import tensor_list
    for start in range(0,count,batch_size):
        batch=make_batch(split,start,min(batch_size,count-start),seed,gap,kind,memory_control)
        state, budget=ingest(model,batch,policy=policy)
        for key in costs: costs[key]+=budget[key]
        for request in range(requests):
            tb=tables(split,start,len(batch.answers),seed,batch.answers,request)
            snapshot=[t.clone() for t in tensor_list((state.nodes['F'],state.ingress['F'],state.ports.latest['F']))]
            result=query(model,tb.first,tb.second,state,rounds,feedback,query_control)
            logits=result['logits'][:,-1]
            predictions.extend(logits.argmax(-1).tolist()); targets.extend(tb.answers.tolist()); keys.extend(tb.keys)
            counterfactual = batch_from_keys([(a,b,int(q)-65) for (a,b,_),q in zip(tb.keys,batch.answers.roll(1))])
            counterfactual_targets.extend(counterfactual.answers.tolist())
            valid.extend(result['state'].ports.latest['F'].valid.tolist())
            nll+=float(F.cross_entropy(logits,tb.answers,reduction='sum'))
            if query_control=='intact':
                unchanged &= all(torch.equal(a,b) for a,b in zip(snapshot,tensor_list((result['state'].nodes['F'],
                    result['state'].ingress['F'],result['state'].ports.latest['F']))))
            state=result['state']
            for key in costs: costs[key]+=result[key]
    return dict(accuracy=sum(a==b for a,b in zip(predictions,targets))/len(targets),nats=nll/len(targets),
                predictions=predictions,targets=targets,keys=keys,memory_valid=valid,
                counterfactual_targets=counterfactual_targets,
                counterfactual_accuracy=sum(a==b for a,b in zip(predictions,counterfactual_targets))/len(targets),
                exact_memory_preserved=unchanged,split=split,seed=seed,count=count,gap=gap,kind=kind,
                memory_control=memory_control,query_control=query_control,rounds=rounds,feedback=feedback,
                policy=policy,requests=requests,**costs)


@torch.no_grad()
def evaluate_revision(model,split,seed,count=512,batch_size=64):
    """Read a fact, ask three live queries, then observe and read a correction."""
    records={name:dict(predictions=[],targets=[]) for name in ('before','after','withheld')}
    for start in range(0,count,batch_size):
        n=min(batch_size,count-start)
        old=make_batch(split,start,n,seed,16,'recall')
        state,_=ingest(model,old)
        for request in range(3):
            tb=tables(split,start,n,seed,old.answers,request)
            r=query(model,tb.first,tb.second,state);state=r['state']
            records['before']['predictions'].extend(r['logits'][:,-1].argmax(-1).tolist())
            records['before']['targets'].extend(tb.answers.tolist())
        for control in ('carried','withheld'):
            changed=make_batch(split,start,n,seed,16,'correction',control)
            changed=replace(changed,reset=torch.zeros_like(changed.reset))
            updated,_=ingest(model,changed,state)
            tb=tables(split,start,n,seed,changed.answers,3)
            r=query(model,tb.first,tb.second,updated)
            row=records['after' if control=='carried' else 'withheld']
            row['predictions'].extend(r['logits'][:,-1].argmax(-1).tolist());row['targets'].extend(tb.answers.tolist())
    for row in records.values():row['accuracy']=sum(a==b for a,b in zip(row['predictions'],row['targets']))/len(row['targets'])
    return records
