"""Local producer/edge learning with a frozen recurrent composition workspace.

Supervision is applied to current activity before each truncated-backpropagation
boundary. Inference uses the unchanged retained_composition ingest/query path.
"""
import torch
from torch.nn import functional as F
from .retained_composition import EDGE, empty_inputs, masks, query, teacher_inputs
from .publication_data import batch_digest

COUNTERS = ('expert_executions', 'external_events', 'source_occurrences')
PRODUCER_PREFIXES = ('nodes.F.', 'independent_hats.F.', 'independent_banks.F.',
                     'connections.ingress.F.')
EDGE_WEIGHT = 'connections.edges.' + EDGE + '.value.weight'


def allowed(name, mode):
    return name == EDGE_WEIGHT or (mode == 'joint' and name.startswith(PRODUCER_PREFIXES))


def configure(model, mode):
    if mode not in ('joint', 'edge'): raise ValueError('Unknown training mode.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(allowed(name, mode))


def optimizer_for(model, mode, producer_lr=.0003, edge_lr=.001):
    configure(model, mode)
    edge = model.connections.edges[EDGE].value.weight
    groups = [{'params': [edge], 'lr': edge_lr}]
    producer = [p for p in model.parameters() if p.requires_grad and p is not edge]
    if producer: groups.append({'params': producer, 'lr': producer_lr})
    return torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=.01)


def producer_segment(model, batch, start, end, state=None):
    data = batch.inputs[:, start:end]
    n, t = data.shape
    inputs = empty_inputs(model, n, t); inputs['F'] = data
    available = {v: batch.available[:, start:end] if v == 'F' else
                 torch.zeros(n, t, dtype=torch.bool) for v in inputs}
    return model(inputs, state=state, reset=batch.reset[:, start:end], availability=available,
                 node_masks=masks(model, n, t, ('F',)), rounds=1, feedback=False,
                 return_node_activity=('F',), return_publication_activity=True)


def project(model, payload):
    # Use exactly the ordinary edge's value/gate path. Its semantic projection
    # remains zero and frozen, as in the parent integration experiment.
    edge = model.connections.edges[EDGE]
    return edge.gate(payload).sigmoid()*edge.value(payload)


def update(model, optimizer, batch, table_batch, segment=16):
    if segment < 1: raise ValueError('Positive segment required.')
    optimizer.zero_grad(set_to_none=True)
    reference = teacher_inputs(model, torch.arange(65, 69))
    scale = reference.square().mean().clamp_min(1e-8)
    known_count = int(batch.retained.ne(-100).sum())
    write_count = int(batch.writes.sum())
    hold_count = batch.writes.numel()-write_count
    if not min(known_count, write_count, hold_count): raise ValueError('Acquisition batch required.')
    metrics = dict(retained_alignment=0., candidate_alignment=0., gate_nats=0., regularizer=0.)
    costs = dict.fromkeys(COUNTERS, 0)
    state = None
    for start in range(0, batch.inputs.shape[1], segment):
        end = min(start+segment, batch.inputs.shape[1])
        result = producer_segment(model, batch, start, end, state)
        activity = result['publication_activity']['F']
        retained = batch.retained[:, start:end]; known = retained.ne(-100)
        writes = batch.writes[:, start:end]
        payload = result['node_activity']['F']['payload']
        candidate = activity['candidate'][:, :, 0]
        logit = activity['logit'][:, :, 0]
        # Denominators cover the full batch, so unequal chunks do not change
        # weighting. Empty selections contribute differentiable zero sums.
        retention = (project(model, payload[known])-reference[retained[known]-65]).square().sum()
        retention = retention/(known_count*model.cfg.width*scale)
        acquisition = (project(model, candidate[writes])-reference[batch.inputs[:, start:end][writes]-65]).square().sum()
        acquisition = acquisition/(write_count*model.cfg.width*scale)
        gate = .5*(F.softplus(-logit[writes]).sum()/write_count +
                   F.softplus(logit[~writes]).sum()/hold_count)
        regularizer = (model.cfg.auxiliary_loss_weight*result['aux'] +
                       model.cfg.hat_balance_weight*result['hat_aux'])*(end-start)/batch.inputs.shape[1]
        local = retention+acquisition+gate+regularizer
        for name, value in zip(metrics, (retention, acquisition, gate, regularizer)):
            metrics[name] += float(value.detach())
        for name in costs: costs[name] += result[name]
        if end == batch.inputs.shape[1]:
            final = query(model, table_batch.first, table_batch.second, result['state'])
            ce = F.cross_entropy(final['logits'][:, -1], table_batch.answers)
            (local+ce).backward()
            for name in costs: costs[name] += final[name]
        else:
            local.backward()
        state = result['state'].detach()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=sum(metrics.values())+float(ce.detach()), nats=float(ce.detach()),
                accuracy=float(final['logits'][:, -1].argmax(-1).eq(table_batch.answers).float().mean()),
                gradient_norm=float(norm), batch_sha256=batch_digest(batch), keys=table_batch.keys,
                **metrics, **costs)


@torch.no_grad()
def publication_metrics(model, split, seed, count=512, gap=16, kind='correction', batch_size=64):
    from .publication_data import make_batch
    totals = dict(true_positive=0, false_positive=0, false_negative=0, true_negative=0,
                  hold_pairs=0, exact_hold_pairs=0)
    for start in range(0, count, batch_size):
        batch = make_batch(split, start, min(batch_size, count-start), seed, gap, kind)
        state = None; payloads = []; written = []; logits = []
        for offset in range(0, batch.inputs.shape[1], 16):
            result = producer_segment(model, batch, offset, min(offset+16, batch.inputs.shape[1]), state)
            state = result['state'].detach()
            a = result['publication_activity']['F']
            payloads.append(result['node_activity']['F']['payload'])
            written.append(a['written'][:, :, 0]); logits.append(a['logit'][:, :, 0])
        prediction = torch.cat(logits, 1).ge(0) & batch.available
        truth = batch.writes
        for key, mask in (('true_positive', prediction & truth), ('false_positive', prediction & ~truth),
                          ('false_negative', ~prediction & truth), ('true_negative', ~prediction & ~truth)):
            totals[key] += int(mask.sum())
        held = ~torch.cat(written, 1)[:, 1:] & ~batch.reset[:, 1:]
        payload = torch.cat(payloads, 1)
        equal = payload[:, 1:].eq(payload[:, :-1]).all(-1)
        totals['hold_pairs'] += int(held.sum()); totals['exact_hold_pairs'] += int((held & equal).sum())
    tp, fp, fn = (totals[k] for k in ('true_positive', 'false_positive', 'false_negative'))
    return dict(write_precision=tp/max(1, tp+fp), write_recall=tp/max(1, tp+fn), **totals)
