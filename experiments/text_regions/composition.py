"""Independent source encoding followed by complete regional query rounds."""
import math
import torch
from torch.nn import functional as F
from .composition_data import batch_from_keys, evaluation_keys, shuffled_bridge


def encode_sources(model, batch, control='intact'):
    if control not in ('intact', 'without_b', 'without_c'):
        raise ValueError('Unknown source control.')
    n = batch.size
    inputs = {'A': torch.zeros(n, 4, dtype=torch.long), 'B': batch.first, 'C': batch.second}
    availability = {v: torch.full((n, 4), v != 'A' and control != 'without_'+v.lower(), dtype=torch.bool)
                    for v in ('A', 'B', 'C')}
    # Retain the same node/expert execution budget when evidence is withheld.
    node_masks = {name: torch.full((n, 4), name in ('B', 'C'), dtype=torch.bool)
                  for name in model.nodes}
    return model(inputs, availability=availability, node_masks=node_masks, rounds=1, feedback=False)


def answer_query(model, batch, prefix_state, rounds=2, feedback=True, trace=False):
    n = batch.size
    # A carries the only query-dependent byte. B/C receive a fixed read cue,
    # which can address their recurrent state without revealing A's query.
    inputs = {'A': batch.query, 'B': torch.full((n,1),ord('?'),dtype=torch.long),
              'C': torch.full((n,1),ord('?'),dtype=torch.long)}
    availability = {v: torch.ones(n,1,dtype=torch.bool) for v in inputs}
    # The source byte is ingested exactly once, regardless of internal rounds.
    return model(inputs, state=prefix_state, availability=availability,
                 rounds=rounds, feedback=feedback, trace=trace)


def training_loss(model, batch, rounds=2, feedback=True, source_reconstruction=0.):
    prefix = encode_sources(model, batch)
    query = answer_query(model, batch, prefix['state'], rounds=rounds, feedback=feedback)
    ce = F.cross_entropy(query['logits'][:, -1], batch.answers)
    cfg = model.cfg
    loss = ce + cfg.auxiliary_loss_weight*(prefix['aux'] + query['aux'])/2
    loss = loss + cfg.hat_balance_weight*(prefix['hat_aux'] + query['hat_aux'])/2
    if source_reconstruction:
        # Self-supervision from the bytes actually observed by B and C. Both
        # pre-query and post-query states must preserve their own four entries.
        states = torch.cat([r['state'].ports.latest[n].payload
                            for r in (prefix,query) for n in ('B','C')])
        targets = torch.cat([batch.first,batch.second,batch.first,batch.second])-65
        reconstructed = model.source_reconstruction(states).reshape(-1,4)
        local = F.cross_entropy(reconstructed,targets.flatten())
        query['source_loss'] = local
        query['source_accuracy'] = reconstructed.argmax(-1).eq(targets.flatten()).float().mean()
        loss = loss + source_reconstruction*local
    return loss, ce, prefix, query


def empty_metrics():
    return dict(episodes=0, correct=0, nll_sum=0., prefix_expert_executions=0,
                query_expert_executions=0, source_occurrences=0, predictions=[], targets=[], keys=[])


def accumulate(metrics, batch, prefix, result, targets=None):
    targets = batch.answers if targets is None else targets
    logits = result['logits'][:, -1]
    predictions = logits.argmax(-1)
    metrics['episodes'] += batch.size
    metrics['correct'] += int(predictions.eq(targets).sum())
    metrics['nll_sum'] += float(F.cross_entropy(logits, targets, reduction='sum'))
    metrics['prefix_expert_executions'] += prefix['expert_executions']
    metrics['query_expert_executions'] += result['expert_executions']
    metrics['source_occurrences'] += prefix['source_occurrences'] + result['source_occurrences']
    metrics['predictions'].extend(predictions.tolist())
    metrics['targets'].extend(targets.tolist())
    metrics['keys'].extend(batch.keys)


def finish(metrics):
    metrics['accuracy'] = metrics['correct']/metrics['episodes']
    metrics['nats_per_answer'] = metrics['nll_sum']/metrics['episodes']
    metrics['bits_per_answer'] = metrics['nats_per_answer']/math.log(2)
    metrics['expert_executions'] = metrics['prefix_expert_executions'] + metrics['query_expert_executions']
    return metrics


@torch.no_grad()
def evaluate(model, split='validation', count=128, batch_size=64, rounds=(1,2,4),
             controls=False, seed=819):
    keys = evaluation_keys(split, count, seed)
    was_training = model.training; model.eval()
    records = {f'k{k}': empty_metrics() for k in rounds}
    if controls:
        records.update({name: empty_metrics() for name in
                        ('k2_no_exchange', 'k2_shuffled_bridge', 'k2_counterfactual', 'k2_reset')})
    try:
        for start in range(0, len(keys), batch_size):
            batch = batch_from_keys(keys[start:start+batch_size])
            prefix = encode_sources(model, batch)
            state = prefix['state']
            for k in rounds:
                result = answer_query(model, batch, state, k, feedback=True)
                accumulate(records[f'k{k}'], batch, prefix, result)
            if controls:
                result = answer_query(model, batch, state, 2, feedback=False)
                accumulate(records['k2_no_exchange'], batch, prefix, result)
                reset = state.reset(torch.ones(batch.size, dtype=torch.bool))
                result = answer_query(model, batch, reset, 2, feedback=True)
                accumulate(records['k2_reset'], batch, prefix, result)
                changed = shuffled_bridge(batch)
                changed_prefix = encode_sources(model, changed)
                result = answer_query(model, changed, changed_prefix['state'], 2, feedback=True)
                accumulate(records['k2_shuffled_bridge'], changed, changed_prefix, result, targets=batch.answers)
                accumulate(records['k2_counterfactual'], changed, changed_prefix, result)
        return {name: finish(value) for name, value in records.items()}
    finally:
        model.train(was_training)
