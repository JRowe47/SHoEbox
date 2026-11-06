"""Matched complete-episode updates with or without cross-segment gradients.

Both arms execute exactly the same chunks. Only detachment of the carried
graph state differs. Losses are normalized over the whole episode in both
arms, so moving a boundary cannot reweight the examples or local factors.
"""
import hashlib
import torch
from torch.nn import functional as F

from .composition import training_loss
from .composition_data import training_batch
from .fact_supervision import retention_loss
from scripts.train_integrated_balanced import clip_task_gradients


def episode_forward(model, batch, segment=16, detach_between=False, capture=True):
    if segment < 1:
        raise ValueError('Segment length must be positive.')
    state, parts = None, []
    for start in range(0, batch.length, segment):
        chunk = batch.slice(start, min(start + segment, batch.length))
        result = model(chunk.inputs, state=state, reset=chunk.reset,
                       availability=chunk.availability, rounds=2, feedback=True,
                       return_node_activity=tuple(model.nodes) if capture else ())
        parts.append(result)
        state = result['state'].detach() if detach_between else result['state']
    events = sum(p['external_events'] for p in parts)
    # Every graph node runs twice per event. Ingress activity is source-specific.
    hat_denominators = [p['external_events'] * len(model.nodes) * 2 +
                        p['source_occurrences'] for p in parts]
    return {
        'logits': torch.cat([p['logits'] for p in parts], 1),
        'state': parts[-1]['state'],
        'aux': sum(p['aux'] * p['external_events'] for p in parts) / events,
        'hat_aux': sum(p['hat_aux'] * w for p, w in zip(parts, hat_denominators)) /
                   sum(hat_denominators),
        'node_activity': {name: {key: torch.cat([p['node_activity'][name][key]
                                                for p in parts], 1)
                                 for key in ('payload', 'valid')}
                          for name in model.nodes} if capture else {},
        **{key: sum(p[key] for p in parts) for key in
           ('expert_executions', 'external_events', 'source_occurrences')},
    }


def episode_loss(model, batch, segment=16, detach_between=False, local_weight=1.):
    result = episode_forward(model, batch, segment, detach_between, bool(local_weight))
    mask = batch.targets.ne(-100)
    ce = (F.cross_entropy(result['logits'][mask], batch.targets[mask])
          if bool(mask.any()) else result['logits'].sum() * 0.)
    loss = ce + model.cfg.auxiliary_loss_weight * result['aux']
    loss = loss + model.cfg.hat_balance_weight * result['hat_aux']
    result['ce'] = ce
    if local_weight:
        local, result['retention'] = retention_loss(model, batch, result['node_activity'])
        loss = loss + local_weight * local
    return loss, result


def batch_digest(batch):
    """Hash actual observations, masks and loss targets, not just batch IDs."""
    digest = hashlib.sha256()
    def add(name, tensor):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if hasattr(batch, 'keys'):
        for name in ('first', 'second', 'query', 'answers'):
            add(name, getattr(batch, name))
        digest.update(repr(batch.keys).encode())
    else:
        inputs = batch.inputs if isinstance(batch.inputs, dict) else {'text': batch.inputs}
        for name, value in sorted(inputs.items()): add('input/' + name, value)
        for name, value in sorted(batch.availability.items()): add('available/' + name, value)
        for name in ('targets', 'reset', 'final', 'answers'): add(name, getattr(batch, name))
        digest.update(repr(batch.identities).encode())
    return digest.hexdigest()


def horizon_update(model, optimizer, rng, streams, protocol):
    parameters = list(model.parameters())
    accumulated = [None for _ in parameters]
    weights = protocol['task_weights']; denominator = sum(weights.values())
    for group in optimizer.param_groups: group['lr'] = protocol['lr']
    row = {'expert_executions': 0, 'source_occurrences': 0, 'external_events': 0,
           'batch_sha256': {}}
    for name in ('composition', 'correction', 'views'):
        optimizer.zero_grad(set_to_none=True)
        if name == 'composition':
            batch = training_batch(rng, protocol['composition_batch'])
            loss, ce, prefix, result = training_loss(model, batch, 2, True, .5, 1.)
            row.update(composition_nats=float(ce.detach()), composition_episodes=batch.size,
                       composition_accuracy=float(result['logits'][:, -1].argmax(-1)
                                                  .eq(batch.answers).float().mean()))
            for key in ('expert_executions', 'source_occurrences', 'external_events'):
                row[key] += prefix[key]
            del ce, prefix
        else:
            stream = streams[name]
            if stream.position not in (0, stream.batch.length):
                raise ValueError('A horizon update must start at an episode boundary.')
            batch = stream.next_chunk(stream.batch.length)
            loss, result = episode_loss(model, batch, protocol['segment'],
                                       protocol['detach_between'],
                                       protocol['fact_supervision']['weight'])
            row[name + '_nats'] = float(result['ce'].detach())
            row[name + '_supervised'] = int(batch.targets.ne(-100).sum())
            row[name + '_final_queries'] = int(batch.final.sum())
            row[name + '_episodes_started'] = int(batch.reset[:, 0].sum())
            for key, value in result.get('retention', {}).items(): row[name + '_' + key] = value
        row['batch_sha256'][name] = batch_digest(batch)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite task loss.')
        loss.backward()
        norm = clip_task_gradients(parameters)
        row[name + '_gradient_norm'] = float(norm)
        for i, parameter in enumerate(parameters):
            if parameter.grad is not None:
                contribution = parameter.grad.detach() * (weights[name] / denominator)
                accumulated[i] = contribution.clone() if accumulated[i] is None else accumulated[i] + contribution
        for key in ('expert_executions', 'source_occurrences', 'external_events'): row[key] += result[key]
        del loss, result
    for parameter, gradient in zip(parameters, accumulated): parameter.grad = gradient
    norm = clip_task_gradients(parameters)
    optimizer.step()
    row.update(gradient_norm=float(norm), lr=protocol['lr'])
    return row
