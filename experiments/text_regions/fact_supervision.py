"""Training-only local decoding of facts that a region can actually observe."""
import torch
from torch import nn
from torch.nn import functional as F

from .view_data import QUERY_TYPES, FACTOR_SOURCES

HEADS = ('memory_retention', 'fact_retention')


def attach_heads(model, optimizer=None, seed=20261108):
    """Add matched auxiliary heads without advancing the parent's Torch RNG.

    Appending to each original AdamW group preserves its existing moments and
    the parameter order produced by optimizer_for when loading a checkpoint.
    """
    if any(hasattr(model, name) for name in HEADS):
        raise ValueError('Fact heads are already attached.')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.memory_retention = nn.Linear(model.cfg.width, 4)
        model.fact_retention = nn.Linear(model.cfg.width, 12)
    if optimizer is not None:
        if len(optimizer.param_groups) != 2:
            raise ValueError('Expected the existing matrix/bias AdamW groups.')
        for name in HEADS:
            head = getattr(model, name)
            optimizer.param_groups[0]['params'].append(head.weight)
            optimizer.param_groups[1]['params'].append(head.bias)


def forward_sources(config):
    """Conservative eligibility: only the configured forward ancestors.

    Delayed feedback may expose additional facts; it never expands these
    auxiliary targets. Inference does not use this bookkeeping.
    """
    sources = {}
    specs = {n.name: n for n in config.nodes}
    for stage in config.stages():
        for name in stage:
            own = specs[name].view
            reachable = {own} if own is not None else set()
            for edge in config.edges:
                if edge.destination == name and edge.kind == 'forward':
                    reachable.update(sources[edge.source])
            sources[name] = reachable
    return sources


def view_targets(chunk, sources):
    """Causal retained-factor targets [B,T,3], including across chunk slices.

    observation_positions already excludes withheld observations. Resets in
    this slice invalidate earlier observations, and padding does not erase a
    fact that was seen earlier. Hidden factors receive no gradient.
    """
    targets = torch.full((*chunk.targets.shape, 3), -100,
                         dtype=torch.long, device=chunk.targets.device)
    time = torch.arange(chunk.time_offset, chunk.time_offset+chunk.length,
                        device=targets.device)[None].expand_as(chunk.targets)
    if chunk.last_reset_positions is None:
        raise ValueError('Fact batches require causal reset-position metadata.')
    last_reset = chunk.last_reset_positions
    for axis, factor in enumerate(QUERY_TYPES):
        known = torch.zeros_like(chunk.reset)
        for view in sorted(set(sources) & set(FACTOR_SOURCES[factor])):
            observed = chunk.observation_positions[view][factor][:, None]
            known |= observed.ge(0) & observed.le(time) & observed.ge(last_reset)
        targets[:, :, axis] = torch.where(known, chunk.factors[factor][:, None], -100)
    return targets


def retention_loss(model, chunk, activity):
    """Mean over supported node/factor groups, then mean within each group."""
    losses, correct, count = [], 0, 0
    sources = forward_sources(model.cfg)
    for name, record in activity.items():
        if isinstance(chunk.inputs, dict):
            targets = view_targets(chunk, sources[name])
            logits = model.fact_retention(record['payload']).reshape(*targets.shape, 4)
        else:
            if chunk.retained_targets is None:
                raise ValueError('Memory batches require causal retained targets.')
            targets = chunk.retained_targets[:, :, None]
            logits = model.memory_retention(record['payload'])[:, :, None, :]
        for axis in range(targets.shape[-1]):
            labels = targets[:, :, axis]
            mask = labels.ne(-100) & record['valid']
            if bool(mask.any()):
                selected, truth = logits[:, :, axis][mask], labels[mask]-65
                losses.append(F.cross_entropy(selected, truth))
                correct += int(selected.detach().argmax(-1).eq(truth).sum())
                count += int(mask.sum())
    # The zero path is differentiable without introducing any target gradient.
    loss = torch.stack(losses).mean() if losses else model.readout.weight.sum()*0.
    return loss, {'local_labels': count, 'local_correct': correct,
                  'local_groups': len(losses), 'local_nats': float(loss.detach())}
