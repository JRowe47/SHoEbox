"""Task-separated gradients and bounded contributions to shared subsystems.

No extra forward or backward pass is needed: text and fresh memory gradients
are captured separately, measured, then combined for one optimizer step.
"""
import math
import torch
from . import byte_stream
from .joint_text import chunk_digest
from .multifact import backward, COUNTERS


def gradients(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


def norm(values):
    # Accumulate in FP64 so large but finite gradients remain measurable.
    return math.sqrt(sum(float(v.double().square().sum()) for v in values if v is not None))


def statistics(left, right):
    a, b = norm(left), norm(right)
    dot = sum(float((x.double()*y.double()).sum()) for x, y in zip(left, right)
              if x is not None and y is not None)
    return dict(text_norm=a, memory_norm=b, dot=dot,
                cosine=dot/(a*b) if a and b else None)


def combine(parameters, text, memory, groups, mode='sum', cap=1.):
    """Bound each task within each optimizer group, preserving sparse absence.

    `sum` is the matched unmodified-objective control. `group_clip` caps each
    task/group at `cap` before addition. The caller applies the common global
    norm cap after this step. No gradient direction is projected or reversed.
    """
    if mode not in ('sum', 'group_clip'):
        raise ValueError('Unknown gradient combination mode.')
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('Finite positive gradient cap required.')
    if len(parameters) != len(text) or len(text) != len(memory):
        raise ValueError('Gradient lists must align with parameters.')
    indices = [i for values in groups.values() for i in values]
    if sorted(indices) != list(range(len(parameters))):
        raise ValueError('Groups must partition the parameters exactly once.')
    if any(v is not None and (v.shape != p.shape or not bool(torch.isfinite(v).all()))
           for values in (text, memory) for p, v in zip(parameters, values)):
        raise FloatingPointError('Nonfinite or misaligned task gradients.')
    report = {}
    for name, indices in groups.items():
        row = statistics([text[i] for i in indices], [memory[i] for i in indices])
        scales = [min(1., cap/max(row[key], 1e-300)) if mode == 'group_clip' else 1.
                  for key in ('text_norm', 'memory_norm')]
        row.update(text_scale=scales[0], memory_scale=scales[1])
        report[name] = row
        for i in indices:
            a, b = text[i], memory[i]
            if a is None and b is None:
                parameters[i].grad = None
            elif a is None:
                parameters[i].grad = b*scales[1]
            elif b is None:
                parameters[i].grad = a*scales[0]
            else:
                parameters[i].grad = a*scales[0] + b*scales[1]
    return report


def subsystem(name):
    if name.startswith('nodes.'):
        return '.'.join(name.split('.')[:2])
    if name.startswith(('independent_hats.', 'independent_banks.')):
        return '.'.join(name.split('.')[:2])
    return name.split('.')[0]


def update(model, optimizer, chunk, state, batch, tables, reference,
           mode='group_clip', cap=1., memory_weight=1.):
    parameters = [p for p in model.parameters() if p.requires_grad]
    positions = {id(p): i for i, p in enumerate(parameters)}
    groups = {g['name']: [positions[id(p)] for p in g['params']] for g in optimizer.param_groups}
    optimizer.zero_grad(set_to_none=True)
    result = byte_stream.forward(model, chunk.values, chunk.targets, state, chunk.reset)
    result['loss'].backward()
    state = result['state'].detach()
    row = dict(text_loss=float(result['loss'].detach()), text_nats=float(result['ce'].detach()),
               text_bytes=chunk.values.numel(), text_batch_sha256=chunk_digest(chunk),
               **{k: result[k] for k in COUNTERS})
    del result
    text = gradients(parameters)
    optimizer.zero_grad(set_to_none=True)
    memory_row = backward(model, batch, tables, memory_weight, reference)
    memory = gradients(parameters)
    sections = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            sections.setdefault(subsystem(name), []).append(positions[id(p)])
    row['subsystems'] = {name: statistics([text[i] for i in ids], [memory[i] for i in ids])
                         for name, ids in sections.items()}
    row['gradient_groups'] = combine(parameters, text, memory, groups, mode, cap)
    row['gradient_norm'] = norm([p.grad for p in parameters])
    # Stable even when FP32 squared norms overflow. Nonfinite inputs were
    # rejected by combine before it wrote any optimizer gradient.
    scale = min(1., 1./(row['gradient_norm']+1e-6))
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)
    optimizer.step()
    for k in COUNTERS:
        row[k] += memory_row[k]
    row.update(memory=memory_row, global_clip_scale=scale)
    return state, row
