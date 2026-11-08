"""Bounded, framed publications. Two occurrences are transport, not an archive."""
from dataclasses import dataclass, fields, is_dataclass, replace
import torch
from torch.nn import functional as F


def map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if is_dataclass(value):
        return replace(value, **{f.name: map_tensors(getattr(value, f.name), function) for f in fields(value)})
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(v, function) for v in value)
    return value


def detach_state(value, compact=False):
    """Drop hard one-hot transport buffers only at an explicit TBPTT boundary.

    Inside the live window, straight-through distributions retain their graph.
    Once detached, integer bytes carry the exact same forward information.
    """
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, Frame) and compact:
        return replace(value, **{f.name: None if f.name in ('distribution', 'soft') else
                                  detach_state(getattr(value, f.name), compact)
                                  for f in fields(value)})
    if is_dataclass(value):
        return replace(value, **{f.name: detach_state(getattr(value, f.name), compact) for f in fields(value)})
    if isinstance(value, dict):
        return {k: detach_state(v, compact) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_state(v, compact) for v in value)
    return value


@dataclass(frozen=True)
class Frame:
    source: str
    event: int
    revision: int
    values: torch.Tensor
    distribution: torch.Tensor | None = None
    strength: torch.Tensor | None = None
    origin_source: str = ''
    origin_event: int = -1
    origin_revision: int = -1
    episode_events: torch.Tensor | None = None
    episode_revisions: torch.Tensor | None = None
    soft: torch.Tensor | None = None
    valid: torch.Tensor | None = None

    @property
    def width(self):
        return self.values.shape[1]

    def detach(self):
        return map_tensors(self, lambda t: t.detach())

    def trace(self):
        scalar = lambda v: v.detach().cpu().tolist() if torch.is_tensor(v) else v
        return {'source': self.source, 'event': scalar(self.event), 'revision': scalar(self.revision),
                'width': self.width, 'values': self.values.detach().cpu().tolist(),
                'origin_source': self.origin_source or self.source,
                'origin_event': (self.episode_events.cpu().tolist() if self.episode_events is not None else
                                 self.origin_event if self.origin_event >= 0 else scalar(self.event)),
                'origin_revision': (self.episode_revisions.cpu().tolist() if self.episode_revisions is not None else
                                    self.origin_revision if self.origin_revision >= 0 else scalar(self.revision)),
                'episode_events': None if self.episode_events is None else self.episode_events.cpu().tolist()}


def byte_frame(source, event, revision, logits, probabilities=None, *, compact=False):
    if not torch.is_grad_enabled():
        return Frame(source, event, revision, logits.argmax(-1))
    probabilities = logits.softmax(-1) if probabilities is None else probabilities
    values = probabilities.argmax(-1)
    if compact:
        return Frame(source, event, revision, values, soft=probabilities)
    hard = F.one_hot(values, 256).to(probabilities.dtype)
    # Forward transport contains exact bytes; backward connects producer and HAT.
    straight_through = hard + (probabilities - probabilities.detach())
    return Frame(source, event, revision, values, straight_through)


def publish(ports, source, frames):
    """Return a new snapshot; revisions replace the current occurrence."""
    result = dict(ports)
    for width, frame in frames.items():
        key = f'{source}:{width}'
        history = result.get(key, ())
        if history and history[-1].event == frame.event:
            history = history[:-1]
        result[key] = (*history, frame)[-2:]
    return result


def recipe_frames(spec, current, snapshot):
    result = []
    for edge in spec.inputs:
        bank = snapshot if edge.delayed else current
        # Startup uses only available complete frames; no invented zero tokens.
        result.extend(bank.get(f'{edge.source}:{edge.width}', ())[-edge.count:])
    return result
