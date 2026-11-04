"""Immutable publication snapshots. Dependencies are IDs, never historical payloads."""
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Mapping
import torch


def map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if is_dataclass(value):
        return replace(value, **{f.name: map_tensors(getattr(value, f.name), function) for f in fields(value)})
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(map_tensors(v, function) for v in value)
    return value


def tensor_list(value):
    result = []
    map_tensors(value, lambda x: result.append(x) or x)
    return result


def plain_dict(value):
    if is_dataclass(value):
        return {f.name: plain_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: plain_dict(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(plain_dict(v) for v in value)
    return value


def choose_rows(mask, new, old):
    return torch.where(mask.reshape(-1, *([1]*(new.ndim-1))), new, old)


@dataclass(frozen=True)
class SemanticHeader:
    namespace: str
    memberships: torch.Tensor  # [B, 3, groups], differentiable local coordinates
    group_ids: torch.Tensor
    lace_ids: torch.Tensor

    @property
    def block_namespaces(self):
        return tuple((self.namespace, block) for block in range(self.memberships.shape[1]))


@dataclass(frozen=True)
class Dependency:
    edge: str
    source: str
    port: str
    version: torch.Tensor
    epoch: torch.Tensor
    occurrence: torch.Tensor
    round_index: torch.Tensor
    consumed: torch.Tensor
    intervention: str = 'none'
    substituted: torch.Tensor | None = None


@dataclass(frozen=True)
class Publication:
    source: str
    port: str
    payload: torch.Tensor
    valid: torch.Tensor
    version: torch.Tensor
    epoch: torch.Tensor
    occurrence: torch.Tensor
    round_index: torch.Tensor
    header: SemanticHeader
    dependencies: tuple[Dependency, ...] = ()
    links: torch.Tensor | None = None
    slots: torch.Tensor | None = None

    def dependency(self, edge, consumed):
        return Dependency(edge, self.source, self.port, self.version, self.epoch,
                          self.occurrence, self.round_index, consumed & self.valid,
                          substituted=torch.zeros_like(consumed))

    def detach(self):
        return map_tensors(self, lambda x: x.detach())

    def reset(self, rows):
        cleared = map_tensors(self, lambda x: choose_rows(rows, torch.zeros_like(x), x))
        return replace(cleared, dependencies=tuple(normalize_dependency(d) for d in cleared.dependencies))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['header'] = SemanticHeader(**data['header'])
        dependencies = []
        for d in data['dependencies']:
            d = dict(d)
            if d.get('substituted') is None:
                d['substituted'] = torch.full_like(d['consumed'], d.get('intervention') == 'substitution')
            dependencies.append(Dependency(**d))
        data['dependencies'] = tuple(dependencies)
        return cls(**data)


@dataclass(frozen=True)
class TensorPorts:
    latest: Mapping[str, Publication]

    def snapshot(self):
        return dict(self.latest)

    def stage(self, staged, publication):
        if publication.source in staged:
            raise ValueError('A stage may publish each node only once.')
        if publication.port != 'latent':
            raise ValueError('The reference runtime has one latent output port per node.')
        staged[publication.source] = publication

    def commit(self, staged):
        return TensorPorts({**self.latest, **staged})

    def invalidate(self, source, rows):
        p = self.latest[source]
        return self.commit({source: replace(p, valid=p.valid & ~rows)})

    def reset(self, rows):
        return TensorPorts({k: p.reset(rows) for k, p in self.latest.items()})

    def detach(self):
        return TensorPorts({k: p.detach() for k, p in self.latest.items()})


def publish(previous, payload, header, active, epoch, occurrence, round_index, dependencies, links=None, slots=None):
    """Functional row replacement; inactive rows keep both state and publication."""
    old_deps = {d.edge: normalize_dependency(d) for d in previous.dependencies}
    new_deps = {d.edge: normalize_dependency(d) for d in dependencies}
    merged = []
    for key in sorted(old_deps.keys() | new_deps.keys()):
        template = new_deps.get(key, old_deps.get(key))
        empty = map_tensors(template, torch.zeros_like)
        old, new = old_deps.get(key, empty), new_deps.get(key, empty)
        item = replace(template, **{f.name: choose_rows(active, getattr(new, f.name), getattr(old, f.name))
                                    for f in fields(template) if torch.is_tensor(getattr(template, f.name))})
        if item.substituted is not None:
            label = 'substitution' if bool(item.substituted.all()) else ('mixed' if bool(item.substituted.any()) else 'none')
            item = replace(item, intervention=label)
        merged.append(item)
    return Publication(previous.source, previous.port, choose_rows(active, payload, previous.payload),
                       previous.valid | active, previous.version + active.long(),
                       choose_rows(active, epoch, previous.epoch),
                       choose_rows(active, occurrence, previous.occurrence),
                       choose_rows(active, round_index, previous.round_index),
                       SemanticHeader(header.namespace,
                                      choose_rows(active, header.memberships, previous.header.memberships),
                                      choose_rows(active, header.group_ids, previous.header.group_ids),
                                      choose_rows(active, header.lace_ids, previous.header.lace_ids)), tuple(merged),
                       links.clone() if links is not None else previous.links,
                       slots.clone() if slots is not None else previous.slots)


def normalize_dependency(dependency):
    """Keep the summary label consistent with authoritative per-row identity."""
    rows = dependency.substituted
    if rows is None:
        rows = torch.full_like(dependency.consumed, dependency.intervention == 'substitution')
    label = 'substitution' if bool(rows.all()) else ('mixed' if bool(rows.any()) else 'none')
    return replace(dependency, substituted=rows, intervention=label)
