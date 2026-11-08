"""Typed transport and immutable, owner-local memory. IDs contain no task labels.

Sparse records are allocated on demand. A row is one isolated world; tensors in
that row have no batch axis. The enclosing cue trace has a leading batch axis.
Forking copies containers but shares live history tensors until a selected write.
"""
from dataclasses import dataclass, field, replace
import torch
from torch import nn
from torch.nn import functional as F
from .ports import map_tensors


@dataclass(frozen=True)
class EngramHandle:
    namespace: int
    region: str
    node: str
    slot: int
    generation: int
    version: int
    semantic_version: int


@dataclass(frozen=True)
class BindingHandle:
    namespace: int
    shard: int
    slot: int
    generation: int
    version: int


@dataclass(frozen=True)
class LocalAnchor:
    handle: EngramHandle
    capsule: torch.Tensor
    key: torch.Tensor
    role: torch.Tensor
    provenance: str
    ready_round: int
    source_version: int


@dataclass(frozen=True)
class AnchorLink:
    handle: EngramHandle
    cue: torch.Tensor
    role: torch.Tensor
    strength: torch.Tensor


@dataclass(frozen=True)
class BindingRecord:
    handle: BindingHandle
    key: torch.Tensor
    code: torch.Tensor
    context: torch.Tensor
    links: tuple[AnchorLink, ...]
    neighbors: tuple[BindingHandle, ...]
    utility: torch.Tensor
    confidence: torch.Tensor
    created: int
    revised: int
    representation_version: int


@dataclass(frozen=True)
class CueRequest:
    request_id: tuple
    namespace: int
    requester_region: str
    requester_node: str
    cue: torch.Tensor
    context: torch.Tensor
    watermark: int


@dataclass(frozen=True)
class BindingOffer:
    offer_id: tuple
    namespace: int
    producer: str
    anchor: LocalAnchor
    key: torch.Tensor
    code: torch.Tensor
    context: torch.Tensor
    write: torch.Tensor
    erase: torch.Tensor
    actions: torch.Tensor  # reinforce / revise / separate / no-write
    ready_round: int
    provenance: str = 'external'


@dataclass(frozen=True)
class ReinstatementRequest:
    request_id: tuple
    binding: BindingHandle
    handle: EngramHandle
    original_cue: torch.Tensor
    local_cue: torch.Tensor
    role: torch.Tensor
    watermark: int


@dataclass(frozen=True)
class CreditReceipt:
    identity: tuple
    request_id: tuple
    contributors: tuple[tuple[BindingHandle, EngramHandle], ...]
    support: torch.Tensor
    window: int
    kind: str = 'live_autograd'  # No invented historical content derivative.


@dataclass(frozen=True)
class RecallResult:
    evidence: torch.Tensor
    contributors: tuple
    status: str
    receipts: tuple[CreditReceipt, ...] = ()


@dataclass(frozen=True)
class MemoryRow:
    namespace: int = 0
    anchors: dict = field(default_factory=dict)  # region -> slot -> LocalAnchor
    bindings: dict = field(default_factory=dict)  # shard -> slot -> BindingRecord
    postings: dict = field(default_factory=dict)  # (shard, DG-cell) -> binding handles
    recent: dict = field(default_factory=dict)  # bounded bootstrap posting lists
    anchor_counters: dict = field(default_factory=dict)
    binding_counters: dict = field(default_factory=dict)
    committed: dict = field(default_factory=dict)  # producer high-watermarks
    credited: dict = field(default_factory=dict)  # requester high-watermarks
    round: int = 0

    def anchor(self, handle, watermark):
        if handle.namespace != self.namespace:
            return None
        found = self.anchors.get(handle.region, {}).get(handle.slot)
        return found if found is not None and found.handle == handle and found.ready_round <= watermark else None

    def binding(self, handle, watermark):
        if handle.namespace != self.namespace:
            return None
        found = self.bindings.get(handle.shard, {}).get(handle.slot)
        return found if found is not None and found.handle == handle and found.revised <= watermark else None


@dataclass(frozen=True)
class HippocampusState:
    rows: tuple[MemoryRow, ...]
    cues: torch.Tensor
    schema: int = 1
    window: int = 0

    def fork(self, indices):
        # Snapshot containers are never mutated, even when two branches share a row.
        return replace(self, rows=tuple(self.rows[i] for i in indices.tolist()),
                       cues=self.cues.index_select(0, indices))

    def reset(self, mask):
        return replace(self, rows=tuple(MemoryRow(namespace=row.namespace+1) if reset else row
                                      for row, reset in zip(self.rows, mask.tolist())),
                       cues=torch.where(mask[:, None, None], 0., self.cues))

    @classmethod
    def from_dict(cls, data):
        if data['schema'] != 1:
            raise ValueError('Unsupported hippocampus state schema.')
        rows = []
        for raw in data['rows']:
            raw = dict(raw)
            raw['anchors'] = {r:{s:LocalAnchor(**(dict(a) | {'handle':EngramHandle(**a['handle'])}))
                                  for s,a in slots.items()} for r,slots in raw['anchors'].items()}
            raw['bindings'] = {r:{s:BindingRecord(**(dict(a) | {
                'handle':BindingHandle(**a['handle']),
                'links':tuple(AnchorLink(**(dict(link) | {'handle':EngramHandle(**link['handle'])})) for link in a['links']),
                'neighbors':tuple(BindingHandle(**h) for h in a['neighbors'])}))
                for s,a in slots.items()} for r,slots in raw['bindings'].items()}
            raw['postings'] = {k:tuple(BindingHandle(**h) for h in v) for k,v in raw['postings'].items()}
            raw['recent'] = {k:tuple(BindingHandle(**h) for h in v) for k,v in raw['recent'].items()}
            rows.append(MemoryRow(**raw))
        return cls(tuple(rows), data['cues'], data['schema'], data['window'])


class RegionalInterface(nn.Module):
    """One shared trunk replicated by protocol, with one cue trace per region.

    The GRU only consumes current ingress and its own previous cue trace. Neither
    candidates nor recalled values enter the initial request. Experts interpret
    returned capsules through their own conditional reader hooks.
    """
    def __init__(self, width, config):
        super().__init__()
        d = config.address_width
        self.trace = nn.GRUCell(width, width)
        self.address = nn.Linear(width*2, d, bias=False)
        self.offer_delta = nn.Linear(width, d, bias=False)
        self.role = nn.Linear(width, d, bias=False)
        self.admission = nn.Linear(width*2, 1)
        self.erase = nn.Linear(width*2, 1)
        self.actions = nn.Linear(d*3, 4)
        nn.init.normal_(self.offer_delta.weight, std=.01)
        nn.init.constant_(self.admission.bias, .2)
        nn.init.constant_(self.erase.bias, -1.)
        nn.init.zeros_(self.actions.weight)
        with torch.no_grad():
            self.actions.bias.copy_(torch.tensor([-1., -1., 1., -2.]))

    def cue(self, ingress, previous):
        trace = self.trace(ingress, previous)
        return trace, F.normalize(self.address(torch.cat((trace, ingress), -1)), dim=-1)

    def offer(self, trace, latent, cue):
        pair = torch.cat((trace, latent), -1)
        key = F.normalize(cue + self.offer_delta(latent), dim=-1)
        return key, self.role(latent).tanh(), self.admission(pair).sigmoid(), self.erase(pair).sigmoid()
