"""Serializable topology and capacity. External depth is independent of SHoE depth."""
from dataclasses import asdict, dataclass, field
from ..hat_minkron.model import HATConfig

BASELINE_REVISION = 'b4bb7cfea18c3833819f6ed08973fd014f15f107'


@dataclass(frozen=True)
class NodeSpec:
    name: str
    region: str
    view: str | None = None


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    destination: str
    kind: str = 'forward'

    @property
    def key(self):
        return f'{self.source}__{self.destination}__{self.kind}'


def default_nodes():
    return tuple(NodeSpec(n, r, v) for n, r, v in (
        ('A', 'R0', 'A'), ('B', 'R0', 'B'), ('C', 'R0', 'C'),
        ('U', 'R1', None), ('P', 'R1', None), ('V', 'R2', None), ('Q', 'R2', None)))


def default_edges():
    forward = [('A', 'U'), ('B', 'U'), ('B', 'V'), ('C', 'V'),
               ('A', 'P'), ('U', 'P'), ('V', 'Q'), ('C', 'Q')]
    lateral = [('P', 'Q'), ('Q', 'P')]
    descending = [('P', 'U'), ('P', 'A'), ('P', 'B'),
                  ('Q', 'V'), ('Q', 'B'), ('Q', 'C')]
    return tuple(EdgeSpec(a, b, kind) for kind, pairs in (
        ('forward', forward), ('lateral', lateral), ('descending', descending)) for a, b in pairs)


@dataclass(frozen=True)
class GraphConfig:
    nodes: tuple[NodeSpec, ...] = field(default_factory=default_nodes)
    edges: tuple[EdgeSpec, ...] = field(default_factory=default_edges)
    readout: tuple[str, ...] = ('P', 'Q')
    width: int = 100
    experts_per_block: int = 128
    shared_per_block: int = 64
    expert_hidden: int = 4
    groups: int = 16
    active_groups: int = 2
    experts_per_group: int = 4
    address_dim: int = 16
    laces: int = 64
    active_laces: int = 2
    hat_levels: int = 2
    hat_symbols: int = 64
    hat_active_symbols: int = 2
    kron_heads: int = 4
    kron_key: int = 8
    kron_value: int = 8
    rounds: int = 1
    recurrent_feedback: bool = False
    feedback_gate_init: float = .05
    zero_safe_norm: bool = True
    context: int = 128
    auxiliary_loss_weight: float = .01
    hat_balance_weight: float = .001
    seed: int = 20261104

    @property
    def views(self):
        return tuple(dict.fromkeys(n.view for n in self.nodes if n.view is not None))

    @property
    def regions(self):
        return {r: tuple(n.name for n in self.nodes if n.region == r)
                for r in dict.fromkeys(n.region for n in self.nodes)}

    def stages(self):
        remaining = {n.name for n in self.nodes}
        done, stages = set(), []
        while remaining:
            stage = tuple(sorted(n for n in remaining if all(
                e.source in done for e in self.edges if e.destination == n and e.kind == 'forward')))
            if not stage:
                raise ValueError('Forward edges must form a DAG; delayed edges may form cycles.')
            stages.append(stage)
            done.update(stage)
            remaining.difference_update(stage)
        return tuple(stages)

    def validate(self):
        names = [n.name for n in self.nodes]
        if not names or len(set(names)) != len(names):
            raise ValueError('Each node must have exactly one state owner and a unique name.')
        for n in self.nodes:
            if not n.name.isidentifier() or '.' in n.name or '__' in n.name or not n.region:
                raise ValueError('Node names must be identifiers without double underscores; region is required.')
            if n.view is not None and (not n.view.isidentifier() or n.view == 'text'):
                raise ValueError('View names must be identifiers other than the reserved text stream.')
        if not self.views or not self.readout or len(set(self.readout)) != len(self.readout):
            raise ValueError('At least one source view and distinct readout nodes are required.')
        if any(n not in names for n in self.readout):
            raise ValueError('Readout references an unknown node.')
        if len(set(e.key for e in self.edges)) != len(self.edges):
            raise ValueError('Duplicate edge.')
        for e in self.edges:
            if e.source not in names or e.destination not in names:
                raise ValueError('Edge references an unknown node.')
            if e.kind not in ('forward', 'lateral', 'descending'):
                raise ValueError('Unknown edge kind.')
        if self.rounds < 1 or self.context < 1 or not 0 < self.shared_per_block < self.experts_per_block:
            raise ValueError('Invalid rounds, segment length, or sharing layout.')
        if not 0 < self.feedback_gate_init < 1:
            raise ValueError('Initial feedback gate must be strictly between zero and one.')
        if min(self.width, self.groups, self.expert_hidden, self.kron_heads, self.kron_key, self.kron_value) < 1:
            raise ValueError('Dimensions must be positive.')
        if self.width % 4 or self.experts_per_block // self.groups < self.experts_per_group:
            raise ValueError('Width must be divisible by four and each group must contain enough experts.')
        self.hat_config().validate()
        self.stages()

    def hat_config(self):
        return HATConfig(width=self.width, experts=3*self.experts_per_block,
                         expert_hidden=self.expert_hidden, groups=self.groups,
                         active_groups=self.active_groups, experts_per_group=self.experts_per_group,
                         address_dim=self.address_dim, laces=self.laces, active_laces=self.active_laces,
                         hat_levels=self.hat_levels, hat_symbols=self.hat_symbols,
                         hat_active_symbols=self.hat_active_symbols, kron_heads=self.kron_heads,
                         kron_key=self.kron_key, kron_value=self.kron_value,
                         context=max(16, self.context), seed=self.seed)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['nodes'] = tuple(NodeSpec(**n) for n in data['nodes'])
        data['edges'] = tuple(EdgeSpec(**e) for e in data['edges'])
        data['readout'] = tuple(data['readout'])
        return cls(**data)

    @classmethod
    def tiny(cls, **kwargs):
        values = dict(width=32, experts_per_block=32, shared_per_block=16, groups=4,
                      experts_per_group=2, address_dim=8, laces=16, hat_symbols=16,
                      kron_heads=2, kron_key=4, kron_value=4, context=16)
        values.update(kwargs)
        return cls(**values)
