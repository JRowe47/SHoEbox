"""Cortical Input Map v2, expressed as ordered transport recipes.

Names document intended specialization; neither names nor C0-C10 select code.
Widths are complete learned output ports, never slices of another token.
"""
from dataclasses import asdict, dataclass, field, replace
from experiments.hat_minkron.model import HATConfig


@dataclass(frozen=True)
class Input:
    source: str
    width: int = 4
    count: int = 1
    delayed: bool = False


@dataclass(frozen=True)
class Region:
    name: str
    order: int
    group: str
    inputs: tuple[Input, ...] = ()


def cortical_map():
    def p(source, width=4, count=1):
        return Input(source, width, count)
    def c(source, width=1):
        return Input(source, width, delayed=True)
    return (
        Region('A1', 1, 'Local form'),
        Region('B1', 1, 'Temporal progression'),
        Region('L2', 2, 'Temporal language', (p('A1', count=2), c('L3'))),
        Region('S2', 2, 'ATL/PRC', (p('A1'), p('B1'), c('S3'))),
        Region('R2', 2, 'PPC', (p('A1', count=2), p('B1'), c('R3'))),
        Region('F2', 2, 'Caudal PFC', (p('B1', count=2), p('A1', 2), c('F4'))),
        Region('L3', 3, 'Temporal language', (p('L2', count=2), p('S2', 2), c('F3'), c('EC-C', 2))),
        Region('S3', 3, 'ATL/PRC', (p('S2', count=2), p('L2', 2), c('E4'), c('EC-C', 2))),
        Region('R3', 3, 'PPC', (p('R2', count=2), p('S2'), c('F3'))),
        Region('F3', 3, 'dlPFC', (p('F2', count=2), p('L2'), p('R2'), c('G5'), c('M4'), c('EC-X', 2))),
        Region('E4', 4, 'PMC/AG', (p('S3', count=2), p('R3', count=2), p('L3'), c('G5'), c('EC-X', 2))),
        Region('F4', 4, 'Lateral PFC', (p('F3', count=2), p('R3'), p('L3'), c('G5'), c('M4'))),
        Region('V4', 4, 'OFC/vmPFC', (p('S3', count=2), p('F3'), p('B1', 2), c('G5'))),
        Region('M4', 4, 'ACC/insula', (p('R3'), p('F3'), p('B1', 2), c('F4'))),
        Region('G5', 5, 'FPC', (p('E4', count=2), p('F4', count=2), p('V4'))),
        Region('EC-C', 4, 'EC', (p('L3'), p('S3'), p('S2', 2), c('EC-X'))),
        Region('EC-X', 5, 'EC', (p('E4'), p('F3'), c('G5'), c('M4'), c('EC-C'))),
        Region('Y', 4, 'IFG/premotor', (p('L3', count=2), p('F4'), p('S2', 2), p('EC-C', 2))),
    )


REGIONS = cortical_map()
NAMES = tuple(r.name for r in REGIONS)
SOURCES = NAMES + ('Y-output', 'HC-C', 'HC-X') + tuple(f'external:{i}' for i in range(16))
SOURCE_IDS = {name: i for i, name in enumerate(SOURCES)}
SPINE = ('A1', 'B1', 'S2', 'Y')
# EC is an ordinary paired activation unit, preserving aligned episodic writes.
ROUTING_GROUPS = tuple((name,) for name in NAMES if name not in (*SPINE, 'EC-C', 'EC-X')) + (('EC-C', 'EC-X'),)


@dataclass(frozen=True)
class SkeletonConfig:
    width: int = 16
    experts: int = 48
    expert_hidden: int = 4
    groups: int = 4
    active_groups: int = 1
    experts_per_group: int = 1
    laces: int = 8
    active_laces: int = 2
    address_dim: int = 4
    hat_symbols: int = 16
    hat_levels: int = 1
    hat_active_symbols: int = 2
    kron_heads: int = 1
    kron_key: int = 4
    kron_value: int = 4
    rounds: int = 2
    episode_capacity: int = 32
    episode_reads: int = 2
    episode_address: int = 16
    episode_active: int = 4
    memory_enabled: bool = True
    auxiliary_weight: float = .001
    seed: int = 20261108
    port_scale: int = 1
    terminal_hat: bool = False
    region_overrides: dict = field(default_factory=dict)
    region_topk: int = 0
    edge_topk: int = 2
    routing_dim: int = 32
    routing_noise: float = .5

    def topology(self):
        return tuple(replace(region, inputs=tuple(replace(edge, width=edge.width*self.port_scale)
                                                  for edge in region.inputs)) for region in REGIONS)

    def for_region(self, name):
        if name not in NAMES:
            raise ValueError('Unknown cortical region: '+name)
        return replace(self, **self.region_overrides.get(name, {}), region_overrides={})

    @property
    def max_port_bytes(self):
        return 4*self.port_scale

    def hat_config(self, index=0):
        return HATConfig(width=self.width, experts=self.experts,
                         expert_hidden=self.expert_hidden, groups=self.groups,
                         active_groups=self.active_groups, experts_per_group=self.experts_per_group,
                         laces=self.laces, active_laces=self.active_laces, address_dim=self.address_dim,
                         hat_symbols=self.hat_symbols, hat_active_symbols=self.hat_active_symbols, hat_levels=self.hat_levels,
                         kron_heads=self.kron_heads, kron_key=self.kron_key, kron_value=self.kron_value,
                         chunk_size=32, seed=self.seed+1009*index)

    def validate(self):
        if type(self.region_topk) is not int or not 0 <= self.region_topk <= len(ROUTING_GROUPS):
            raise ValueError('Invalid optional regional activation budget.')
        if type(self.edge_topk) is not int or self.edge_topk < 1 or self.routing_dim < 1 or not 0 <= self.routing_noise <= 5:
            raise ValueError('Invalid regional router configuration.')
        for field in ('width', 'experts', 'expert_hidden', 'groups', 'active_groups',
                      'experts_per_group', 'laces', 'active_laces', 'address_dim',
                      'hat_symbols', 'hat_levels', 'hat_active_symbols', 'kron_heads', 'kron_key', 'kron_value',
                      'rounds', 'episode_capacity', 'episode_reads', 'episode_address', 'episode_active', 'port_scale'):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f'{field} must be a positive integer.')
        if self.rounds > 8:
            raise ValueError('The skeleton bounds recurrent rounds at eight.')
        if not self.episode_reads <= self.episode_capacity:
            raise ValueError('Episode read count exceeds capacity.')
        if self.episode_reads > 19:
            raise ValueError('EC framing supports at most 19 episodic return frames.')
        if not self.episode_active <= self.episode_address:
            raise ValueError('Sparse address exceeds address width.')
        if not 0 <= self.auxiliary_weight <= 1:
            raise ValueError('Invalid auxiliary weight.')
        if self.port_scale > 8 or type(self.terminal_hat) is not bool:
            raise ValueError('Invalid port scale or terminal HAT setting.')
        allowed = {'width', 'experts', 'expert_hidden', 'groups', 'active_groups',
                   'experts_per_group', 'laces', 'active_laces', 'address_dim',
                   'hat_symbols', 'hat_active_symbols', 'hat_levels',
                   'kron_heads', 'kron_key', 'kron_value'}
        if not isinstance(self.region_overrides, dict):
            raise ValueError('Region overrides must be a mapping.')
        for name, values in self.region_overrides.items():
            if name not in NAMES or not isinstance(values, dict) or set(values)-allowed:
                raise ValueError('Invalid regional capacity override.')
            self.for_region(name).validate()
        try:
            self.hat_config().validate()
        except AssertionError as exc:
            raise ValueError('Invalid sparse region dimensions.') from exc
        seen = set()
        for region in REGIONS:
            for edge in region.inputs:
                if edge.source not in NAMES or edge.width not in (1, 2, 4) or edge.count not in (1, 2):
                    raise ValueError('Invalid frame recipe.')
                if not edge.delayed and edge.source not in seen:
                    raise ValueError('Current-round backbone must be acyclic.')
            seen.add(region.name)

    def to_dict(self):
        return asdict(self)


def cortical_10m(**overrides):
    """9,982,290 parameters; the skeleton's exact 18-region wiring and ports.

    Capacity grows inside each ordinary region. No extra semantic controllers,
    regional archives, sharing, or changes to recurrence scheduling are added.
    """
    values = dict(width=64, experts=192, expert_hidden=16, groups=8,
                  laces=32, address_dim=16, hat_symbols=64,
                  kron_heads=2, kron_key=8, kron_value=8,
                  episode_address=64, episode_active=8)
    values.update(overrides)
    return SkeletonConfig(**values)


def cortical_wide(**overrides):
    """Wider transport and active paths, with extra capacity at orders 1 and 2.

    Ports retain three independent learned code widths: 1/2/4 -> 4/8/16.
    External bytes and Y's externally emitted byte remain literal bytes.
    A 16-byte code is one publication, not sixteen observations of history.
    """
    regional = {}
    for name in ('A1', 'B1'):
        regional[name] = dict(width=128, expert_hidden=32, hat_symbols=256,
                              hat_active_symbols=8, kron_heads=4, address_dim=32,
                              laces=64, active_laces=4)
    for name in ('L2', 'S2', 'R2', 'F2'):
        regional[name] = dict(width=96, expert_hidden=24, hat_symbols=256,
                              hat_active_symbols=8, kron_heads=3, address_dim=24,
                              laces=48, active_laces=4)
    values = dict(width=64, experts=96, expert_hidden=16, groups=8,
                  active_groups=2, experts_per_group=2, laces=32, active_laces=4,
                  address_dim=16, hat_symbols=128, hat_active_symbols=4,
                  kron_heads=2, kron_key=8, kron_value=8,
                  episode_address=64, episode_active=8, port_scale=4,
                  terminal_hat=True, region_overrides=regional)
    values.update(overrides)
    return SkeletonConfig(**values)


def cortical_cpu(**overrides):
    """Sparse regional execution, complete wide ports and a live ingress spine."""
    values = dict(region_topk=3, edge_topk=2)
    values.update(overrides)
    return cortical_wide(**values)
