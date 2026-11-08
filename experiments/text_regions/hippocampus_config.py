"""Independent capacity, work and intervention controls for shared memory."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HippocampusConfig:
    enabled: bool = False
    mode: str = 'shared'  # 'local' is comparison B, with no associative directory.
    cue_clock: str = 'round'  # Legacy checkpoints preserve their original clock.
    reinstatement: str = 'ingress'  # 'expert' enters after the regional HAT adapter.
    address_width: int = 32
    dg_cells: int = 256
    dg_active: int = 8
    shards: int = 2
    bindings_per_shard: int = 128
    anchors_per_region: int = 64
    shards_per_request: int = 2
    posting_reads: int = 64
    candidates_per_shard: int = 16
    selected_bindings: int = 2
    neighbors: int = 8
    links_per_binding: int = 4
    readbacks: int = 8
    refinement_steps: int = 2
    queue_capacity: int = 64
    service_budget: int = 16
    deadline_rounds: int = 2
    representation_version: int = 1
    maintenance_budget: int = 2
    exploration_interval: int = 16
    bootstrap: bool = True
    admission: bool = True
    revision: bool = True
    retention: bool = True
    feedback: bool = True
    read: bool = True
    control: str = 'intact'

    def validate(self):
        if self.mode not in ('shared', 'local'):
            raise ValueError('Hippocampus mode must be shared or local.')
        if self.cue_clock not in ('round', 'event'):
            raise ValueError('Cue clock must be round or event.')
        if self.reinstatement not in ('ingress', 'expert'):
            raise ValueError('Reinstatement must enter ingress or expert input.')
        if self.control not in ('intact', 'blocked', 'shuffled', 'pooled', 'no_refinement', 'no_feedback'):
            raise ValueError('Unknown hippocampus intervention.')
        for name in ('address_width', 'dg_cells', 'dg_active', 'shards', 'bindings_per_shard',
                     'anchors_per_region', 'shards_per_request', 'posting_reads', 'candidates_per_shard',
                     'selected_bindings', 'links_per_binding', 'readbacks', 'queue_capacity',
                     'service_budget', 'deadline_rounds', 'representation_version', 'exploration_interval'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.shards & (self.shards-1) or self.shards > 32:
            raise ValueError('Reference directory supports power-of-two shard counts through 32.')
        if not 0 <= self.refinement_steps <= 2 or not 0 <= self.neighbors <= 8:
            raise ValueError('Association depth/degree exceeds the v1 contract.')
        if self.maintenance_budget < 0 or self.dg_active > self.dg_cells or self.shards_per_request > self.shards:
            raise ValueError('Invalid capacity/work budget.')
        if self.links_per_binding > 4 or self.readbacks > 8 or self.shards_per_request > 2:
            raise ValueError('Selected work exceeds the v1 contract.')
