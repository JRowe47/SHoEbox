"""The predictive six-region pilot; the full cortical map remains in config.py."""
from dataclasses import asdict, dataclass
import math
from experiments.hat_minkron.model import HATConfig


PILOT_NAMES = ('A1', 'B1', 'L2', 'S2', 'L3', 'Y')
CADENCES = dict(A1=1, B1=1, L2=4, S2=4, L3=16, Y=1)


@dataclass(frozen=True)
class ObjectiveConfig:
    """Training-only protocol. Kept outside the inference/state signature."""
    mode: str = 'legacy'
    version: int = 1
    l2_weight: float = .05
    s2_weight: float = .05
    l3_weight: float = .05
    variance_weight: float = .01
    covariance_weight: float = .001
    target_std: float = .1
    normalizer_variance_floor: float = .01
    normalizer_rate: float = .05
    head_seed: int = 20261109

    def __post_init__(self):
        if self.mode not in ('legacy', 'latent') or self.version != 1:
            raise ValueError('Unknown regional objective protocol.')
        for name, value in asdict(self).items():
            if name in ('mode', 'version', 'head_seed'):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid objective {name}.')
            if not name.endswith('_weight') and value == 0:
                raise ValueError(f'Objective {name} must be positive.')
        if type(self.head_seed) is not int or self.head_seed < 0:
            raise ValueError('Invalid head seed.')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class HierarchyConfig:
    message_width: int = 64
    local_width: int = 64
    mid_width: int = 96
    high_width: int = 128
    decoder_width: int = 48
    experts_per_layer: int = 32
    active_groups: int = 2
    experts_per_group: int = 2
    groups: int = 4
    expert_hidden: int = 16
    hat_symbols: int = 64
    hat_active_symbols: int = 4
    address_dim: int = 16
    laces: int = 16
    active_laces: int = 4
    kron_heads: int = 2
    kron_key: int = 8
    kron_value: int = 8
    mid_loss_weight: float = .2
    high_loss_weight: float = .1
    expert_balance_weight: float = .001
    local_only: bool = False
    top_down: bool = True
    seed: int = 20261108

    @property
    def names(self):
        return ('A1',) if self.local_only else PILOT_NAMES

    def width(self, name):
        return (self.local_width if name in ('A1', 'B1') else
                self.mid_width if name in ('L2', 'S2') else
                self.high_width if name == 'L3' else self.decoder_width)

    def hat_config(self, name):
        # Config.experts is the total over three layers, not the bank size.
        width = self.width(name)
        return HATConfig(width=width, experts=3*self.experts_per_layer,
                         expert_hidden=max(4, self.expert_hidden*width//self.local_width),
                         groups=self.groups, active_groups=self.active_groups,
                         experts_per_group=self.experts_per_group,
                         address_dim=self.address_dim, laces=self.laces,
                         active_laces=self.active_laces, hat_levels=1,
                         hat_symbols=self.hat_symbols, hat_active_symbols=self.hat_active_symbols,
                         kron_heads=self.kron_heads, kron_key=self.kron_key,
                         kron_value=self.kron_value, seed=self.seed+1009*PILOT_NAMES.index(name))

    def validate(self):
        for key, value in asdict(self).items():
            if key.endswith('_weight'):
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid {key}.')
            elif key in ('local_only', 'top_down'):
                if type(value) is not bool:
                    raise ValueError(f'{key} must be boolean.')
            elif type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer.')
        if self.experts_per_layer % self.groups:
            raise ValueError('Expert groups must divide each bank evenly.')
        for name in self.names:
            try:
                self.hat_config(name).validate()
            except AssertionError as exc:
                raise ValueError('Invalid sparse region dimensions.') from exc

    def to_dict(self):
        return asdict(self)
