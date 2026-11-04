"""Opt-in current-pass representation objectives and explicit delayed utility.

These helpers do not run extra model passes, store replay examples, or place
labels in graph state. Pair independently encoded ingress activity, never node
publications that may already contain another view through feedback.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceView:
    name: str
    payload: torch.Tensor
    available: torch.Tensor
    provenance: str = 'independent_ingress'

    def __post_init__(self):
        if self.provenance != 'independent_ingress':
            raise ValueError('Paired views must be independently encoded ingress activity.')
        if self.payload.ndim not in (2, 3) or self.available.shape != self.payload.shape[:-1]:
            raise ValueError('Expected payload [B,D] or [B,T,D] and matching availability.')
        if self.available.dtype != torch.bool or self.available.device != self.payload.device:
            raise ValueError('Availability must be boolean on the payload device.')
        if min(self.payload.shape) < 1:
            raise ValueError('Source-view dimensions must be nonempty.')
        if not self.payload.is_floating_point():
            raise ValueError('Latent payload must be floating point.')

    @classmethod
    def from_activity(cls, name, activity):
        """Wrap the model's optional view_activity tap, preserving autograd."""
        item = activity[name]
        return cls(name, item['payload'], item['available'],
                   item.get('provenance', 'independent_ingress'))


    def pool(self, mode='last'):
        """Pool each description independently before paired availability tests."""
        if mode not in ('last', 'mean', 'none'):
            raise ValueError('Pooling must be last, mean or none.')
        if self.payload.ndim == 2 or mode == 'none':
            return self
        available = self.available.any(1)
        safe = torch.where(self.available[..., None], self.payload, torch.zeros_like(self.payload))
        if mode == 'mean':
            payload = safe.sum(1)/self.available.sum(1).clamp_min(1)[:, None]
        else:
            positions = torch.arange(self.payload.shape[1], device=self.payload.device)[None]
            index = torch.where(self.available, positions, -1).max(1).values.clamp_min(0)
            payload = safe[torch.arange(len(safe), device=safe.device), index]
        payload = torch.where(available[:, None], payload, torch.zeros_like(payload))
        return SourceView(self.name, payload, available, self.provenance)


@dataclass(frozen=True)
class ViewObjectiveConfig:
    shared_dim: int = 16
    private_dim: int = 16
    pooling: str = 'last'
    agreement_enabled: bool = False
    prediction_enabled: bool = False
    variance_enabled: bool = False
    covariance_enabled: bool = False
    private_enabled: bool = False
    separation_enabled: bool = False
    agreement_weight: float = .01
    prediction_weight: float = .01
    variance_weight: float = .01
    covariance_weight: float = .001
    private_weight: float = .01
    separation_weight: float = .001
    variance_target: float = 1.
    variance_epsilon: float = 1e-4

    def validate(self):
        if self.pooling not in ('last', 'mean', 'none'):
            raise ValueError('Pooling must be last, mean or none.')
        if min(self.shared_dim, self.private_dim) < 1:
            raise ValueError('Projection dimensions must be positive.')
        if (not math.isfinite(self.variance_target) or not math.isfinite(self.variance_epsilon)
                or self.variance_target <= 0 or self.variance_epsilon <= 0):
            raise ValueError('Variance target and epsilon must be positive.')
        for key, value in vars(self).items():
            if key.endswith('_weight') and (not math.isfinite(value) or value < 0):
                raise ValueError('Auxiliary weights must be nonnegative.')


class PairedViewObjective(nn.Module):
    """Shared-code prediction plus independent private-detail supervision.

    All terms are independently enabled and disabled by default. Agreement and
    prediction use only jointly available samples. Variance/covariance use each
    view's available samples separately, requiring at least two observations.
    Private labels are bytes (0..255), with -100 as the ignored target, and are
    consumed only by this loss helper. They never affect the returned codes.
    By default, each description is pooled at its own last available occurrence
    before comparing views; targets then have shape [B]. pooling='none' retains
    aligned per-occurrence comparisons and targets with shape [B,T].
    """
    def __init__(self, width, views=('A', 'B'), cfg=None):
        super().__init__()
        self.cfg = cfg or ViewObjectiveConfig()
        self.cfg.validate()
        if width < 1 or len(views) != 2 or len(set(views)) != 2:
            raise ValueError('Two distinct view names and positive width required.')
        if any(not v.isidentifier() or v in ('training', 'forward') for v in views):
            raise ValueError('View names must be valid module identifiers.')
        self.width, self.views = width, tuple(views)
        self.shared = nn.ModuleDict({v: nn.Linear(width, self.cfg.shared_dim) for v in views})
        self.predict = nn.ModuleDict({v: nn.Linear(self.cfg.shared_dim, self.cfg.shared_dim) for v in views})
        self.private = nn.ModuleDict({v: nn.Linear(width, self.cfg.private_dim) for v in views})
        self.private_readout = nn.ModuleDict({v: nn.Linear(self.cfg.private_dim, 256) for v in views})

    def _codes(self, view):
        if view.payload.shape[-1] != self.width or view.name not in self.views:
            raise ValueError('View identity or width does not match objective.')
        values = view.payload.reshape(-1, self.width)
        mask = view.available.reshape(-1)
        # Unavailable values cannot contribute even if their placeholder is NaN.
        safe = torch.where(mask[:, None], values, torch.zeros_like(values))
        shared = self.shared[view.name](safe)
        private = self.private[view.name](safe)
        return (torch.where(mask[:, None], shared, torch.zeros_like(shared)),
                torch.where(mask[:, None], private, torch.zeros_like(private)), mask)

    def forward(self, left: SourceView, right: SourceView, private_targets=None):
        left, right = left.pool(self.cfg.pooling), right.pool(self.cfg.pooling)
        if (left.name, right.name) != self.views:
            raise ValueError('View order must match configured names.')
        if left.payload.shape != right.payload.shape or left.payload.device != right.payload.device:
            raise ValueError('Paired inputs must have matching shape and device.')
        a, pa, ma = self._codes(left)
        b, pb, mb = self._codes(right)
        pair = ma & mb
        cfg = self.cfg
        zero = (a.sum()+b.sum()+pa.sum()+pb.sum())*0
        terms = {key: zero for key in ('agreement', 'prediction', 'variance', 'covariance', 'private', 'separation')}
        if bool(pair.any()):
            if cfg.agreement_enabled:
                terms['agreement'] = F.mse_loss(a[pair], b[pair])
            if cfg.prediction_enabled:
                terms['prediction'] = .5*(F.mse_loss(self.predict[left.name](a[pair]), b[pair].detach()) +
                                           F.mse_loss(self.predict[right.name](b[pair]), a[pair].detach()))
        available_stats = 0
        for shared, private, mask in ((a, pa, ma), (b, pb, mb)):
            count = int(mask.sum())
            if count < 2:
                continue
            available_stats += 1
            code = shared[mask]
            centered = code-code.mean(0, keepdim=True)
            if cfg.variance_enabled:
                std = (centered.square().mean(0)+cfg.variance_epsilon).sqrt()
                terms['variance'] = terms['variance'] + F.relu(cfg.variance_target-std).mean()
            if cfg.covariance_enabled:
                covariance = centered.T @ centered/(count-1)
                off_diagonal = covariance-torch.diag_embed(covariance.diagonal())
                terms['covariance'] = terms['covariance'] + off_diagonal.square().sum()/cfg.shared_dim
            if cfg.separation_enabled:
                priv = private[mask]-private[mask].mean(0, keepdim=True)
                cross = centered.T @ priv/(count-1)
                terms['separation'] = terms['separation'] + cross.square().mean()
        for term in ('variance', 'covariance', 'separation'):
            terms[term] = terms[term]/max(available_stats, 1)
        if cfg.private_enabled:
            if private_targets is None or set(private_targets) != set(self.views):
                raise ValueError('Private objective requires one byte-target tensor per view.')
            private_count = 0
            for view, codes, mask in ((left, pa, ma), (right, pb, mb)):
                targets = private_targets[view.name]
                if targets.shape != view.available.shape or targets.dtype != torch.long or targets.device != codes.device:
                    raise ValueError('Private targets must be int64 with the availability shape/device.')
                targets = targets.reshape(-1)
                used = mask & (targets != -100)
                if bool(used.any()):
                    terms['private'] = terms['private'] + F.cross_entropy(self.private_readout[view.name](codes[used]), targets[used])
                    private_count += 1
            terms['private'] = terms['private']/max(private_count, 1)
        loss = sum(terms[k]*getattr(cfg, k+'_weight') for k in terms)
        shape = left.payload.shape[:-1]
        return {'loss': loss, **terms, 'paired_samples': int(pair.sum()),
                'variance_views': available_stats,
                'codes': {left.name: {'shared': a.reshape(*shape, cfg.shared_dim), 'private': pa.reshape(*shape, cfg.private_dim)},
                          right.name: {'shared': b.reshape(*shape, cfg.shared_dim), 'private': pb.reshape(*shape, cfg.private_dim)}}}


@dataclass(frozen=True)
class UtilityPacket:
    """Detached bounded sufficient statistics from one retained forward segment."""
    edge_names: tuple[str, ...]
    node_names: tuple[str, ...]
    edge_mass: torch.Tensor
    edge_gain_sum: torch.Tensor
    lace_mass: torch.Tensor
    lace_gain_sum: torch.Tensor
    supervised_events: int


class DelayedUtility(nn.Module):
    """Per-edge and selected-lace signed associations, changed only by commit.

    Eligibility is contemporaneous with supervised events in the supplied
    trace. Every executed round counts; no retrospective/replay pass is used.
    This is an association estimate, not edgewise causal attribution.
    """
    def __init__(self, edge_names, node_names, laces, layers=3, enabled=False, decay=.97, gain_clip=2.):
        super().__init__()
        self.edge_names, self.node_names = tuple(edge_names), tuple(node_names)
        if len(set(self.edge_names)) != len(self.edge_names) or len(set(self.node_names)) != len(self.node_names):
            raise ValueError('Utility identities must be unique.')
        if (not self.edge_names or not self.node_names or min(laces, layers) < 1
                or not 0 <= decay < 1 or not math.isfinite(gain_clip) or gain_clip <= 0):
            raise ValueError('Invalid utility dimensions, decay or gain clip.')
        self.edge_index = {key: i for i, key in enumerate(self.edge_names)}
        self.node_index = {key: i for i, key in enumerate(self.node_names)}
        self.enabled, self.decay, self.gain_clip = enabled, decay, gain_clip
        self.layers, self.laces = layers, laces
        self.register_buffer('edge_mass', torch.zeros(len(self.edge_names)))
        self.register_buffer('edge_gain_sum', torch.zeros(len(self.edge_names)))
        self.register_buffer('lace_mass', torch.zeros(len(self.node_names), layers, laces))
        self.register_buffer('lace_gain_sum', torch.zeros(len(self.node_names), layers, laces))
        self.register_buffer('feedback_updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def make_packet(self, trace, baseline_losses, regional_losses, supervised=None):
        if not self.enabled:
            return None
        if baseline_losses.ndim != 2 or baseline_losses.shape != regional_losses.shape:
            raise ValueError('Comparison losses must have matching [B,T] shapes.')
        if baseline_losses.device != regional_losses.device or baseline_losses.device != self.edge_mass.device:
            raise ValueError('Comparison and utility devices must match.')
        if supervised is None:
            supervised = torch.ones_like(baseline_losses, dtype=torch.bool)
        if supervised.shape != baseline_losses.shape or supervised.dtype != torch.bool or supervised.device != baseline_losses.device:
            raise ValueError('Supervision mask must be boolean [B,T] on the comparison device.')
        n = int(supervised.sum())
        gain = baseline_losses.detach()-regional_losses.detach()
        if not bool(torch.isfinite(gain[supervised]).all()):
            raise ValueError('Nonfinite supervised gain.')
        gain = gain.clamp(-self.gain_clip, self.gain_clip)
        gain = torch.where(supervised, gain, torch.zeros_like(gain))
        mass, value = torch.zeros_like(self.edge_mass), torch.zeros_like(self.edge_gain_sum)
        lm, lv = torch.zeros_like(self.lace_mass), torch.zeros_like(self.lace_gain_sum)
        b, t = gain.shape
        for record in trace:
            event = record['event']
            if not isinstance(event, int) or not 0 <= event < t:
                raise ValueError('Trace event lies outside comparison window.')
            for dependency in record.get('dependencies', ()):
                key = dependency.edge
                if key not in self.edge_index:
                    continue  # Ingress transport has no inter-node utility entry.
                used = dependency.consumed
                if used.shape != (b,) or used.dtype != torch.bool or used.device != gain.device:
                    raise ValueError('Dependency consumption mask does not match comparison batch.')
                substituted = getattr(dependency, 'substituted', None)
                if substituted is not None:
                    if substituted.shape != (b,) or substituted.dtype != torch.bool or substituted.device != gain.device:
                        raise ValueError('Substitution mask does not match comparison batch.')
                    used = used & ~substituted
                elif dependency.intervention != 'none':
                    continue
                used = used & supervised[:, event]
                index = self.edge_index[key]
                mass[index] += used.sum()
                value[index] += gain[:, event][used].sum()
            routing = record.get('routing')
            if routing is None:
                continue
            name = record['node']
            if name not in self.node_index:
                raise ValueError('Trace references an unknown node.')
            rows, ids = routing['rows'], routing['laces']
            if rows.ndim != 1 or rows.dtype != torch.long or ids.ndim != 3 or ids.shape[:2] != (len(rows), self.layers):
                raise ValueError('Malformed selected-lace trace.')
            if rows.device != gain.device or ids.device != gain.device or ids.dtype != torch.long:
                raise ValueError('Selected lace identities must be int64 on the comparison device.')
            if bool(((rows < 0) | (rows >= b)).any()) or bool(((ids < 0) | (ids >= self.laces)).any()) or ids.shape[2] < 1:
                raise ValueError('Selected lace identity out of bounds.')
            eligible = supervised[rows, event]
            if 'valid' in record:
                valid = record['valid']
                if valid.shape != (b,) or valid.dtype != torch.bool or valid.device != gain.device:
                    raise ValueError('Publication validity must be boolean [B] on the comparison device.')
                eligible = eligible & valid[rows]
            selected = ids[eligible]
            signed = gain[rows[eligible], event]
            node = self.node_index[name]
            for depth in range(self.layers):
                indices = selected[:, depth].reshape(-1)
                weights = torch.ones_like(indices, dtype=lm.dtype)/ids.shape[2]
                lm[node, depth].scatter_add_(0, indices, weights)
                lv[node, depth].scatter_add_(0, indices, signed.repeat_interleave(ids.shape[2]).to(lv.dtype)/ids.shape[2])
        divisor = max(n, 1)
        return UtilityPacket(self.edge_names, self.node_names, mass/divisor, value/divisor,
                             lm/divisor, lv/divisor, n)

    @torch.no_grad()
    def commit_feedback(self, packet):
        """Call explicitly after optimizer.step(); never called by forward/eval."""
        if not self.training:
            raise RuntimeError('Feedback commits are training-only.')
        if not self.enabled or packet is None:
            return
        if packet.edge_names != self.edge_names or packet.node_names != self.node_names:
            raise ValueError('Utility packet topology differs from this module.')
        pairs = ((self.edge_mass, packet.edge_mass), (self.edge_gain_sum, packet.edge_gain_sum),
                 (self.lace_mass, packet.lace_mass), (self.lace_gain_sum, packet.lace_gain_sum))
        for stored, supplied in pairs:
            if supplied.shape != stored.shape or supplied.device != stored.device or supplied.requires_grad or supplied.grad_fn is not None:
                raise ValueError('Utility packets must be detached with matching dimensions/devices.')
            if not bool(torch.isfinite(supplied).all()):
                raise ValueError('Nonfinite utility packet.')
        if bool((packet.edge_mass < 0).any()) or bool((packet.lace_mass < 0).any()):
            raise ValueError('Eligibility mass must be nonnegative.')
        if packet.supervised_events == 0:
            return
        for prefix in ('edge', 'lace'):
            old_mass = getattr(self, prefix+'_mass')
            old_value = getattr(self, prefix+'_gain_sum')
            new_mass = getattr(packet, prefix+'_mass')
            new_value = getattr(packet, prefix+'_gain_sum')
            active = new_mass > 0
            old_mass.copy_(torch.where(active, self.decay*old_mass+(1-self.decay)*new_mass, old_mass))
            old_value.copy_(torch.where(active, self.decay*old_value+(1-self.decay)*new_value, old_value))
        if bool((packet.edge_mass > 0).any()) or bool((packet.lace_mass > 0).any()):
            self.feedback_updates.add_(1)

    def gate_bias(self, scale=.1):
        """Detached per-edge gate-logit biases for an explicitly enabled pass."""
        values = self._bias('edge', scale)
        return {key: values[i] for i, key in enumerate(self.edge_names)}

    def lace_bias(self, scale=.1):
        """Detached [layers,laces] biases per node for the next routing pass."""
        values = self._bias('lace', scale)
        return {key: values[i] for i, key in enumerate(self.node_names)}

    def _bias(self, prefix, scale):
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Utility gate scale must be nonnegative.')
        values = getattr(self, prefix+'_gain_sum')/getattr(self, prefix+'_mass').clamp_min(1e-8)
        if not self.enabled:
            values = torch.zeros_like(values)
        return (scale*values.clamp(-self.gain_clip, self.gain_clip)).detach()
