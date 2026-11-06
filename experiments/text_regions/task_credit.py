"""Bounded conditional write-policy derivatives across graph detach boundaries.

The trace is a forward sensitivity of each *current* owned publication, not an
episode buffer. Candidate payloads, semantic headers and policy features are
treated as constants by this estimator. Within a chunk their ordinary gradients
remain available. For a straight-through (ST) hard write, the conditional update
is E' = (1-a) E + p(1-p) (candidate-old) outer [features, 1]. A replacement
therefore removes the previous publication's responsibility; a hold retains it.

At a chunk boundary the carried memory is a new autograd leaf. Its eventual loss
gradient already includes the actual selected reads and subsequent writes. Only
the boundary's prefix sensitivity is contracted into the policy gradient, once;
new within-chunk gate derivatives must not be injected a second time. Decay is
an explicit bias toward recent gate responsibility (set it to 1 for exact
conditional-ST comparison). This does not estimate derivatives through historic
candidate/feature networks or through past optimizer steps.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class CreditBoundary:
    """Single-use delayed-gradient receipt; do not serialize its autograd leaf."""
    memory: torch.Tensor
    sensitivity: torch.Tensor
    consumed: bool = False

    @torch.no_grad()
    def apply(self, policy: torch.Tensor, enabled=True):
        """Add prefix credit after backward, before gradient clipping/optimizer."""
        if self.consumed:
            raise RuntimeError('Boundary credit has already been consumed.')
        self.consumed = True
        if policy.shape != self.sensitivity.shape[1::2]:
            raise ValueError('Policy must be [owners, features+1].')
        if not enabled or self.memory.grad is None:
            return 0.
        gradient = torch.einsum('bmd,bmdf->mf', self.memory.grad, self.sensitivity)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('Nonfinite delayed write-policy gradient.')
        gradient = gradient.to(device=policy.device, dtype=policy.dtype)
        if policy.grad is None:
            policy.grad = gradient.clone()
        else:
            policy.grad.add_(gradient)
        return float(gradient.norm())


@dataclass(frozen=True)
class WriteEligibility:
    owners: tuple[str, ...]
    sensitivity: torch.Tensor  # [batch, owners, packed publication width, features+1]
    versions: torch.Tensor  # [batch, owners], identity of represented publication
    age: torch.Tensor  # events since last physical replacement
    decay: float = .995

    @classmethod
    def zeros(cls, batch, owners, state_width, feature_width, device='cpu',
              dtype=torch.float32, decay=.995):
        owners = tuple(owners)
        if (min(batch, len(owners), state_width, feature_width) < 1 or
                len(set(owners)) != len(owners) or not math.isfinite(decay) or not 0 <= decay <= 1):
            raise ValueError('Invalid eligibility dimensions, owners or decay.')
        shape = (batch, len(owners))
        return cls(owners, torch.zeros(*shape, state_width, feature_width+1, device=device, dtype=dtype),
                   torch.zeros(shape, device=device, dtype=torch.long),
                   torch.zeros(shape, device=device, dtype=torch.long), decay)

    @torch.no_grad()
    def advance(self, previous, candidate, features, probability, action, versions,
                eligible=None, active=None):
        """Propagate one event, rebinding compact responsibility to its version.

        ``action`` is the actual hard/ST forward value (or a soft interpolation
        value in [0,1]). ``eligible=False`` excludes forced interventions from
        the gate derivative while their physical replacement still erases old
        responsibility. ``active=False`` leaves that row completely unchanged.
        """
        b, m, d, f = self.sensitivity.shape
        if previous.shape != (b, m, d) or candidate.shape != previous.shape:
            raise ValueError('Packed old/candidate publications have wrong shape.')
        if features.shape != (b, m, f-1) or probability.shape != (b, m) or action.shape != (b, m):
            raise ValueError('Write-policy features/probabilities/actions have wrong shape.')
        if versions.shape != (b, m) or versions.dtype != torch.long:
            raise ValueError('Publication versions must be int64 [batch, owners].')
        if eligible is None:
            eligible = torch.ones_like(action, dtype=torch.bool)
        if active is None:
            active = torch.ones_like(action, dtype=torch.bool)
        if eligible.shape != (b, m) or active.shape != (b, m) or eligible.dtype != torch.bool or active.dtype != torch.bool:
            raise ValueError('Eligibility/activity masks must be bool [batch, owners].')
        tensors = (previous, candidate, features, probability, action, versions, eligible, active)
        if any(value.device != self.sensitivity.device for value in tensors):
            raise ValueError('Eligibility tensors must share a device.')
        if not bool(((probability >= 0) & (probability <= 1) & (action >= 0) & (action <= 1)).all()):
            raise ValueError('Write probabilities/actions must lie in [0,1].')
        delta_version = versions-self.versions
        if not bool(((delta_version >= 0) & (delta_version <= 1)).all()) or bool((~active & delta_version.ne(0)).any()):
            raise ValueError('Versions must advance by at most one per active event; reset traces explicitly.')
        # Version changes are mandatory for an actual hard replacement and
        # prohibited for a hard hold. Soft experimental writes may also advance.
        if bool((active & action.eq(1) & delta_version.ne(1)).any()) or bool((active & action.eq(0) & delta_version.ne(0)).any()):
            raise ValueError('Physical write/hold decision does not match publication version.')
        feature = torch.cat((features.detach(), torch.ones_like(features[..., :1])), -1)
        derivative = probability.detach()*(1-probability.detach())*eligible
        fresh = (candidate.detach()-previous.detach())[..., None]*feature[..., None, :]*derivative[..., None, None]
        updated = (1-action.detach().to(self.sensitivity.dtype))[..., None, None]*self.sensitivity*self.decay+fresh
        updated = torch.where(active[..., None, None], updated, self.sensitivity)
        age = torch.where(active, torch.where(delta_version.ne(0), 0, self.age+1), self.age)
        return WriteEligibility(self.owners, updated, versions.detach().clone(), age, self.decay)

    def advance_chunk(self, activity):
        """Advance from task-memory forward activity, shape [B,T,M,...]."""
        trace = self
        for t in range(activity['write_probs'].shape[1]):
            if 'reset' in activity and bool(activity['reset'][:, t].any()):
                trace = trace.reset(activity['reset'][:, t])
            trace = trace.advance(activity['memory_before'][:, t], activity['memory_candidates'][:, t],
                                  activity['write_features'][:, t], activity['write_probs'][:, t],
                                  activity['write_actions'][:, t], activity['memory_versions'][:, t],
                                  None if 'write_eligible' not in activity else activity['write_eligible'][:, t],
                                  None if 'write_active' not in activity else activity['write_active'][:, t])
        return trace

    def capture(self, memory):
        """Create the next detached prefix leaf and an immutable trace snapshot."""
        if memory.shape != self.sensitivity.shape[:3] or memory.device != self.sensitivity.device:
            raise ValueError('Boundary memory does not match eligibility.')
        return CreditBoundary(memory.detach().clone().requires_grad_(True), self.sensitivity.detach().clone())

    def reset(self, rows):
        if rows.shape != self.versions.shape[:1] or rows.dtype != torch.bool or rows.device != self.versions.device:
            raise ValueError('Reset mask must be boolean [batch] on the trace device.')
        return WriteEligibility(self.owners, torch.where(rows[:, None, None, None], 0, self.sensitivity),
                                torch.where(rows[:, None], 0, self.versions),
                                torch.where(rows[:, None], 0, self.age), self.decay)

    def to(self, device):
        return WriteEligibility(self.owners, self.sensitivity.to(device), self.versions.to(device), self.age.to(device), self.decay)

    def tensors(self):
        return (self.sensitivity, self.versions, self.age)

    def storage_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.tensors())

    def to_dict(self):
        return dict(owners=self.owners, sensitivity=self.sensitivity.detach().clone(),
                    versions=self.versions.detach().clone(), age=self.age.detach().clone(), decay=self.decay)

    @classmethod
    def from_dict(cls, data):
        result = cls(tuple(data['owners']), data['sensitivity'].detach().clone(),
                     data['versions'].detach().clone(), data['age'].detach().clone(), float(data['decay']))
        b, m, d, f = result.sensitivity.shape
        if (m != len(result.owners) or min(b, m, d, f-1) < 1 or result.versions.shape != (b, m) or
                result.age.shape != (b, m) or result.versions.dtype != torch.long or result.age.dtype != torch.long or
                len(set(result.owners)) != m or not math.isfinite(result.decay) or not 0 <= result.decay <= 1):
            raise ValueError('Malformed saved eligibility.')
        return result


@dataclass
class CompetitiveCreditBoundary(CreditBoundary):
    """Single-use receipt retaining cross-owner softmax responsibility."""

    @torch.no_grad()
    def apply(self, policy: torch.Tensor, enabled=True):
        if self.consumed:
            raise RuntimeError('Boundary credit has already been consumed.')
        self.consumed = True
        if policy.shape != self.sensitivity.shape[-2:]:
            raise ValueError('Policy must be [owners, features+1].')
        if not enabled or self.memory.grad is None:
            return 0.
        gradient = torch.einsum('bmd,bmdnf->nf', self.memory.grad, self.sensitivity)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('Nonfinite delayed write-policy gradient.')
        gradient = gradient.to(device=policy.device, dtype=policy.dtype)
        if policy.grad is None:
            policy.grad = gradient.clone()
        else:
            policy.grad.add_(gradient)
        return float(gradient.norm())


@dataclass(frozen=True)
class CompetitiveWriteEligibility(WriteEligibility):
    """Conditional sensitivity for owner logits competing with a fixed null.

    E has shape [batch, publication owner, state width, policy owner, features+1].
    The owner softmax Jacobian is p_m (delta_mn - p_n), so a later read from one
    publication can train every competing write head. The fixed null logit has
    no parameter or separate sensitivity column. ``eligible`` masks the output
    publication's ST tangent; it does not freeze other logits in that softmax.
    A forced bootstrap row sets every owner's eligibility false.
    """

    @classmethod
    def zeros(cls, batch, owners, state_width, feature_width, device='cpu',
              dtype=torch.float32, decay=.995):
        owners = tuple(owners)
        if (min(batch, len(owners), state_width, feature_width) < 1 or
                len(set(owners)) != len(owners) or not math.isfinite(decay) or not 0 <= decay <= 1):
            raise ValueError('Invalid eligibility dimensions, owners or decay.')
        shape = (batch, len(owners))
        return cls(owners, torch.zeros(*shape, state_width, len(owners), feature_width+1,
                                      device=device, dtype=dtype),
                   torch.zeros(shape, device=device, dtype=torch.long),
                   torch.zeros(shape, device=device, dtype=torch.long), decay)

    @torch.no_grad()
    def advance(self, previous, candidate, features, probability, action, versions,
                eligible=None, active=None):
        b, m, d, n, f = self.sensitivity.shape
        if n != m or previous.shape != (b, m, d) or candidate.shape != previous.shape:
            raise ValueError('Packed old/candidate publications have wrong shape.')
        if features.shape != (b, m, f-1) or probability.shape != (b, m) or action.shape != (b, m):
            raise ValueError('Write-policy features/probabilities/actions have wrong shape.')
        if versions.shape != (b, m) or versions.dtype != torch.long:
            raise ValueError('Publication versions must be int64 [batch, owners].')
        if eligible is None:
            eligible = torch.ones_like(action, dtype=torch.bool)
        if active is None:
            active = torch.ones_like(action, dtype=torch.bool)
        if eligible.shape != (b, m) or active.shape != (b, m) or eligible.dtype != torch.bool or active.dtype != torch.bool:
            raise ValueError('Eligibility/activity masks must be bool [batch, owners].')
        tensors = (previous, candidate, features, probability, action, versions, eligible, active)
        if any(value.device != self.sensitivity.device for value in tensors):
            raise ValueError('Eligibility tensors must share a device.')
        if not bool(((probability >= 0) & (probability <= 1) & (action >= 0) & (action <= 1)).all()):
            raise ValueError('Write probabilities/actions must lie in [0,1].')
        if bool((probability.sum(-1) > 1+1e-6).any()):
            raise ValueError('Owner softmax mass cannot exceed one including the implicit null.')
        multi_write = (action.to(probability.dtype)*active).sum(-1) > 1+1e-6
        if bool((multi_write & (eligible & active).any(-1)).any()):
            raise ValueError('Competitive writes allow at most one owner unless all tangents are forced off.')
        delta_version = versions-self.versions
        if not bool(((delta_version >= 0) & (delta_version <= 1)).all()) or bool((~active & delta_version.ne(0)).any()):
            raise ValueError('Versions must advance by at most one per active event; reset traces explicitly.')
        if bool((active & action.eq(1) & delta_version.ne(1)).any()) or bool((active & action.eq(0) & delta_version.ne(0)).any()):
            raise ValueError('Physical write/hold decision does not match publication version.')
        feature = torch.cat((features.detach(), torch.ones_like(features[..., :1])), -1)
        p = probability.detach()
        eye = torch.eye(m, device=p.device, dtype=p.dtype)
        jacobian = p[:, :, None]*(eye[None]-p[:, None, :])*eligible[:, :, None]
        difference = candidate.detach()-previous.detach()
        fresh = (difference[..., None, None]*jacobian[:, :, None, :, None]*
                 feature[:, None, None, :, :])
        hold = (1-action.detach().to(self.sensitivity.dtype))[..., None, None, None]
        updated = hold*self.sensitivity*self.decay+fresh
        updated = torch.where(active[..., None, None, None], updated, self.sensitivity)
        age = torch.where(active, torch.where(delta_version.ne(0), 0, self.age+1), self.age)
        return CompetitiveWriteEligibility(self.owners, updated, versions.detach().clone(), age, self.decay)

    def capture(self, memory):
        if memory.shape != self.sensitivity.shape[:3] or memory.device != self.sensitivity.device:
            raise ValueError('Boundary memory does not match eligibility.')
        return CompetitiveCreditBoundary(memory.detach().clone().requires_grad_(True),
                                         self.sensitivity.detach().clone())

    def reset(self, rows):
        if rows.shape != self.versions.shape[:1] or rows.dtype != torch.bool or rows.device != self.versions.device:
            raise ValueError('Reset mask must be boolean [batch] on the trace device.')
        return CompetitiveWriteEligibility(self.owners, torch.where(rows[:, None, None, None, None], 0, self.sensitivity),
                                           torch.where(rows[:, None], 0, self.versions),
                                           torch.where(rows[:, None], 0, self.age), self.decay)

    def to(self, device):
        return CompetitiveWriteEligibility(self.owners, self.sensitivity.to(device), self.versions.to(device),
                                           self.age.to(device), self.decay)

    @classmethod
    def from_dict(cls, data):
        result = cls(tuple(data['owners']), data['sensitivity'].detach().clone(),
                     data['versions'].detach().clone(), data['age'].detach().clone(), float(data['decay']))
        if result.sensitivity.ndim != 5:
            raise ValueError('Competitive eligibility requires cross-owner sensitivity.')
        b, m, d, n, f = result.sensitivity.shape
        if (m != len(result.owners) or m != n or min(b, m, d, f-1) < 1 or
                result.versions.shape != (b, m) or result.age.shape != (b, m) or
                result.versions.dtype != torch.long or result.age.dtype != torch.long or
                len(set(result.owners)) != m or not math.isfinite(result.decay) or not 0 <= result.decay <= 1):
            raise ValueError('Malformed saved competitive eligibility.')
        return result


@torch.no_grad()
def matched_read_advantage(live_losses, blocked_losses, mask=None):
    """Diagnostic conditional gain from the same inputs and pre-event state.

    The caller must fork the state before either forward and block only reads in
    the comparison branch. No unrelated-example loss baseline is accepted or
    maintained, and this diagnostic does not change the gradient estimator.
    """
    if live_losses.shape != blocked_losses.shape or live_losses.device != blocked_losses.device:
        raise ValueError('Matched comparison losses must have the same shape/device.')
    if mask is None:
        mask = torch.ones_like(live_losses, dtype=torch.bool)
    if mask.shape != live_losses.shape or mask.dtype != torch.bool or mask.device != live_losses.device:
        raise ValueError('Comparison mask must be bool with matching shape/device.')
    gain = blocked_losses.detach()-live_losses.detach()
    if not bool(torch.isfinite(gain[mask]).all()):
        raise ValueError('Nonfinite matched comparison gain.')
    return torch.where(mask, gain, torch.zeros_like(gain))
