"""Small payload adaptation with version-owned, factored delayed sensitivities.

For fixed orthonormal directions U, a candidate is x + (a * tanh(x U^T)) U.
Only a is new trainable capacity. Historical x and write decisions are held fixed
by the delayed estimator; ordinary autograd still trains x in the live window.
The publication Jacobian factors as diag(z) U, so each owner retains only rank
scalars, not a payload-by-parameter tensor or a history of observations.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .task_credit import WriteEligibility, CompetitiveWriteEligibility


class PayloadAdapter(nn.Module):
    def __init__(self, owners, width, rank):
        super().__init__()
        if not 1 <= rank <= width:
            raise ValueError('Payload rank must be between one and payload width.')
        # Deterministic orthonormal DCT directions; consume no initialization RNG.
        channel = torch.arange(width, dtype=torch.float64) + .5
        frequency = torch.arange(rank, dtype=torch.float64)[:, None]
        basis = torch.cos(math.pi * frequency * channel / width) * math.sqrt(2 / width)
        basis[0] /= math.sqrt(2)
        self.register_buffer('basis', basis.float())
        self.coefficients = nn.Parameter(torch.zeros(owners, rank))

    def forward(self, payload):
        features = (payload @ self.basis.T).tanh()
        return payload + (features * self.coefficients) @ self.basis, features


@dataclass(frozen=True)
class PayloadEligibility:
    owners: tuple[str, ...]
    features: torch.Tensor  # [batch, owner, rank], bounded by one
    versions: torch.Tensor  # [batch, owner]

    @classmethod
    def zeros(cls, batch, owners, rank, device='cpu', dtype=torch.float32):
        owners = tuple(owners)
        if min(batch, len(owners), rank) < 1 or len(set(owners)) != len(owners):
            raise ValueError('Invalid payload eligibility dimensions or owners.')
        return cls(owners, torch.zeros(batch, len(owners), rank, device=device, dtype=dtype),
                   torch.zeros(batch, len(owners), device=device, dtype=torch.long))

    @torch.no_grad()
    def advance(self, features, action, versions):
        if features.shape != self.features.shape or action.shape != self.versions.shape:
            raise ValueError('Payload features/actions have wrong shape.')
        if versions.shape != self.versions.shape or versions.dtype != torch.long:
            raise ValueError('Payload versions must be int64 [batch, owners].')
        if any(x.device != self.features.device for x in (features, action, versions)):
            raise ValueError('Payload eligibility tensors must share a device.')
        if not bool(((action == 0) | (action == 1)).all()):
            raise ValueError('Payload eligibility requires hard write decisions.')
        if not bool((versions - self.versions == action.long()).all()):
            raise ValueError('Payload write/hold does not match publication version.')
        if not bool(torch.isfinite(features).all()) or bool((features.abs() > 1).any()):
            raise ValueError('Payload features must be finite and bounded by one.')
        updated = torch.where(action.bool()[..., None], features.detach(), self.features)
        return PayloadEligibility(self.owners, updated, versions.detach().clone())

    def advance_chunk(self, activity):
        trace = self
        for t in range(activity['write_actions'].shape[1]):
            if bool(activity['reset'][:, t].any()):
                trace = trace.reset(activity['reset'][:, t])
            trace = trace.advance(activity['payload_features'][:, t],
                                  activity['write_actions'][:, t], activity['memory_versions'][:, t])
        return trace

    def reset(self, rows):
        if rows.shape != self.versions.shape[:1] or rows.dtype != torch.bool or rows.device != self.features.device:
            raise ValueError('Reset must be bool [batch] on the trace device.')
        return PayloadEligibility(self.owners, self.features.masked_fill(rows[:, None, None], 0),
                                  self.versions.masked_fill(rows[:, None], 0))

    def to(self, device):
        return PayloadEligibility(self.owners, self.features.to(device), self.versions.to(device))

    def storage_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.features, self.versions))

    def to_dict(self):
        return dict(owners=self.owners, features=self.features.detach().clone(),
                    versions=self.versions.detach().clone())

    @classmethod
    def from_dict(cls, data):
        result = cls(tuple(data['owners']), data['features'].detach().clone(),
                     data['versions'].detach().clone())
        if result.features.ndim != 3:
            raise ValueError('Malformed payload eligibility.')
        b, m, r = result.features.shape
        cls.zeros(b, result.owners, r, result.features.device, result.features.dtype)
        if m != len(result.owners) or result.versions.shape != (b, m):
            raise ValueError('Malformed payload eligibility owners/versions.')
        if result.versions.dtype != torch.long or bool((result.versions < 0).any()):
            raise ValueError('Malformed payload eligibility versions.')
        # Validate bounds and device without advancing identities.
        result.advance(result.features, torch.zeros_like(result.versions), result.versions)
        if bool((result.versions.eq(0)[..., None] & result.features.ne(0)).any()):
            raise ValueError('Unwritten publications cannot have payload eligibility.')
        return result


@dataclass
class PayloadWriteBoundary:
    write: object
    payload_features: torch.Tensor
    payload_consumed: bool = False

    @property
    def memory(self):
        return self.write.memory

    def apply(self, policy, enabled=True):
        return self.write.apply(policy, enabled)

    @torch.no_grad()
    def apply_payload(self, adapter, enabled=True):
        if self.payload_consumed:
            raise RuntimeError('Payload boundary credit has already been consumed.')
        self.payload_consumed = True
        if self.payload_features.shape[1:] != adapter.coefficients.shape:
            raise ValueError('Payload adapter does not match boundary sensitivity.')
        if not enabled or self.memory.grad is None:
            return 0.
        width = adapter.basis.shape[1]
        gradient = ((self.memory.grad[..., :width] @ adapter.basis.T) * self.payload_features).sum(0)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('Nonfinite delayed payload gradient.')
        parameter = adapter.coefficients
        if parameter.grad is None:
            parameter.grad = gradient.clone()
        else:
            parameter.grad.add_(gradient)
        return float(gradient.norm())


@dataclass(frozen=True)
class PayloadWriteEligibility:
    write: object
    payload: PayloadEligibility

    def advance_chunk(self, activity):
        return PayloadWriteEligibility(self.write.advance_chunk(activity), self.payload.advance_chunk(activity))

    def capture(self, memory):
        if not torch.equal(self.write.versions, self.payload.versions) or self.write.owners != self.payload.owners:
            raise ValueError('Write and payload publication identities differ.')
        return PayloadWriteBoundary(self.write.capture(memory), self.payload.features.detach().clone())

    def reset(self, rows):
        return PayloadWriteEligibility(self.write.reset(rows), self.payload.reset(rows))

    def to(self, device):
        return PayloadWriteEligibility(self.write.to(device), self.payload.to(device))

    def storage_bytes(self):
        return self.write.storage_bytes() + self.payload.storage_bytes()

    def to_dict(self):
        return dict(write=self.write.to_dict(), payload=self.payload.to_dict())


def make_eligibility(model, batch, saved=None):
    """Keep legacy checkpoint layout unchanged when the adapter is disabled."""
    kind = CompetitiveWriteEligibility if model.cfg.write_mode == 'competitive' else WriteEligibility
    source = saved['write'] if saved is not None and model.cfg.payload_rank else saved
    write = kind.from_dict(source) if source is not None else kind.zeros(
        batch, model.owner_names, model.memory_dim, model.cfg.policy_features,
        device=model.write_policy.device, dtype=model.write_policy.dtype)
    if not model.cfg.payload_rank:
        return write
    payload = PayloadEligibility.from_dict(saved['payload']) if saved is not None else PayloadEligibility.zeros(
        batch, model.owner_names, model.cfg.payload_rank, model.write_policy.device, model.write_policy.dtype)
    result = PayloadWriteEligibility(write, payload)
    if write.owners != payload.owners or not torch.equal(write.versions, payload.versions):
        raise ValueError('Saved write and payload publication identities differ.')
    if payload.features.shape != (batch, model.cfg.owners, model.cfg.payload_rank):
        raise ValueError('Saved payload eligibility configuration mismatch.')
    return result
