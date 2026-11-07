"""Conditional admission credit carried by the existing producer state.

Each current MinKron matrix carries a sensitivity to its owner's admission
weights. Local recurrence inputs and historical policy features are held fixed;
the exact decay/erase state Jacobian transports that sensitivity. This is not
full RTRL: historical derivatives between layers, regions, feature networks and
optimizer steps remain truncated. No examples or past activations are retained.

Useful coordination of state commits, publication and recall is potentially
threshold-dependent. Keep this training control independently ablatable.
"""
from dataclasses import dataclass, replace
import torch
from torch.nn import functional as F

from ..hat_minkron.kron import normalize_direction
from .state import NodeState
from .graph_credit import make_graph_eligibility


@torch.no_grad()
def recurrence_factors(module, x, reset_each=False):
    """Recover the small local transition from the actual one-event input."""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError('Producer credit requires a single recurrent event.')
    qkv, erase, _, forget, _ = module.projection(x.detach()).split(module.splits, -1)
    _, k, v = F.silu(qkv).split([module.H*module.K, module.H*module.K, module.H*module.V], -1)
    k, v = [normalize_direction(a.reshape(x.shape[0], module.H, d), safe_zero=module.safe_zero)
            for a, d in ((k, module.K), (v, module.V))]
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    f = module.forget_projection(forget).to(dtype) + module.dt_bias.to(dtype)
    decay = (-module.A_log.to(dtype).exp()[None, :, None] *
             F.softplus(f).reshape(x.shape[0], module.H, module.K)).exp()
    if reset_each:
        decay = torch.zeros_like(decay)
    return decay, k.to(dtype), v.to(dtype), erase[:, 0].sigmoid().to(dtype)


def transport(sensitivity, factors):
    """J_state @ E, without materializing the state-by-state Jacobian."""
    decay, key, value, beta = factors
    decayed = sensitivity * decay[..., None, None]
    score = torch.einsum('bhkvf,bhk,bhv->bhf', decayed, key, value)
    scale = beta / value.square().sum(-1).clamp_min(1e-18)
    return decayed - (key[..., None, None] * value[:, :, None, :, None] *
                      (scale[..., None] * score)[:, :, None, None, :])


def commit_sensitivity(previous, transported, old, candidate, features, probability,
                       actions, eligible, active, state_commit='publication'):
    if state_commit not in ('publication', 'always'):
        raise ValueError('Unknown producer state-commit mode.')
    shape = (-1,) + (1,) * (previous.ndim - 1)
    if state_commit == 'always':
        updated = transported
    else:
        feature = torch.cat((features, torch.ones_like(features[:, :1])), -1)
        derivative = probability * (1-probability) * eligible
        fresh = ((candidate-old)[..., None] * derivative.reshape(shape) *
                 feature[:, None, None, None, :])
        updated = torch.where(actions.reshape(shape), transported, previous) + fresh
    return torch.where(active.reshape(shape), updated, previous)


def node_contents(state):
    return state.contextualizer + state.blocks


@dataclass
class ProducerBoundary:
    owners: tuple[str, ...]
    leaves: tuple[tuple[torch.Tensor, ...], ...]
    sensitivity: tuple[tuple[torch.Tensor, ...], ...]
    consumed: bool = False

    @torch.no_grad()
    def apply(self, policy, enabled=True):
        if self.consumed:
            raise RuntimeError('Producer-state credit has already been consumed.')
        self.consumed = True
        if policy.shape != (len(self.owners), self.sensitivity[0][0].shape[-1]):
            raise ValueError('Producer-state policy shape mismatch.')
        gradient = torch.zeros_like(policy)
        if enabled:
            for i, (leaves, sensitivities) in enumerate(zip(self.leaves, self.sensitivity)):
                for leaf, sensitivity in zip(leaves, sensitivities):
                    if leaf.grad is not None:
                        gradient[i].add_((leaf.grad[..., None] * sensitivity).sum((0, 1, 2, 3)))
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('Nonfinite delayed producer-state gradient.')
        if enabled:
            if policy.grad is None:
                policy.grad = gradient
            else:
                policy.grad.add_(gradient)
        return float(gradient.norm())


@dataclass(frozen=True)
class ProducerEligibility:
    owners: tuple[str, ...]
    sensitivity: tuple[tuple[torch.Tensor, ...], ...]
    clocks: torch.Tensor
    epochs: torch.Tensor

    @classmethod
    def zeros(cls, model, batch):
        owners = model.cfg.task_publication
        state = model.initial_state(batch)
        width = model.publisher.policy.shape[-1]
        return cls(owners, tuple(tuple(x.new_zeros(*x.shape, width) for x in node_contents(state.nodes[n]))
                                for n in owners),
                   torch.zeros(batch, len(owners), dtype=torch.long, device=state.epochs.device),
                   state.epochs[:, None].expand(batch, len(owners)).clone())

    def validate(self, model, batch):
        template = type(self).zeros(model, batch)
        if (self.owners != template.owners or self.clocks.shape != template.clocks.shape or
                self.epochs.shape != template.epochs.shape or self.clocks.dtype != torch.long or
                self.epochs.dtype != torch.long or self.clocks.device != template.clocks.device or
                self.epochs.device != template.epochs.device or bool((self.clocks < 0).any()) or
                bool((self.epochs < 0).any()) or len(self.sensitivity) != len(template.sensitivity)):
            raise ValueError('Malformed producer-state owners/clocks/epochs.')
        for actual, expected in zip(self.sensitivity, template.sensitivity):
            if len(actual) != len(expected):
                raise ValueError('Malformed producer-state layer count.')
            for value, target in zip(actual, expected):
                if (value.shape != target.shape or value.dtype != target.dtype or
                        value.device != target.device or not bool(torch.isfinite(value).all()) or value.requires_grad):
                    raise ValueError('Malformed producer-state sensitivity.')
        return self

    def capture(self, state):
        clocks = torch.stack([state.nodes[n].clock for n in self.owners], 1)
        if not torch.equal(self.clocks, clocks) or not torch.equal(
                self.epochs, state.epochs[:, None].expand_as(self.epochs)):
            raise ValueError('Producer-state credit clock/epoch mismatch.')
        nodes, leaves = dict(state.nodes), []
        for name in self.owners:
            old = state.nodes[name]
            values = tuple(x.detach().clone().requires_grad_(True) for x in node_contents(old))
            count = len(old.contextualizer)
            nodes[name] = NodeState(values[:count], values[count:], old.clock)
            leaves.append(values)
        boundary = ProducerBoundary(self.owners, tuple(leaves), self.sensitivity)
        return replace(state, nodes=nodes), boundary

    def storage_bytes(self):
        tensors = [self.clocks, self.epochs] + [t for owner in self.sensitivity for t in owner]
        return sum(t.numel()*t.element_size() for t in tensors)

    def to_dict(self):
        return dict(owners=self.owners, sensitivity=tuple(tuple(t.detach().clone() for t in owner)
                    for owner in self.sensitivity), clocks=self.clocks.clone(), epochs=self.epochs.clone())

    @classmethod
    def from_dict(cls, data, model, batch):
        return cls(tuple(data['owners']), tuple(tuple(t.detach().clone() for t in owner)
                   for owner in data['sensitivity']), data['clocks'].clone(), data['epochs'].clone()).validate(model, batch)


class ProducerRecorder:
    """Forward-only trace update, scoped hooks cleaned up even after failure.

    Existing stage batching notices module hooks and leaves these contextualizers
    on their ordinary per-node path. Inference and credit-disabled trainers have
    no hooks. The trace only holds each module's current transition until commit.
    """
    def __init__(self, model, trace):
        self.model, self.trace = model, trace
        self.values = {n: list(s) for n, s in zip(trace.owners, trace.sensitivity)}
        self.clocks, self.epochs = trace.clocks.clone(), trace.epochs.clone()
        self.modules = {n: tuple(level.memory for level in model.nodes[n].adapter.levels) +
                          tuple(block.memory for block in model.nodes[n].core.blocks) for n in trace.owners}
        self.pending, self.handles = {}, []

    @torch.no_grad()
    def recurrent(self, module, args, kwargs):
        if module in self.pending:
            raise ValueError('Repeated recurrent transition before owner commit.')
        reset_each = kwargs.get('reset_each', args[3] if len(args) > 3 else False)
        self.pending[module] = recurrence_factors(module, args[0], reset_each)

    @torch.no_grad()
    def commit(self, module, args, output):
        name, _, result, active, epochs = args[:5]
        cfg = args[9]
        i = self.trace.owners.index(name)
        previous, candidate = result['previous_state'], result['state']
        _, committed, activity = output
        reset = epochs != self.epochs[:, i]
        if bool(((epochs-self.epochs[:, i] < 0) | (epochs-self.epochs[:, i] > 1)).any()):
            raise ValueError('Producer-state epochs must advance one reset at a time.')
        expected_clock = self.clocks[:, i].masked_fill(reset, 0)
        if not torch.equal(previous.clock, expected_clock) or not torch.equal(committed.clock, previous.clock+active.long()):
            raise ValueError('Producer-state transition clock mismatch.')
        rows = active.nonzero().flatten()
        for j, (memory, old, new) in enumerate(zip(self.modules[name], node_contents(previous), node_contents(candidate))):
            e = self.values[name][j].masked_fill(reset[:, None, None, None, None], 0)
            transported = e
            if rows.numel():
                factors = self.pending.pop(memory)
                transported = e.index_copy(0, rows, transport(e[rows], factors))
            self.values[name][j] = commit_sensitivity(e, transported, old, new,
                activity['write_features'], activity['write_probs'], activity['write_actions'].bool(),
                activity['write_eligible'], active, cfg.publication_state_commit).detach()
        self.clocks[:, i] = committed.clock
        self.epochs[:, i] = epochs

    def __enter__(self):
        try:
            for modules in self.modules.values():
                for module in modules:
                    self.handles.append(module.register_forward_pre_hook(self.recurrent, with_kwargs=True))
            self.handles.append(self.model.publisher.register_forward_hook(self.commit))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()

    def finish(self, state):
        if self.pending:
            raise ValueError('Uncommitted producer transitions.')
        trace = ProducerEligibility(self.trace.owners, tuple(tuple(self.values[n]) for n in self.trace.owners),
                                    self.clocks, self.epochs)
        # Check identities without creating another set of autograd leaves.
        if not torch.equal(trace.clocks, torch.stack([state.nodes[n].clock for n in trace.owners], 1)) or not torch.equal(
                trace.epochs, state.epochs[:, None].expand_as(trace.epochs)):
            raise ValueError('Producer-state final identity mismatch.')
        return trace


@dataclass
class StateGraphBoundary:
    publication: object
    producer: ProducerBoundary

    def apply(self, policy, enabled=True):
        return self.publication.apply(policy, enabled)

    def apply_payload(self, adapter, enabled=True):
        return self.publication.apply_payload(adapter, enabled)

    def apply_state(self, policy, enabled=True):
        return self.producer.apply(policy, enabled)


@dataclass(frozen=True)
class StateGraphEligibility:
    publication: object
    producer: ProducerEligibility

    def capture(self, state):
        state, publication = self.publication.capture(state)
        state, producer = self.producer.capture(state)
        return state, StateGraphBoundary(publication, producer)

    def storage_bytes(self):
        return self.publication.storage_bytes() + self.producer.storage_bytes()

    def to_dict(self):
        return dict(publication=self.publication.to_dict(), producer=self.producer.to_dict())


def make_state_graph_eligibility(model, batch, saved=None):
    publication = make_graph_eligibility(model, batch, saved['publication'] if saved is not None else None)
    producer = (ProducerEligibility.from_dict(saved['producer'], model, batch) if saved is not None
                else ProducerEligibility.zeros(model, batch))
    return StateGraphEligibility(publication, producer)


def recorded_segment(model, values, reset, state, producer):
    """Functional segment, including local trace updates, safe to recompute.

    A fresh recorder is required on every invocation. Reusing a recorder or
    installing hooks only on the first pass changes the arithmetic/trace clocks
    during activation recomputation. Its hooks are removed even when
    recomputation raises an exception.
    """
    with ProducerRecorder(model, producer) as recorder:
        result = model(values, state=state, reset=reset, return_write_activity=True)
        advanced = recorder.finish(result['state'])
    return result, advanced


def run_recorded_segment(model, values, reset, state, producer, checkpoint_segments=False):
    if checkpoint_segments and torch.is_grad_enabled():
        from .activation_checkpoint import checkpoint_tree
        result, advanced = checkpoint_tree(recorded_segment, model, values, reset, state, producer)
        # Conditional eligibility is never part of the exact content graph.
        from .ports import map_tensors
        return result, map_tensors(advanced, lambda value: value.detach())
    return recorded_segment(model, values, reset, state, producer)


def state_window_forward(model, chunk, state, credit, truncate_every=0, *, checkpoint_segments=False):
    length = chunk.values.shape[1]
    if truncate_every < 0 or (truncate_every and length % truncate_every):
        raise ValueError('Truncation must be zero or a positive divisor of the window.')
    span = truncate_every or length
    results, boundaries = [], []
    for start in range(0, length, span):
        state, boundary = credit.capture(state)
        boundaries.append(boundary)
        result, producer = run_recorded_segment(model, chunk.values[:, start:start+span],
            chunk.reset[:, start:start+span], state, credit.producer, checkpoint_segments)
        credit = StateGraphEligibility(credit.publication.advance(result['write_activity']), producer)
        results.append(result)
        state = result['state'].detach()
    return dict(logits=torch.cat([r['logits'] for r in results], 1),
        write_probs=torch.cat([r['write_activity']['write_probs'] for r in results], 1),
        aux=torch.stack([model.cfg.auxiliary_loss_weight*r['aux'] +
                         model.cfg.hat_balance_weight*r['hat_aux'] for r in results]).mean() / .001,
        state=state, external_events=sum(r['external_events'] for r in results),
        expert_executions=sum(r['expert_executions'] for r in results)), credit, boundaries
