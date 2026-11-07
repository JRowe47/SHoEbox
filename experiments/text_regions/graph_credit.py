"""Owner/version/epoch-bound conditional publication credit in a staged graph.

The existing estimator is applied to each region's actual execution order. It
restores conditional gate and low-rank payload credit across detach boundaries;
historical feature, demand, candidate and recurrent Jacobians remain truncated.
There is no replay buffer. Receipts are single-use and never serialized.
"""
from dataclasses import dataclass
import torch
from .task_credit import WriteEligibility
from .task_payload import PayloadEligibility, PayloadWriteEligibility


@dataclass(frozen=True)
class GraphEligibility:
    trace: object
    epochs: torch.Tensor

    @property
    def write(self):
        return self.trace.write if isinstance(self.trace, PayloadWriteEligibility) else self.trace

    def capture(self, state):
        versions = torch.stack([state.ports.latest[n].version for n in self.write.owners], 1)
        if not torch.equal(versions, self.write.versions) or not torch.equal(state.epochs, self.epochs):
            raise ValueError('Graph credit publication version/epoch mismatch.')
        boundary = self.trace.capture(state.memory_tensor(self.write.owners))
        return state.with_memory_tensor(self.write.owners, boundary.memory), boundary

    def advance(self, activity):
        return GraphEligibility(self.trace.advance_chunk(activity),
                                self.epochs + activity['reset'].long().sum(1))

    def storage_bytes(self):
        return self.trace.storage_bytes() + self.epochs.numel() * self.epochs.element_size()

    def to_dict(self):
        return dict(trace=self.trace.to_dict(), epochs=self.epochs.detach().clone())


def make_graph_eligibility(model, batch, saved=None):
    c, p = model.cfg, model.publisher
    if p is None:
        raise ValueError('Graph credit requires task publishers.')
    source = saved['trace'] if saved is not None else None
    write_source = source['write'] if source is not None and c.publication_payload_rank else source
    write = WriteEligibility.from_dict(write_source) if write_source is not None else WriteEligibility.zeros(
        batch, p.owners, p.memory_dim, c.publication_features, p.policy.device, p.policy.dtype)
    if write.owners != p.owners or write.sensitivity.shape != (
            batch, len(p.owners), p.memory_dim, c.publication_features + 1):
        raise ValueError('Graph credit owner/shape mismatch.')
    trace = write
    if c.publication_payload_rank:
        payload = PayloadEligibility.from_dict(source['payload']) if source is not None else PayloadEligibility.zeros(
            batch, p.owners, c.publication_payload_rank, p.policy.device, p.policy.dtype)
        if payload.owners != p.owners or payload.features.shape != (batch, len(p.owners), c.publication_payload_rank
                ) or not torch.equal(payload.versions, write.versions):
            raise ValueError('Graph payload credit owner/shape/version mismatch.')
        trace = PayloadWriteEligibility(write, payload)
    epochs = saved['epochs'].detach().clone() if saved is not None else torch.zeros(
        batch, device=p.policy.device, dtype=torch.long)
    if epochs.shape != (batch,) or epochs.dtype != torch.long or bool((epochs < 0).any()
            ) or epochs.device != p.policy.device:
        raise ValueError('Graph credit epoch mismatch.')
    return GraphEligibility(trace, epochs)


def graph_window_forward(model, chunk, state, credit, truncate_every=0):
    length = chunk.values.shape[1]
    if truncate_every < 0 or (truncate_every and length % truncate_every):
        raise ValueError('Truncation must be zero or a positive divisor of the window.')
    span = truncate_every or length
    results, boundaries = [], []
    for start in range(0, length, span):
        state, boundary = credit.capture(state)
        boundaries.append(boundary)
        result = model(chunk.values[:, start:start+span], state=state,
                       reset=chunk.reset[:, start:start+span], return_write_activity=True)
        credit = credit.advance(result['write_activity'])
        results.append(result)
        state = result['state'].detach()
    return dict(logits=torch.cat([r['logits'] for r in results], 1),
        write_probs=torch.cat([r['write_activity']['write_probs'] for r in results], 1),
        # MixedRun applies .001 to this term; retain the graph's configured losses.
        aux=torch.stack([model.cfg.auxiliary_loss_weight*r['aux'] +
                         model.cfg.hat_balance_weight*r['hat_aux'] for r in results]).mean() / .001,
        state=state, external_events=sum(r['external_events'] for r in results),
        expert_executions=sum(r['expert_executions'] for r in results)), credit, boundaries
