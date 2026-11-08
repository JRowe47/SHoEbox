"""Task-trained admission and bounded demand feedback for ordinary graph nodes.

Each region decides independently because forward stages execute sequentially.
Demand is aggregated after a complete round and used only in the next round.
Candidate, stored content and prior demand may jointly acquire useful admission
rules at larger data scales: a potentially threshold-dependent emergent feature.
Keep these controls separately ablatable instead of hard-coding a write budget.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .ports import publish
from .task_payload import PayloadAdapter
from .task_state_commit import commit_producer_state


class GraphPublicationPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.owners = cfg.task_publication
        self.indices = {name: i for i, name in enumerate(self.owners)}
        self.memory_dim = cfg.width + 3 * cfg.groups
        self.candidate_features = nn.Linear(cfg.width, cfg.publication_features)
        self.memory_features = nn.Linear(self.memory_dim + 4, cfg.publication_features, bias=False)
        self.demand_features = nn.Linear(2, cfg.publication_features, bias=False)
        nn.init.zeros_(self.memory_features.weight)
        nn.init.zeros_(self.demand_features.weight)
        self.policy = nn.Parameter(torch.randn(len(self.owners), cfg.publication_features + 1) * .15)
        with torch.no_grad():
            self.policy[:, -1] = torch.linspace(-.15, .15, len(self.owners))
        self.payload_adapter = PayloadAdapter(len(self.owners), cfg.width, cfg.publication_payload_rank
                                             ) if cfg.publication_payload_rank else None

    def forward(self, name, previous, result, active, epochs, occurrence, round_index,
                dependencies, demand, cfg, override=None):
        i = self.indices[name]
        candidate = result['latent']
        old = torch.cat((previous.payload, previous.header.memberships.flatten(1)), -1)
        stats = torch.stack((previous.valid.to(candidate.dtype),
            (occurrence - previous.occurrence).clamp_min(0).to(candidate.dtype).log1p(),
            F.cosine_similarity(candidate.detach(), previous.payload.detach(), dim=-1),
            F.cosine_similarity(result['header'].memberships.detach().flatten(1),
                                previous.header.memberships.detach().flatten(1), dim=-1)), -1)
        features = self.candidate_features(candidate) + self.memory_features(torch.cat((old.detach(), stats), -1))
        if cfg.publication_feedback:
            features = features + self.demand_features(demand)
        features = features.tanh()
        logit = (features * self.policy[i, :-1]).sum(-1) + self.policy[i, -1]
        probability = logit.sigmoid()
        bootstrap = active & ~previous.valid & cfg.publication_bootstrap
        writing = active & (probability.ge(.5) | bootstrap)
        eligible = active & ~bootstrap
        if override is not None:
            writing = active & (override == 'always')
            eligible = torch.zeros_like(active)
        payload_features = None
        if self.payload_adapter is not None:
            adapter = self.payload_adapter
            payload_features = (candidate @ adapter.basis.T).tanh()
            candidate = candidate + (payload_features * adapter.coefficients[i]) @ adapter.basis
        packed = torch.cat((candidate, result['header'].memberships.flatten(1)), -1)
        publication = publish(previous, candidate, result['header'], writing,
                              epochs, occurrence, round_index, dependencies)
        tangent = ((probability - probability.detach()) * eligible)[:, None]
        publication = replace(publication,
            payload=publication.payload + tangent * (candidate - previous.payload),
            header=replace(publication.header, memberships=publication.header.memberships +
                tangent[:, :, None] * (result['header'].memberships - previous.header.memberships)))
        node_state = result['state']
        if cfg.publication_state_commit == 'publication' and not (
                cfg.memory_context.enabled and cfg.memory_context.decouple_working_state):
            node_state = commit_producer_state(result['previous_state'], node_state,
                                               writing, probability, eligible)
        activity = dict(write_probs=probability, write_actions=writing.to(candidate.dtype),
            write_features=features, memory_before=old, memory_candidates=packed,
            memory_versions=publication.version, write_eligible=eligible, write_active=active,
            demand=demand, bootstrap=bootstrap, logit=logit)
        if payload_features is not None:
            activity['payload_features'] = payload_features
        return publication, node_state, activity


def accumulate_demand(storage, connections, name, record, dependencies, active, blocked=False):
    """Mean and peak expected selection per valid incoming edge, current round.

    A reader contributes only while active and the source edge is valid. An
    owner with no such readers receives zero demand, including blocked recall.
    Multiple edges from one owner intentionally represent distinct pathways.
    """
    for i, edge in enumerate(connections.incoming[name]):
        if edge.source not in storage:
            continue
        if record is not None:
            valid = record['valid'][:, i] & active
            probability = record['probabilities'][..., i]
            values = torch.stack((probability.mean(-1), probability.amax(-1)), -1)
        else:
            valid = next((d.consumed for d in dependencies if d.edge == edge.key),
                         torch.zeros_like(active)) & active
            values = valid[:, None].expand(-1, 2).float()
        if blocked:
            valid = torch.zeros_like(valid)
        storage[edge.source].append((values, valid))


def finish_demand(storage, previous):
    demand = {}
    for name, readings in storage.items():
        if not readings:
            demand[name] = torch.zeros_like(previous[name])
            continue
        values = torch.stack([r[0] for r in readings], 1)
        mask = torch.stack([r[1] for r in readings], 1)
        mean = (values[..., 0] * mask).sum(1) / mask.sum(1).clamp_min(1)
        peak = (values[..., 1] * mask).amax(1)
        demand[name] = torch.stack((mean, peak), -1)
    return demand
