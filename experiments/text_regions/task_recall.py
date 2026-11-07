"""Bounded, role-preserving recall ports for the regional expert composer.

Slots are current views of versioned publications, not a second replay store.
Each can select an owner or null. Composer feedback changes the next request;
current recall usage can influence the next publication decision. Task loss
trains these controls through the live graph, including the existing delayed
publication-credit boundary. No source labels or query annotations enter here.
Historic credit remains on the write/payload controls; request-network and
composer-feedback derivatives span the live training window only.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RecallWorkspace(nn.Module):
    def __init__(self, width, semantic_width, address_dim, slots, policy_features):
        super().__init__()
        if min(width, semantic_width, address_dim, slots, policy_features) < 1:
            raise ValueError('Recall dimensions must be positive.')
        self.slots = slots
        self.slot_queries = nn.Parameter(torch.randn(slots, address_dim) * .15)
        self.slot_null = nn.Parameter(torch.zeros(slots))
        self.query_feedback = nn.Linear(width, address_dim, bias=False)
        self.null_feedback = nn.Linear(width, slots, bias=False)
        self.usage_feedback = nn.Linear(2, policy_features, bias=False)
        self.coverage_logit = nn.Parameter(torch.full((slots,), math.log(math.expm1(1.))))
        self.payload_merge = nn.Linear(slots * width, width, bias=False)
        self.semantic_merge = nn.Linear(slots * semantic_width, semantic_width, bias=False)
        nn.init.zeros_(self.query_feedback.weight)
        nn.init.zeros_(self.null_feedback.weight)
        nn.init.zeros_(self.usage_feedback.weight)
        with torch.no_grad():
            self.payload_merge.weight.copy_(torch.cat([torch.eye(width) / slots] * slots, -1))
            self.semantic_merge.weight.copy_(torch.cat([torch.eye(semantic_width) / slots] * slots, -1))

    def select(self, base_query, keys, memory, valid, null_score, feedback, *,
               hard=True, feedback_enabled=True, blocked=False):
        """Return [B,S,M+1] selection and distinct [B,S,D] content ports.

        Hard forward choices use a softmax straight-through derivative. Invalid
        owners are impossible, including an entirely empty publication bank.
        Coverage discourages repeated owners without forbidding useful reuse.
        """
        if valid.dtype != torch.bool or valid.shape != memory.shape[:2]:
            raise ValueError('Recall validity must be boolean [batch, owners].')
        if keys.shape[:2] != valid.shape or base_query.shape != keys[:, 0].shape:
            raise ValueError('Recall query/key shapes disagree.')
        b, m, _ = memory.shape
        demand = F.layer_norm(feedback, (feedback.shape[-1],))
        query_delta = self.query_feedback(demand) if feedback_enabled else torch.zeros_like(base_query)
        null_delta = self.null_feedback(demand) if feedback_enabled else memory.new_zeros(b, self.slots)
        probabilities, selections = [], []
        coverage = memory.new_zeros(b, m)
        strengths = F.softplus(self.coverage_logit)
        for slot in range(self.slots):
            query = F.normalize(base_query + query_delta + self.slot_queries[slot], dim=-1)
            scores = 4 * (query[:, None] * keys).sum(-1) - strengths[slot] * coverage
            scores = scores.masked_fill(~valid, -torch.inf)
            null = null_score + self.slot_null[slot] + null_delta[:, slot:slot+1]
            probability = torch.cat((scores, null), -1).softmax(-1)
            selected = probability
            if hard:
                choice = F.one_hot(probability.argmax(-1), m + 1).to(probability.dtype)
                # Parentheses preserve exact hard values; backward is softmax.
                selected = choice + (probability - probability.detach())
            probabilities.append(probability)
            selections.append(selected)
            coverage = coverage + selected[:, :m]
        weights = torch.stack(selections, 1)
        soft_weights = torch.stack(probabilities, 1)
        effective = weights[..., :m] * (0. if blocked else 1.)
        slots = torch.einsum('bsm,bmd->bsd', effective, memory)
        # Potentially threshold-dependent emergent feature: separate role ports
        # can support relational recall after sufficient task exposure. Preserve
        # the learned projections and pooled-layout ablation until larger runs;
        # small-pilot inactivity is not grounds for deleting this mechanism.
        return dict(contents=slots, weights=weights, probabilities=soft_weights,
                    effective=effective, coverage_strength=strengths)

    def compose(self, payloads, semantics, layout='distinct'):
        """Apply slot-specific input maps at the expert composer's boundary."""
        if layout not in ('distinct', 'pooled'):
            raise ValueError('Recall layout must be distinct or pooled.')
        if layout == 'pooled':
            # Same parameters/selection work, but item identity is lost before
            # the learned slot maps. This is a parameter-matched ablation.
            payloads = payloads.mean(1, keepdim=True).expand_as(payloads)
            semantics = semantics.mean(1, keepdim=True).expand_as(semantics)
        return self.payload_merge(payloads.flatten(1)), self.semantic_merge(semantics.flatten(1))

    def write_feedback(self, probabilities, enabled=True):
        usage = probabilities[..., :-1]
        features = torch.stack((usage.mean(1), usage.amax(1)), -1)
        # Potentially threshold-dependent emergent feature: task-trained demand
        # may connect retrieval utility to publication admission only after both
        # readers and experts specialize. Retain an explicit feedback ablation.
        result = self.usage_feedback(features)
        return result if enabled else result * 0
