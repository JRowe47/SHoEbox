"""Recall over configured incoming ports without changing graph scheduling.

The graph supplies projected edge payloads and semantic headers. Current forward
ports and round-start delayed ports keep their original causal meaning. The
request may inspect every available candidate; only selected content enters the
expert input maps. This is sparse content selection, not sparse candidate scoring.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .recall import RecallWorkspace


class RegionalRecall(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width, self.groups = cfg.width, cfg.groups
        semantic = 3 * cfg.groups
        self.request = nn.Linear(cfg.width, cfg.address_dim, bias=False)
        self.key = nn.Linear(cfg.width + semantic, cfg.address_dim, bias=False)
        self.null = nn.Linear(cfg.width, 1)
        nn.init.zeros_(self.null.weight)
        nn.init.constant_(self.null.bias, -1.)
        self.workspace = RecallWorkspace(cfg.width, semantic, cfg.address_dim, cfg.recall_slots)

    def forward(self, values, semantics, valid, ingress, feedback, *,
                layout='distinct', hard=True, feedback_enabled=True, control='intact'):
        if control not in ('intact', 'pooled', 'no_feedback', 'blocked'):
            raise ValueError('Unknown regional recall control.')
        # Mask both key and value inputs; invalid or substituted headers cannot
        # affect any selection. No gradients pass into masked publication rows.
        memory = torch.cat((values, semantics.flatten(2)), -1)
        memory = torch.where(valid[..., None], memory, 0.)
        count = valid.sum(1, keepdim=True).clamp_min(1)
        request = ingress + memory[..., :self.width].sum(1) / count.sqrt()
        request = F.layer_norm(request, (self.width,))
        keys = F.normalize(self.key(memory), dim=-1)
        recall = self.workspace.select(self.request(request), keys, memory, valid,
            self.null(request), feedback, hard=hard,
            feedback_enabled=feedback_enabled and control != 'no_feedback',
            blocked=control == 'blocked')
        content = recall['contents']
        payload, context = self.workspace.compose(content[..., :self.width],
            content[..., self.width:], 'pooled' if control == 'pooled' else layout)
        # Potentially threshold-dependent emergent feature: a region can learn
        # different input roles and request changes from its previous output.
        # Keep pooled/no-feedback ablations available through broader training.
        return ingress + payload, context.reshape(-1, 3, self.groups), recall
