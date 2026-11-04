"""Directed value/context maps and synchronous stage aggregation."""
from dataclasses import dataclass, replace
import math
import torch
from torch import nn


@dataclass(frozen=True)
class EdgeSubstitution:
    payload: torch.Tensor       # [B,T,width]
    memberships: torch.Tensor  # [B,T,3,groups]; original header must not leak through


class EdgeProjection(nn.Module):
    def __init__(self, cfg, delayed=False):
        super().__init__()
        self.value = nn.Linear(cfg.width, cfg.width, bias=False)
        self.context = nn.Linear(3*cfg.groups, 3*cfg.groups, bias=False)
        self.gate = nn.Linear(cfg.width, 1)
        nn.init.normal_(self.value.weight, std=.02)
        nn.init.normal_(self.context.weight, std=.02)
        nn.init.zeros_(self.gate.weight)
        # A closed graph repeatedly feeds normalized outputs back into itself.
        # Start delayed paths weakly open, preserving all gradients and allowing
        # task learning to strengthen them. Forward-path gates start at 0.5.
        initial = cfg.feedback_gate_init if delayed else .5
        nn.init.constant_(self.gate.bias, math.log(initial/(1-initial)))

    def forward(self, payload, memberships, bias=0.):
        gate = (self.gate(payload)+bias).sigmoid()
        value = gate*self.value(payload)
        context = self.context(memberships.flatten(1)).reshape_as(memberships)
        return value, gate[:, :, None]*context


class GraphConnections(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.edges = nn.ModuleDict({edge.key: EdgeProjection(cfg, delayed=edge.kind != 'forward') for edge in cfg.edges})
        self.ingress = nn.ModuleDict({n.name: nn.Linear(cfg.width, cfg.width, bias=False)
                                      for n in cfg.nodes if n.view is not None})
        for layer in self.ingress.values():
            nn.init.normal_(layer.weight, std=.02)
        self.incoming = {n.name: tuple(e for e in cfg.edges if e.destination == n.name) for n in cfg.nodes}

    def gather(self, spec, current, delayed, ingress, active, event_index, edge_masks, substitutions, edge_bias=None):
        template = current[spec.name].payload
        values = torch.zeros_like(template)
        context = template.new_zeros(template.shape[0], 3, self.cfg.groups)
        count = template.new_zeros(template.shape[0])
        dependencies = []
        if ingress is not None:
            used = ingress.valid & active
            values = values + self.ingress[spec.name](ingress.payload)*used[:, None]
            count = count + used.to(count.dtype)
            dependencies.append(ingress.dependency(f'{ingress.source}__{spec.name}', used))
        for edge in self.incoming[spec.name]:
            if edge.kind != 'forward' and delayed is None:
                continue
            publication = (current if edge.kind == 'forward' else delayed)[edge.source]
            used = publication.valid & active
            if edge.key in edge_masks:
                used = used & edge_masks[edge.key][:, event_index]
            payload, memberships = publication.payload, publication.header.memberships
            intervention = 'none'
            if edge.key in substitutions:
                substitute = substitutions[edge.key]
                payload = substitute.payload[:, event_index]
                memberships = substitute.memberships[:, event_index]
                intervention = 'substitution'
            value, evidence = self.edges[edge.key](payload, memberships, (edge_bias or {}).get(edge.key, 0.))
            values = values + value*used[:, None]
            context = context + evidence*used[:, None, None]
            count = count + used.to(count.dtype)
            dependencies.append(replace(publication.dependency(edge.key, used), intervention=intervention,
                                        substituted=torch.full_like(used, intervention == 'substitution')))
        scale = count.clamp_min(1).sqrt()
        return values/scale[:, None], context/scale[:, None, None], tuple(dependencies)
