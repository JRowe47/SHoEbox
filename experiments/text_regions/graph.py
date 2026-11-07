"""Directed value/context maps and synchronous stage aggregation."""
from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from .runtime import cached
from .regional_recall import RegionalRecall


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

    def initialize_recall(self):
        # Called after the legacy model finishes initialization. New ports do not
        # perturb common weights or the caller RNG, and default-off adds nothing.
        self.recall = nn.ModuleDict()
        if self.cfg.recall_slots:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.cfg.seed + 9157)
                names = self.cfg.recall_nodes or tuple(n for n, edges in self.incoming.items() if edges)
                self.recall.update({name: RegionalRecall(self.cfg) for name in names})

    def gather(self, spec, current, delayed, ingress, active, event_index, edge_masks, substitutions,
               edge_bias=None, recall_feedback=None, recall_control='intact', recall_capture=None):
        if spec.name in getattr(self, 'recall', {}):
            return self.gather_recall(spec, current, delayed, ingress, active, event_index,
                edge_masks, substitutions, edge_bias, recall_feedback, recall_control, recall_capture)
        template = current[spec.name].payload
        values = torch.zeros_like(template)
        context = template.new_zeros(template.shape[0], 3, self.cfg.groups)
        count = template.new_zeros(template.shape[0])
        dependencies = []
        if ingress is not None:
            used = ingress.valid & active
            projection = self.ingress[spec.name]
            value = cached(projection, ingress.payload, lambda: projection(ingress.payload))
            values = values + value*used[:, None]
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
            projection = self.edges[edge.key]
            bias = (edge_bias or {}).get(edge.key, 0.)
            value, evidence = cached(projection, (payload, memberships, bias),
                                     lambda: projection(payload, memberships, bias))
            values = values + value*used[:, None]
            context = context + evidence*used[:, None, None]
            count = count + used.to(count.dtype)
            dependencies.append(replace(publication.dependency(edge.key, used), intervention=intervention,
                                        substituted=torch.full_like(used, intervention == 'substitution')))
        scale = count.clamp_min(1).sqrt()
        return values/scale[:, None], context/scale[:, None, None], tuple(dependencies)

    def gather_recall(self, spec, current, delayed, ingress, active, event_index,
                      edge_masks, substitutions, edge_bias, feedback, control, capture):
        template = current[spec.name].payload
        incoming = torch.zeros_like(template)
        dependencies, values, contexts, masks = [], [], [], []
        if ingress is not None:
            used = ingress.valid & active
            layer = self.ingress[spec.name]
            incoming = cached(layer, ingress.payload, lambda: layer(ingress.payload)) * used[:, None]
            dependencies.append(ingress.dependency(f'{ingress.source}__{spec.name}', used))
        edges = self.incoming[spec.name]
        for edge in edges:
            available = edge.kind == 'forward' or delayed is not None
            publication = (current if edge.kind == 'forward' or delayed is None else delayed)[edge.source]
            used = publication.valid & active & available
            if edge.key in edge_masks:
                used = used & edge_masks[edge.key][:, event_index]
            payload, memberships = publication.payload, publication.header.memberships
            intervention = 'none'
            if edge.key in substitutions:
                sub = substitutions[edge.key]
                payload, memberships = sub.payload[:, event_index], sub.memberships[:, event_index]
                intervention = 'substitution'
            projection = self.edges[edge.key]
            bias = (edge_bias or {}).get(edge.key, 0.)
            value, context = cached(projection, (payload, memberships, bias),
                                    lambda: projection(payload, memberships, bias))
            values.append(value); contexts.append(context); masks.append(used)
            # The query inspects candidate content even when its slot is not
            # selected. Dependencies therefore record examined valid inputs;
            # capture records the actual content selections separately.
            dependencies.append(replace(publication.dependency(edge.key, used),
                intervention=intervention, substituted=torch.full_like(used, intervention == 'substitution')))
        if feedback is None:
            feedback = torch.zeros_like(template)
        payload, context, recall = self.recall[spec.name](torch.stack(values, 1),
            torch.stack(contexts, 1), torch.stack(masks, 1), incoming, feedback,
            layout=self.cfg.recall_layout, hard=self.cfg.recall_hard,
            feedback_enabled=self.cfg.recall_feedback, control=control)
        if capture is not None:
            edge_deps = dependencies[-len(edges):]
            capture.update(weights=recall['weights'], probabilities=recall['probabilities'],
                valid=torch.stack(masks, 1),
                versions=torch.stack([d.version for d in edge_deps], 1),
                occurrences=torch.stack([d.occurrence for d in edge_deps], 1),
                round_indices=torch.stack([d.round_index for d in edge_deps], 1))
        return payload, context, tuple(dependencies)
