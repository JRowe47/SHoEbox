"""Event-major reference engine: ingest once, run full pipelines K times, decode."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from ..hat_minkron.events import ByteBatch
from ..hat_minkron.hat import HAT
from ..hat_minkron.semantic_core import ExpertBank
from .config import GraphConfig
from .graph import GraphConnections
from .node import RecurrentNode, masked_ingress
from .ports import Publication, SemanticHeader, TensorPorts, publish
from .state import GraphState


class RegionLanguageModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or GraphConfig()
        self.cfg.validate()
        c = self.cfg.hat_config()
        self.hat = HAT(c)
        self.shared_banks = nn.ModuleList(ExpertBank(c.child_config(), self.cfg.shared_per_block) for _ in range(3))
        self.nodes = nn.ModuleDict({n.name: RecurrentNode(n, self.cfg, self.shared_banks, i)
                                   for i, n in enumerate(self.cfg.nodes)})
        self.connections = GraphConnections(self.cfg)
        self.readout_mix = nn.Linear(len(self.cfg.readout)*self.cfg.width, self.cfg.width, bias=False)
        self.readout_norm = nn.LayerNorm(self.cfg.width)
        self.readout = nn.Linear(self.cfg.width, 256, bias=False)
        nn.init.normal_(self.readout.weight, std=.02)
        self.specs = {n.name: n for n in self.cfg.nodes}
        self.stages = self.cfg.stages()

    def initial_state(self, batch, device=None):
        if batch < 1:
            raise ValueError('Batch size must be positive.')
        device = device or self.readout.weight.device
        c, dtype = self.cfg, self.readout.weight.dtype
        def empty(name):
            zero = torch.zeros(batch, dtype=torch.long, device=device)
            header = SemanticHeader(name, torch.zeros(batch, 3, c.groups, device=device, dtype=dtype),
                                    torch.zeros(batch, 3, c.active_groups, dtype=torch.long, device=device),
                                    torch.zeros(batch, 3, c.active_laces, dtype=torch.long, device=device))
            return Publication(name, 'latent', torch.zeros(batch, c.width, device=device, dtype=dtype),
                               torch.zeros(batch, dtype=torch.bool, device=device), zero, zero, zero, zero, header)
        names = tuple(self.nodes) + tuple('@'+view for view in ('text', *c.views))
        return GraphState({view: self.hat.initial_state(batch) for view in ('text', *c.views)},
                          {name: node.initial_state(batch, device, dtype) for name, node in self.nodes.items()},
                          TensorPorts({name: empty(name) for name in names}),
                          torch.zeros(batch, dtype=torch.long, device=device),
                          torch.zeros(batch, dtype=torch.long, device=device),
                          torch.zeros(batch, dtype=torch.long, device=device))

    def _prepare(self, inputs, state, availability, node_masks, edge_masks, substitutions, reset):
        controlled = isinstance(inputs, dict)
        if controlled and set(inputs) != set(self.cfg.views):
            raise ValueError(f'Controlled inputs must supply exactly these views: {self.cfg.views}.')
        supplied = inputs if controlled else {'text': inputs}
        events = {k: v if isinstance(v, ByteBatch) else ByteBatch.static(v) for k, v in supplied.items()}
        first = next(iter(events.values()))
        b, t = first.values.shape
        for e in events.values():
            e.validate(self.hat.cfg.max_links, self.hat.cfg.max_slots)
            if e.values.shape != (b, t) or e.values.device != first.values.device:
                raise ValueError('All aligned views must have the same [B,T] shape and device.')
        state = state or self.initial_state(b, first.values.device)
        if state.events_seen.shape != (b,) or state.events_seen.device != first.values.device:
            raise ValueError('State batch/device mismatch.')
        if set(state.nodes) != set(self.nodes):
            raise ValueError('State topology mismatch.')
        mode = 'views' if controlled else 'text'
        if state.input_mode and state.input_mode != mode:
            raise ValueError('Start a fresh state when switching between text and controlled-view inputs.')
        state = replace(state, input_mode=mode)
        for masks, allowed in ((availability, self.cfg.views), (node_masks, self.nodes),
                               (edge_masks, self.connections.edges)):
            if set(masks) - set(allowed):
                raise ValueError('Unknown view, node or edge in a mask.')
            for mask in masks.values():
                if mask.dtype != torch.bool or mask.shape != (b, t) or mask.device != first.values.device:
                    raise ValueError('Masks must be bool [B,T] on the input device.')
        if set(substitutions) - set(self.connections.edges):
            raise ValueError('Unknown substituted edge.')
        for sub in substitutions.values():
            if sub.payload.shape != (b, t, self.cfg.width) or sub.memberships.shape != (b, t, 3, self.cfg.groups):
                raise ValueError('A substitution replaces both payload and local memberships.')
            if sub.payload.device != first.values.device or sub.memberships.device != first.values.device:
                raise ValueError('Substitution device mismatch.')
        if reset is not None:
            if reset.dtype != torch.bool or reset.shape not in ((b,), (b, t)) or reset.device != first.values.device:
                raise ValueError('Reset must be bool [B] or [B,T] on the input device.')
        return events, state, b, t, controlled

    def forward(self, inputs, targets=None, state=None, reset=None, rounds=None, feedback=None,
                availability=None, node_masks=None, edge_masks=None, substitutions=None,
                trace=False, reverse_independent=False):
        availability, node_masks = availability or {}, node_masks or {}
        edge_masks, substitutions = edge_masks or {}, substitutions or {}
        events, state, b, t, controlled = self._prepare(inputs, state, availability, node_masks,
                                                       edge_masks, substitutions, reset)
        k_total = self.cfg.rounds if rounds is None else rounds
        if not isinstance(k_total, int) or k_total < 1:
            raise ValueError('Rounds must be a positive integer.')
        feedback = self.cfg.recurrent_feedback if feedback is None else feedback
        device = state.events_seen.device
        all_active = torch.ones(b, dtype=torch.bool, device=device)
        logits, records = [], []
        aux = self.readout.weight.new_zeros(())
        hat_aux = aux
        activations = ingress_activations = executed = 0
        for event_index in range(t):
            if reset is not None and (reset.ndim == 2 or event_index == 0):
                rows = reset[:, event_index] if reset.ndim == 2 else reset
                if bool(rows.any()):
                    state = state.reset(rows)
            available = {v: availability[v][:, event_index] if v in availability else all_active for v in self.cfg.views}
            ingress_states, staged_ingress = dict(state.ingress), {}
            for view, sequence in events.items():
                active = available[view] if controlled else torch.stack(tuple(available.values())).any(0)
                event = sequence.slice(event_index, event_index+1)
                payload, hstate, haux = masked_ingress(self.hat, event, state.ingress[view], active)
                ingress_states[view] = hstate
                previous = state.ports.latest['@'+view]
                publication = publish(previous, payload, previous.header, active, state.epochs, state.events_seen,
                                      torch.full_like(state.events_seen, -1), (), event.links[:, 0], event.slots[:, 0])
                # Absent inputs invalidate this occurrence; they never read a stale byte.
                publication = replace(publication, valid=active.clone())
                state.ports.stage(staged_ingress, publication)
                hat_aux = hat_aux + haux*int(active.sum())
                ingress_activations += int(active.sum())
            state = replace(state, ingress=ingress_states, ports=state.ports.commit(staged_ingress))
            for k in range(k_total):
                delayed = state.ports.snapshot() if feedback else None
                for stage in self.stages:
                    current, staged, node_states = state.ports.snapshot(), {}, dict(state.nodes)
                    for name in reversed(stage) if reverse_independent else stage:
                        spec = self.specs[name]
                        active = node_masks[name][:, event_index] if name in node_masks else all_active
                        ingress = None
                        if spec.view is not None:
                            ingress = current['@'+spec.view if controlled else '@text']
                            ingress = replace(ingress, valid=ingress.valid & available[spec.view])
                        payload, context, dependencies = self.connections.gather(
                            spec, current, delayed, ingress, active, event_index, edge_masks, substitutions)
                        result = self.nodes[name](payload, context, state.nodes[name], active, trace)
                        node_states[name] = result['state']
                        publication = publish(current[name], result['latent'], result['header'], active,
                                              state.epochs, state.events_seen, torch.full_like(state.events_seen, k),
                                              dependencies)
                        state.ports.stage(staged, publication)
                        weight = int(active.sum())
                        aux = aux + result['aux']*weight
                        hat_aux = hat_aux + result['hat_aux']*weight
                        activations += weight
                        executed += result['executions']
                        if trace:
                            records.append({'node': name, 'region': spec.region, 'event': event_index, 'round': k,
                                            'occurrence': state.events_seen.detach().clone(),
                                            'version': publication.version.detach(), 'routing': result['trace'],
                                            'dependencies': tuple(d for d in dependencies if bool(d.consumed.any()))})
                    state = replace(state, nodes=node_states, ports=state.ports.commit(staged))
                state = replace(state, rounds_seen=state.rounds_seen+1)
            values = [state.ports.latest[n].payload*state.ports.latest[n].valid[:, None] for n in self.cfg.readout]
            latent = self.readout_norm(self.readout_mix(torch.cat(values, -1)))
            logits.append(self.readout(latent))
            state = replace(state, events_seen=state.events_seen+1)
        logits = torch.stack(logits, 1)
        aux = aux/max(activations, 1)
        hat_aux = hat_aux/max(activations+ingress_activations, 1)
        loss = ce = None
        if targets is not None:
            if targets.shape != (b, t) or targets.dtype != torch.long or targets.device != device:
                raise ValueError('Targets must be int64 [B,T] on the input device.')
            if not bool((targets != -100).any()):
                raise ValueError('At least one byte target is required.')
            ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), ignore_index=-100)
            loss = ce + self.cfg.auxiliary_loss_weight*aux + self.cfg.hat_balance_weight*hat_aux
        return {'logits': logits, 'loss': loss, 'ce': ce, 'aux': aux, 'hat_aux': hat_aux,
                'state': state, 'trace': records, 'expert_executions': executed,
                'external_events': b*t, 'source_occurrences': ingress_activations, 'internal_rounds': b*t*k_total}

    def parameter_summary(self):
        c = self.cfg
        physical = 3*(c.shared_per_block + len(c.nodes)*(c.experts_per_block-c.shared_per_block))
        expert_weights = sum(p.numel() for p in self.shared_banks.parameters()) + sum(
            p.numel() for node in self.nodes.values() for block in node.core.blocks for p in block.experts.private.parameters())
        return {'total_unique_parameters': sum(p.numel() for p in self.parameters()),
                'physical_experts': physical, 'expert_matrix_parameters': expert_weights,
                'logical_experts_per_node': 3*c.experts_per_block,
                'expert_calls_per_round_per_external_event': len(c.nodes)*3*c.active_groups*c.experts_per_group,
                'nodes': len(c.nodes), 'regions': len(c.regions), 'byte_vocabulary': 256}

    @torch.no_grad()
    def generate(self, prompt, count=100, temperature=.8, seed=123, **kwargs):
        if count < 0 or temperature <= 0:
            raise ValueError('Nonnegative count and positive temperature required.')
        if isinstance(prompt, str):
            prompt = ByteBatch.text(prompt).to(self.readout.weight.device)
        elif torch.is_tensor(prompt):
            prompt = ByteBatch.static(prompt)
        if not bool((prompt.links == 0).all()):
            raise ValueError('Text generation uses the static byte bank.')
        was_training = self.training
        self.eval()
        try:
            rng = torch.Generator(device=prompt.values.device).manual_seed(seed)
            result = self(prompt, **kwargs)
            pieces = [prompt.values]
            for _ in range(count):
                token = torch.multinomial((result['logits'][:, -1]/temperature).softmax(-1), 1, generator=rng)
                pieces.append(token)
                result = self(token, state=result['state'], **kwargs)
            return torch.cat(pieces, 1), result['state']
        finally:
            self.train(was_training)
