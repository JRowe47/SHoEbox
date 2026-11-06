"""Event-major reference engine: ingest once, run full pipelines K times, decode."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from ..hat_minkron.events import ByteBatch
from .runtime import RuntimeHAT as HAT, forward_scope
from ..hat_minkron.semantic_core import ExpertBank
from .config import GraphConfig
from .graph import GraphConnections
from .node import RecurrentNode, masked_ingress
from .ports import Publication, SemanticHeader, TensorPorts, publish
from .state import GraphState


class RegionLanguageModel(nn.Module):
    def __init__(self, cfg=None, execution_backend='optimized'):
        super().__init__()
        self.set_execution_backend(execution_backend)
        self.cfg = cfg or GraphConfig()
        self.cfg.validate()
        c = self.cfg.hat_config()
        self.hat = HAT(c, safe_zero=self.cfg.zero_safe_norm)
        self.shared_banks = nn.ModuleList(ExpertBank(c.child_config(), self.cfg.shared_per_block) for _ in range(3))
        self.independent_hats = nn.ModuleDict({name: HAT(c, safe_zero=self.cfg.zero_safe_norm)
                                             for name in self.cfg.independent_sources})
        self.independent_banks = nn.ModuleDict({name: nn.ModuleList(
            ExpertBank(c.child_config(), self.cfg.shared_per_block) for _ in range(3))
            for name in self.cfg.independent_sources})
        self.view_owners = {n.view: n.name for n in self.cfg.nodes if n.name in self.cfg.independent_sources}
        self.nodes = nn.ModuleDict({n.name: RecurrentNode(n, self.cfg,
                                   self.independent_banks[n.name] if n.name in self.independent_banks else self.shared_banks, i)
                                   for i, n in enumerate(self.cfg.nodes)})
        self.connections = GraphConnections(self.cfg)
        if self.cfg.readout_mode == 'mixed':
            self.readout_mix = nn.Linear(len(self.cfg.readout)*self.cfg.width, self.cfg.width, bias=False)
            self.readout_norm = nn.LayerNorm(self.cfg.width)
            self.readout = nn.Linear(self.cfg.width, 256, bias=False)
        else:
            # One common affine byte decoder over complete P/Q publications.
            # Regional computation, rather than a nonlinear output adapter,
            # must perform cross-region composition.
            self.readout_mix = nn.Identity()
            self.readout_norm = nn.Identity()
            self.readout = nn.Linear(len(self.cfg.readout)*self.cfg.width, 256)
            nn.init.zeros_(self.readout.bias)
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
        return GraphState({view: self.ingress_hat(view).initial_state(batch) for view in ('text', *c.views)},
                          {name: node.initial_state(batch, device, dtype) for name, node in self.nodes.items()},
                          TensorPorts({name: empty(name) for name in names}),
                          torch.zeros(batch, dtype=torch.long, device=device),
                          torch.zeros(batch, dtype=torch.long, device=device),
                          torch.zeros(batch, dtype=torch.long, device=device))

    def ingress_hat(self, view):
        owner = self.view_owners.get(view)
        return self.independent_hats[owner] if owner is not None else self.hat

    def _prepare(self, inputs, state, availability, node_masks, edge_masks, substitutions, reset):
        controlled = isinstance(inputs, dict)
        if self.cfg.independent_sources and not controlled:
            raise ValueError('Independent source encoders require explicit source views.')
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

    def set_execution_backend(self, backend):
        """Select execution only; weights, topology and checkpoint schema are unchanged."""
        if backend not in ('reference', 'optimized'):
            raise ValueError('Execution backend must be reference or optimized.')
        self.execution_backend = backend
        return self

    @forward_scope
    def forward(self, inputs, targets=None, state=None, reset=None, rounds=None, feedback=None,
                availability=None, node_masks=None, edge_masks=None, substitutions=None,
                trace=False, reverse_independent=False, return_view_activity=False,
                return_edge_activity=False, edge_bias=None, lace_bias=None,
                return_round_activity=False, return_node_activity=(),
                return_publication_activity=False, publication_overrides=None):
        publication_overrides = publication_overrides or {}
        if set(publication_overrides)-set(self.cfg.selective_publication) or any(
                value not in ('always', 'hold') for value in publication_overrides.values()):
            raise ValueError('Publication controls require configured nodes and always/hold policies.')
        if len(set(return_node_activity)) != len(return_node_activity) or set(return_node_activity)-set(self.nodes):
            raise ValueError('Activity capture requires distinct known node names.')
        availability, node_masks = availability or {}, node_masks or {}
        edge_masks, substitutions = edge_masks or {}, substitutions or {}
        events, state, b, t, controlled = self._prepare(inputs, state, availability, node_masks,
                                                       edge_masks, substitutions, reset)
        edge_bias = edge_bias or {}
        if set(edge_bias)-set(self.connections.edges):
            raise ValueError('Unknown edge in utility bias.')
        for bias in edge_bias.values():
            value = torch.as_tensor(bias)
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise ValueError('Edge utility biases must be finite scalars.')
        lace_bias = lace_bias or {}
        if set(lace_bias)-set(self.nodes):
            raise ValueError('Unknown node in lace utility bias.')
        for bias in lace_bias.values():
            if bias.shape != (3, self.cfg.laces) or not bool(torch.isfinite(bias).all()):
                raise ValueError('Lace utility biases must be finite [3,laces] tensors.')
        k_total = self.cfg.rounds if rounds is None else rounds
        if not isinstance(k_total, int) or k_total < 1:
            raise ValueError('Rounds must be a positive integer.')
        feedback = self.cfg.recurrent_feedback if feedback is None else feedback
        device = state.events_seen.device
        all_active = torch.ones(b, dtype=torch.bool, device=device)
        logits, records = [], []
        node_activity = {name: {'payload': [], 'valid': []} for name in return_node_activity}
        publication_activity = {name: {'candidate': [], 'logit': [], 'written': []}
                                for name in self.cfg.selective_publication} if return_publication_activity else {}
        round_activity = {name: {'payload': [], 'valid': []} for name in self.cfg.readout} if return_round_activity else {}
        view_activity = {view: {'payload': [], 'available': []} for view in events} if return_view_activity else {}
        edge_names = tuple(self.connections.edges)
        edge_indices = {key: i for i, key in enumerate(edge_names)}
        usage = torch.zeros(b, t, len(edge_names), dtype=torch.long, device=device) if return_edge_activity else None
        aux = self.readout.weight.new_zeros(())
        hat_aux = aux
        activations = ingress_activations = executed = 0
        for event_index in range(t):
            event_publications = {name: {key: [] for key in activity}
                                  for name, activity in publication_activity.items()}
            event_activity = {name: {'payload': [], 'valid': []} for name in round_activity}
            if reset is not None and (reset.ndim == 2 or event_index == 0):
                rows = reset[:, event_index] if reset.ndim == 2 else reset
                if bool(rows.any()):
                    state = state.reset(rows)
            available = {v: availability[v][:, event_index] if v in availability else all_active for v in self.cfg.views}
            ingress_states, staged_ingress = dict(state.ingress), {}
            for view, sequence in events.items():
                active = available[view] if controlled else torch.stack(tuple(available.values())).any(0)
                event = sequence.slice(event_index, event_index+1)
                payload, hstate, haux = masked_ingress(self.ingress_hat(view), event, state.ingress[view], active)
                if return_view_activity:
                    view_activity[view]['payload'].append(payload)
                    view_activity[view]['available'].append(active.clone())
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
                            spec, current, delayed, ingress, active, event_index, edge_masks, substitutions, edge_bias)
                        if return_edge_activity:
                            for dependency in dependencies:
                                if dependency.edge in edge_indices:
                                    usage[:, event_index, edge_indices[dependency.edge]] += dependency.consumed.long()
                        result = self.nodes[name](payload, context, state.nodes[name], active, trace, lace_bias.get(name))
                        node_states[name] = result['state']
                        writing = active
                        if name in self.cfg.selective_publication:
                            logit = self.nodes[name].publication_logit(result['latent'])
                            decision = logit.ge(0)
                            if name in publication_overrides:
                                decision = torch.full_like(active, publication_overrides[name] == 'always')
                            writing = active & decision
                            if return_publication_activity:
                                event_publications[name]['candidate'].append(result['latent'])
                                event_publications[name]['logit'].append(logit)
                                event_publications[name]['written'].append(writing)
                        publication = publish(current[name], result['latent'], result['header'], writing,
                                              state.epochs, state.events_seen, torch.full_like(state.events_seen, k),
                                              dependencies)
                        if self.cfg.evidence_gated_ports:
                            evidence = torch.zeros_like(active)
                            for dependency in dependencies:
                                evidence = evidence | dependency.consumed
                            publication = replace(publication, valid=current[name].valid | (evidence & writing))
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
                                            'valid': publication.valid.detach(),
                                            'dependencies': tuple(d for d in dependencies if bool(d.consumed.any()))})
                    state = replace(state, nodes=node_states, ports=state.ports.commit(staged))
                state = replace(state, rounds_seen=state.rounds_seen+1)
                for name, activity in event_activity.items():
                    publication = state.ports.latest[name]
                    activity['payload'].append(publication.payload)
                    activity['valid'].append(publication.valid)
            for name, activity in event_activity.items():
                for key, values in activity.items():
                    round_activity[name][key].append(torch.stack(values, 1))
            for name, activity in event_publications.items():
                for key, values in activity.items():
                    publication_activity[name][key].append(torch.stack(values, 1))
            for name, activity in node_activity.items():
                publication = state.ports.latest[name]
                activity['payload'].append(publication.payload)
                activity['valid'].append(publication.valid)
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
        if return_view_activity:
            view_activity = {name: {key: torch.stack(parts, 1) for key, parts in values.items()}
                             for name, values in view_activity.items()}
            for activity in view_activity.values():
                activity['provenance'] = 'independent_ingress'
        round_activity = {name: {key: torch.stack(parts, 1) for key, parts in values.items()}
                          for name, values in round_activity.items()}
        node_activity = {name: {key: torch.stack(parts, 1) for key, parts in values.items()}
                         for name, values in node_activity.items()}
        publication_activity = {name: {key: torch.stack(parts, 1) for key, parts in values.items()}
                                for name, values in publication_activity.items()}
        return {'logits': logits, 'loss': loss, 'ce': ce, 'aux': aux, 'hat_aux': hat_aux,
                'state': state, 'trace': records, 'expert_executions': executed,
                'view_activity': view_activity, 'edge_activity': usage, 'edge_names': edge_names,
                'round_activity': round_activity, 'node_activity': node_activity,
                'publication_activity': publication_activity,
                'external_events': b*t, 'source_occurrences': ingress_activations, 'internal_rounds': b*t*k_total}

    def parameter_summary(self):
        c = self.cfg
        physical = 3*((1+len(c.independent_sources))*c.shared_per_block + len(c.nodes)*(c.experts_per_block-c.shared_per_block))
        expert_weights = sum(p.numel() for p in self.shared_banks.parameters()) + sum(
            p.numel() for p in self.independent_banks.parameters()) + sum(
            p.numel() for node in self.nodes.values() for block in node.core.blocks for p in block.experts.private.parameters())
        return {'total_unique_parameters': sum(p.numel() for p in self.parameters()),
                'physical_experts': physical, 'expert_matrix_parameters': expert_weights,
                'logical_experts_per_node': 3*c.experts_per_block,
                'expert_calls_per_round_per_external_event': len(c.nodes)*3*c.active_groups*c.experts_per_group,
                'nodes': len(c.nodes), 'regions': len(c.regions), 'byte_vocabulary': 256}

    @torch.no_grad()
    def generate(self, prompt, count=100, temperature=.8, seed=123, state=None, reset=None, **kwargs):
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
            result = self(prompt, state=state, reset=reset, **kwargs)
            pieces = [prompt.values]
            for _ in range(count):
                token = torch.multinomial((result['logits'][:, -1]/temperature).softmax(-1), 1, generator=rng)
                pieces.append(token)
                result = self(token, state=result['state'], **kwargs)
            return torch.cat(pieces, 1), result['state']
        finally:
            self.train(was_training)
