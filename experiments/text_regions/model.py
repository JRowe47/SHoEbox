"""Event-major reference engine: ingest once, run full pipelines K times, decode."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from ..hat_minkron.events import ByteBatch
from .runtime import RuntimeHAT as HAT, forward_scope, generation_scope, next_event
from ..hat_minkron.semantic_core import ExpertBank
from .config import GraphConfig
from .graph import GraphConnections
from .node import RecurrentNode, masked_ingress
from .ports import Publication, SemanticHeader, TensorPorts, publish
from .state import GraphState
from .stage_runtime import prepare_stage
from .graph_publication import GraphPublicationPolicy, accumulate_demand, finish_demand


class RegionLanguageModel(nn.Module):
    def __init__(self, cfg=None, execution_backend='optimized'):
        super().__init__()
        self.set_execution_backend(execution_backend)
        self.cfg = cfg or GraphConfig()
        self.cfg.validate()
        self._large_expert_banks = max(self.cfg.shared_per_block,
            self.cfg.experts_per_block-self.cfg.shared_per_block)*self.cfg.width*self.cfg.expert_hidden >= 65536
        self.batch_independent = self._large_expert_banks
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
        self.connections.initialize_recall()
        self.publisher = None
        if self.cfg.task_publication:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.cfg.seed + 17389)
                self.publisher = GraphPublicationPolicy(self.cfg)
        self.hippocampus = None
        self.memory_interface = None
        if self.cfg.hippocampus.enabled:
            from .hippocampus import Hippocampus
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.cfg.seed + 80317)
                self.hippocampus = Hippocampus(self.cfg)
            self.memory_requesters = {region: owners[-1] for region, owners in self.cfg.regions.items()}
            # Topology-derived ingress routes, computed once. Never search the
            # publication candidates to decide the identity of a memory cue.
            ancestry = {n.name: ({n.view} if n.view is not None else set()) for n in self.cfg.nodes}
            for stage in self.stages:
                for name in stage:
                    for edge in self.cfg.edges:
                        if edge.destination == name and edge.kind == 'forward':
                            ancestry[name].update(ancestry[edge.source])
            self.memory_views = {region: tuple(sorted(set().union(*(ancestry[n] for n in owners))))
                                 for region, owners in self.cfg.regions.items()}

        if self.cfg.memory_context.enabled:
            from .memory_context import MemoryContextInterface
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.cfg.seed+88031)
                self.memory_interface = MemoryContextInterface(self.cfg)

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
                          torch.zeros(batch, dtype=torch.long, device=device),
                          recall_feedback={name: torch.zeros(batch, c.width, device=device, dtype=dtype)
                                           for name in self.connections.recall},
                          publication_demand={name: torch.zeros(batch, 2, device=device, dtype=dtype)
                                              for name in c.task_publication},
                          hippocampus=self.hippocampus.initial_state(batch, device, dtype)
                              if self.hippocampus is not None else None,
                          memory_context={name:self.memory_interface.empty(batch,device,dtype) for name in self.nodes}
                              if self.memory_interface is not None else {})

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
        if self.hippocampus is not None and state.hippocampus is None:
            memory = self.hippocampus.initial_state(b, first.values.device, self.readout.weight.dtype)
            memory = replace(memory, rows=tuple(replace(row, namespace=int(state.epochs[i]),
                round=int(state.rounds_seen[i])) for i, row in enumerate(memory.rows)))
            state = replace(state, hippocampus=memory)
        if state.events_seen.shape != (b,) or state.events_seen.device != first.values.device:
            raise ValueError('State batch/device mismatch.')
        if set(state.nodes) != set(self.nodes):
            raise ValueError('State topology mismatch.')
        if set(state.recall_feedback) - set(self.connections.recall):
            raise ValueError('Recall feedback topology mismatch.')
        for value in state.recall_feedback.values():
            if value.shape != (b, self.cfg.width) or value.device != first.values.device:
                raise ValueError('Recall feedback batch/device mismatch.')
        if set(state.publication_demand) != set(self.cfg.task_publication):
            raise ValueError('Publication demand topology mismatch.')
        for value in state.publication_demand.values():
            if value.shape != (b, 2) or value.device != first.values.device:
                raise ValueError('Publication demand batch/device mismatch.')
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
                return_publication_activity=False, publication_overrides=None,
                return_recall_activity=False, recall_control='intact', return_write_activity=False):
        if recall_control not in ('intact', 'pooled', 'no_feedback', 'blocked'):
            raise ValueError('Unknown regional recall control.')
        publication_overrides = publication_overrides or {}
        publishers = self.cfg.selective_publication + self.cfg.task_publication
        if set(publication_overrides)-set(publishers) or any(
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
        write_activity, write_resets = [], []
        recall_activity = {name: [] for name in self.connections.recall} if return_recall_activity else {}
        node_activity = {name: {'payload': [], 'valid': []} for name in return_node_activity}
        publication_activity = {name: {'candidate': [], 'logit': [], 'written': []}
                                for name in publishers} if return_publication_activity else {}
        round_activity = {name: {'payload': [], 'valid': []} for name in self.cfg.readout} if return_round_activity else {}
        view_activity = {view: {'payload': [], 'available': []} for view in events} if return_view_activity else {}
        edge_names = tuple(self.connections.edges)
        edge_indices = {key: i for i, key in enumerate(edge_names)}
        usage = torch.zeros(b, t, len(edge_names), dtype=torch.long, device=device) if return_edge_activity else None
        aux = self.readout.weight.new_zeros(())
        hat_aux = aux
        activations = ingress_activations = executed = 0
        memory_work = []
        for event_index in range(t):
            next_event()
            event_recalls = {name: [] for name in recall_activity}
            event_publications = {name: {key: [] for key in activity}
                                  for name, activity in publication_activity.items()}
            event_activity = {name: {'payload': [], 'valid': []} for name in round_activity}
            rows = torch.zeros_like(all_active)
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
                memory_residuals, memory_receipts = {}, ()
                if self.hippocampus is not None:
                    source_values, source_valid = [], []
                    for region in self.hippocampus.regions:
                        views = self.memory_views[region] if controlled else ('text',)
                        pubs = [state.ports.latest['@'+v] for v in views]
                        mask = torch.stack([p.valid for p in pubs], 1)
                        source_values.append((torch.stack([p.payload for p in pubs], 1)*mask[..., None]).sum(1)
                                             /mask.sum(1, keepdim=True).clamp_min(1))
                        source_valid.append(mask.any(1))
                    grounded = torch.stack(source_valid, 1)
                    if k == 0 or self.cfg.hippocampus.cue_clock == 'round':
                        memory, memory_keys = self.hippocampus.advance_cues(
                            state.hippocampus, torch.stack(source_values, 1), grounded)
                    else:
                        # The cue represents the external event, not the number
                        # of internal computation rounds. Reuse its live tensor;
                        # memory commits and conditional reads still run each round.
                        memory = state.hippocampus
                    owner_active = {n: node_masks[n][:, event_index] if n in node_masks else all_active
                                    for n in self.nodes}
                    state = replace(state, hippocampus=memory)
                    if self.memory_interface is not None:
                        state, memory_receipts, work = self.memory_interface.begin_round(
                            self,state,torch.stack(source_values,1),owner_active)
                    else:
                        query_active = torch.stack([owner_active[self.memory_requesters[r]]
                                                    for r in self.hippocampus.regions], 1)
                        residuals, memory_receipts, work = self.hippocampus.query(
                            memory, memory_keys, self.nodes, self.memory_requesters, query_active,
                            {n: p.payload*p.valid[:, None] for n,p in state.ports.latest.items() if n in self.nodes},
                            owner_active)
                        memory_residuals = {self.memory_requesters[r]: residuals[:, i]
                                            for i,r in enumerate(self.hippocampus.regions)}
                    memory_work.extend(work)
                    memory_latents, memory_active = {}, {}
                delayed = state.ports.snapshot() if feedback else None
                demands = {name: [] for name in self.cfg.task_publication}
                memory_demands = {}
                if (self.hippocampus is not None and self.cfg.hippocampus.feedback
                        and self.cfg.hippocampus.control != 'no_feedback'):
                    # Attribute usage only to actual, version-validated owners.
                    # This is a next-round policy feature, not a second gradient
                    # receipt. Its zero-initialized gain preserves parent logits.
                    for bi, receipts in enumerate(memory_receipts):
                        for receipt in receipts:
                            for owner in dict.fromkeys(handle.node for _,handle in receipt.contributors):
                                if owner in demands:
                                    value = receipt.support*self.hippocampus.publication_feedback_gain.tanh()
                                    values = value.new_zeros(b,2).index_copy(0,
                                        torch.tensor([bi],device=device),value.expand(1,2))
                                    memory_demands[owner] = memory_demands.get(owner,0.)+values
                round_writes = {}
                for stage in self.stages:
                    current, staged, node_states = state.ports.snapshot(), {}, dict(state.nodes)
                    requests = {}
                    recall_feedback = dict(state.recall_feedback)
                    for name in reversed(stage) if reverse_independent else stage:
                        spec = self.specs[name]
                        active = node_masks[name][:, event_index] if name in node_masks else all_active
                        ingress = None
                        if spec.view is not None:
                            ingress = current['@'+spec.view if controlled else '@text']
                            ingress = replace(ingress, valid=ingress.valid & available[spec.view])
                        recall_record = {} if name in self.connections.recall and (
                            return_recall_activity or self.publisher is not None) else None
                        payload, context, dependencies = self.connections.gather(
                            spec, current, delayed, ingress, active, event_index, edge_masks, substitutions, edge_bias,
                            state.recall_feedback.get(name), recall_control, recall_record)
                        if name in memory_residuals and self.cfg.hippocampus.reinstatement == 'ingress':
                            payload = payload + memory_residuals[name]*active[:, None]
                        if name in event_recalls:
                            event_recalls[name].append(recall_record)
                        if self.publisher is not None:
                            accumulate_demand(demands, self.connections, name, recall_record,
                                              dependencies, active, recall_control == 'blocked')
                        if return_edge_activity:
                            for dependency in dependencies:
                                if dependency.edge in edge_indices:
                                    usage[:, event_index, edge_indices[dependency.edge]] += dependency.consumed.long()
                        if self.memory_interface is not None:
                            context_slots = self.memory_interface.gather(name,state,current,delayed,active,
                                event_index,edge_masks,substitutions)
                            state = replace(state,memory_context={**state.memory_context,name:context_slots})
                        requests[name] = (payload, context, state.nodes[name], active, dependencies)
                    prepared = prepare_stage(self, requests)
                    for name, (payload, context, node_state, active, dependencies) in requests.items():
                        spec = self.specs[name]
                        memory = (memory_residuals.get(name)
                                  if self.cfg.hippocampus.reinstatement == 'expert' else None)
                        cue = None
                        if self.cfg.lace_separation.enabled:
                            ri = self.hippocampus.regions.index(spec.region)
                            cue = state.hippocampus.cues[:, ri]
                            cue_rows = getattr(self, '_lace_cue_rows', None)
                            if cue_rows is not None:
                                cue = cue.index_select(0, cue_rows)
                        collector = getattr(self, '_lace_capture', None)
                        capture = None
                        if collector is not None and torch.is_grad_enabled():
                            valid = state.events_seen.eq(collector.event) & active
                            if bool(valid.any()):
                                capture = (collector, (name, k), valid)
                        memory_options = {}
                        if self.memory_interface is not None:
                            memory_options = dict(memory_context=self.memory_interface.interpret(
                                state.memory_context[name],state,self,active),
                                memory_interface=self.memory_interface,event=state.events_seen)
                        result = self.nodes[name](payload, context, node_state, active, trace,
                            lace_bias.get(name), prepared.get(name), reinstatement=memory,
                            cue=cue, lace_capture=capture, **memory_options)
                        if self.hippocampus is not None:
                            memory_latents[name] = result['latent']
                            memory_active[name] = active
                        node_states[name] = result['state']
                        if name in self.connections.recall:
                            previous = state.recall_feedback.get(name, torch.zeros_like(result['latent']))
                            recall_feedback[name] = torch.where(active[:, None], result['latent'], previous)
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
                        if name in self.cfg.task_publication:
                            result['previous_state'] = node_state
                            publication, node_states[name], admission = self.publisher(
                                name, current[name], result, active, state.epochs, state.events_seen,
                                torch.full_like(state.events_seen, k), dependencies,
                                state.publication_demand[name], self.cfg, publication_overrides.get(name))
                            writing = admission['write_actions'].bool()
                            if return_write_activity:
                                round_writes[name] = admission
                            if return_publication_activity:
                                event_publications[name]['candidate'].append(result['latent'])
                                event_publications[name]['logit'].append(admission['logit'])
                                event_publications[name]['written'].append(writing)
                        else:
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
                    state = replace(state, nodes=node_states, ports=state.ports.commit(staged),
                                    recall_feedback=recall_feedback)
                if self.hippocampus is not None:
                    if self.memory_interface is not None and self.cfg.memory_context.event_links:
                        from .memory_events import event_offers
                        offers = event_offers(self,state,memory_keys,torch.stack(source_values,1),
                                              memory_latents,memory_active,grounded)
                    else:
                        offers = self.hippocampus.make_offers(state.hippocampus, memory_keys, memory_latents,
                            self.nodes, memory_active, {n:state.ports.latest[n].version for n in self.nodes}, grounded)
                    state = replace(state, hippocampus=self.hippocampus.commit(state.hippocampus, offers, memory_receipts))
                next_demand = finish_demand(demands, state.publication_demand)
                for owner, demand in memory_demands.items():
                    next_demand[owner] = next_demand[owner]+demand
                state = replace(state, rounds_seen=state.rounds_seen+1,
                                publication_demand=next_demand)
                if return_write_activity and round_writes:
                    write_activity.append({key: torch.stack([round_writes[n][key]
                        for n in self.cfg.task_publication], 1) for key in next(iter(round_writes.values()))})
                    write_resets.append(rows if k == 0 else torch.zeros_like(rows))
                for name, activity in event_activity.items():
                    publication = state.ports.latest[name]
                    activity['payload'].append(publication.payload)
                    activity['valid'].append(publication.valid)
            for name, items in event_recalls.items():
                recall_activity[name].append({key: torch.stack([r[key] for r in items], 1) for key in items[0]})
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
            if self.memory_interface is not None:
                latent = self.memory_interface.compose(latent,state,self)
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
        recall_activity = {name: {key: torch.stack([r[key] for r in items], 1)
                                  for key in items[0]} for name, items in recall_activity.items()}
        write_activity = ({key: torch.stack([a[key] for a in write_activity], 1)
                           for key in write_activity[0]} | {'reset': torch.stack(write_resets, 1)}
                          ) if write_activity else {}
        return {'logits': logits, 'loss': loss, 'ce': ce, 'aux': aux, 'hat_aux': hat_aux,
                'state': state, 'trace': records, 'expert_executions': executed,
                'view_activity': view_activity, 'edge_activity': usage, 'edge_names': edge_names,
                'round_activity': round_activity, 'node_activity': node_activity,
                'publication_activity': publication_activity,
                'recall_activity': recall_activity,
                'write_activity': write_activity,
                'hippocampus_work': memory_work,
                'recall_edges': {n: tuple(e.key for e in self.connections.incoming[n]) for n in recall_activity},
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
    @generation_scope
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
