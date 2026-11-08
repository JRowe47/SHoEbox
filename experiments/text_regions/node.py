"""Adapters around the existing SymbolLevel and three-block RecurrentChild."""
import torch
from torch import nn
from ..hat_minkron.hat import HATState
from ..hat_minkron.events import ByteBatch
from .runtime import RuntimeSymbolLevel as SymbolLevel, RuntimeChild as RecurrentChild, enabled
from .ports import SemanticHeader
from .state import NodeState
from .lace_separation import LaceBiasHead


def masked_ingress(hat, events, state, active):
    """Absent bytes are never embedded or used to advance HAT state."""
    b = events.values.shape[0]
    payload = hat.byte_embedding.weight.new_zeros(b, hat.cfg.width)
    aux = payload.new_zeros(())
    rows = active.nonzero().flatten()
    if not rows.numel():
        return payload, state, aux
    if enabled() and rows.numel() == b:
        result = hat(events, state, backend='step')
        return result['payload'][:, 0], result['state'], result['aux']
    small = HATState(len(rows), tuple({k: v[rows] for k, v in layer.items()} for layer in state.levels),
                     {k: v[rows] for k, v in state.clocks.items()})
    subset = ByteBatch(events.values[rows], events.links[rows], events.slots[rows])
    result = hat(subset, small, backend='step')
    layers = []
    for old, new in zip(state.levels, result['state'].levels):
        layer = dict(old)
        for key, tensor in new.items():
            base = old.get(key, tensor.new_zeros(b, *tensor.shape[1:]))
            layer[key] = base.index_copy(0, rows, tensor)
        layers.append(layer)
    clocks = dict(state.clocks)
    for key, tensor in result['state'].clocks.items():
        clocks[key] = clocks.get(key, tensor.new_zeros(b)).index_copy(0, rows, tensor)
    return payload.index_copy(0, rows, result['payload'][:, 0]), HATState(b, tuple(layers), clocks), result['aux']


class LatentAdapter(nn.Module):
    def __init__(self, cfg, safe_zero=True):
        super().__init__()
        self.levels = nn.ModuleList(SymbolLevel(cfg, safe_zero=safe_zero) for _ in range(cfg.hat_levels))
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, values, states):
        x = values[:, None]
        links = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        final, aux = [], x.new_zeros(())
        for level, state in zip(self.levels, states):
            if enabled():
                x, updated, record = level.single(x, state)
                final.append(updated)
            else:
                x, updated, record = level(x, links, {0: state}, backend='step')
                final.append(updated[0])
            aux = aux + record['balance']/len(self.levels)
        return self.norm(x), tuple(final), aux


class RecurrentNode(nn.Module):
    def __init__(self, spec, cfg, shared, index):
        super().__init__()
        self.name, self.cfg = spec.name, cfg
        hat_cfg = cfg.hat_config()
        self.adapter = LatentAdapter(hat_cfg, safe_zero=cfg.zero_safe_norm)
        self.core = RecurrentChild(hat_cfg.child_config(index), hat_cfg, shared, with_decoder=False)
        for block in self.core.blocks:
            block.memory.safe_zero = cfg.zero_safe_norm
        if spec.name in cfg.selective_publication:
            self.publication_gate = nn.Linear(cfg.width, 1)
            nn.init.zeros_(self.publication_gate.weight)
            nn.init.constant_(self.publication_gate.bias, 2.)
        if cfg.hippocampus.enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed + 48031 + index)
                self.anchor_encoder = nn.Linear(cfg.width, cfg.width)
                self.anchor_query = nn.Linear(cfg.hippocampus.address_width, cfg.width)
                self.anchor_reader = nn.Linear(4*cfg.width, cfg.width)

        if cfg.lace_separation.installed:
            self.lace_head = LaceBiasHead(2*cfg.width, 3, cfg.laces, cfg.lace_separation,
                                          seed=cfg.seed+49031+index)

    def encode_anchor(self, latent):
        return self.anchor_encoder(latent).tanh()

    def read_anchor(self, capsule, cue, context=None):
        """Owner-local conditional computation, without advancing native clocks."""
        query = self.anchor_query(cue).tanh()
        context = torch.zeros_like(capsule) if context is None else context
        return self.anchor_reader(torch.cat((capsule, query, capsule*query, context), -1)).tanh()

    def publication_logit(self, latent):
        """A producer's write decision reads only its computed latent."""
        return self.publication_gate(latent).squeeze(-1)

    def initial_state(self, batch, device, dtype):
        return NodeState(tuple(level.memory.initial_state(batch, device, dtype) for level in self.adapter.levels),
                         tuple(block.memory.initial_state(batch, device, dtype) for block in self.core.blocks),
                         torch.zeros(batch, dtype=torch.long, device=device))

    def forward(self, payload, routing_context, state, active, trace=False, lace_bias=None, prepared=None,
                reinstatement=None, cue=None, lace_capture=None):
        cfg = self.cfg
        b, d = payload.shape
        rows = active.nonzero().flatten()
        latent = payload.new_zeros(b, d)
        memberships = payload.new_zeros(b, 3, cfg.groups)
        groups = torch.zeros(b, 3, cfg.active_groups, dtype=torch.long, device=payload.device)
        laces = torch.zeros(b, 3, cfg.active_laces, dtype=torch.long, device=payload.device)
        aux = payload.new_zeros(())
        if not rows.numel():
            return {'latent': latent, 'header': SemanticHeader(self.name, memberships, groups, laces),
                    'state': state, 'aux': aux, 'hat_aux': aux, 'executions': 0, 'trace': None}
        full = enabled() and rows.numel() == b
        if prepared is None:
            x, adapter_state, hat_aux = self.adapter(payload if full else payload[rows],
                state.contextualizer if full else tuple(s[rows] for s in state.contextualizer))
            if reinstatement is None:
                evidence = torch.stack([block.router.pilot(x) for block in self.core.blocks], -2)
        else:
            x, adapter_state, hat_aux, evidence = prepared
        if reinstatement is not None:
            # Recalled capsules have already been interpreted by their owner.
            # Add their composed latent at the expert input, after the recurrent
            # ingress adapter, so its old state cannot overwrite the readback.
            # The zero compose projection preserves the parent warm start. This
            # path can remain weak until query/owner-reader binding is learned;
            # keep it independently ablatable as capacity and training grow.
            recall = reinstatement if full else reinstatement.index_select(0, rows)
            x = x + recall[:, None]
            evidence = torch.stack([block.router.pilot(x) for block in self.core.blocks], -2)
        query_bias = cache = None
        if cfg.lace_separation.enabled:
            if cue is None:
                raise ValueError('Enabled lace bias requires a causal regional cue.')
            z = torch.cat((x[:, 0], cue if full else cue.index_select(0, rows)), -1)
            query_bias, cache = self.lace_head(z, capture=lace_capture is not None)
        prior, lace = self.core.lace(evidence + (routing_context[:, None] if full else routing_context[rows, None]),
            x, utility_bias=lace_bias, query_bias=query_bias, capture_scores=lace_capture is not None)
        if lace_capture is not None:
            collector, key, valid = lace_capture
            collector.record(key, getattr(self, 'lace_head', None), rows, valid[rows], lace, cache)
        execute = self.core.execute_node if enabled() else self.core.execute
        result = execute(x, prior, state.blocks if full else tuple(s[rows] for s in state.blocks), backend='step',
                                   trace=trace, decode=False)
        final = NodeState(adapter_state if full else tuple(old.index_copy(0, rows, new) for old, new in zip(state.contextualizer, adapter_state)),
                          result['state'] if full else tuple(old.index_copy(0, rows, new) for old, new in zip(state.blocks, result['state'])),
                          state.clock + active.long())
        header = SemanticHeader(self.name,
            result['semantic'][:, 0] if full else memberships.index_copy(0, rows, result['semantic'][:, 0]),
            result['group_ids'][:, 0] if full else groups.index_copy(0, rows, result['group_ids'][:, 0]),
            result['feedback_lace_ids'][:, 0] if full else laces.index_copy(0, rows, result['feedback_lace_ids'][:, 0]))
        record = None
        if trace:
            record = {'rows': rows.detach(), 'groups': result['group_ids'][:, 0].detach(),
                      'laces': lace['lace_ids'][:, 0].detach(),
                      'observed_laces': result['feedback_lace_ids'][:, 0].detach(),
                      'experts': result['expert_ids'][:, 0].detach()}
        return {'latent': result['latent'][:, 0] if full else latent.index_copy(0, rows, result['latent'][:, 0]), 'header': header,
                'state': final, 'aux': result['aux'] + .1*lace['balance'], 'hat_aux': hat_aux,
                'executions': len(rows)*3*cfg.active_groups*cfg.experts_per_group, 'trace': record}
