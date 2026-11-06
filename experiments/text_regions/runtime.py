"""Execution optimizations for the regional graph, with unchanged parameters.

The preserved SHoE experiments remain the reference implementation. Derived
weight tensors live for ONE graph forward only, so updates, device changes,
independent backwards and checkpoint loads cannot leave stale cached graphs.
"""
from contextvars import ContextVar
from functools import wraps
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..hat_minkron.hat import HAT, SymbolLevel
from ..hat_minkron.kron import MinKron
from ..hat_minkron.model import RecurrentChild, RecurrentBlock, StreamingRouter
from ..hat_minkron.semantic_core import LaceController, normalize, switch_balance
from ..hat_minkron.hierarchy_reference import SplitExpertBank

_cache = ContextVar('shoebox_forward_cache', default=None)


def enabled():
    return _cache.get() is not None


def forward_scope(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        token = _cache.set({} if self.execution_backend == 'optimized' else None)
        try:
            return function(self, *args, **kwargs)
        finally:
            _cache.reset(token)
    return wrapped


def cached(module, name, compute):
    cache = _cache.get()
    if cache is None:
        return compute()
    grad = torch.is_grad_enabled()
    key = (module, name, grad)
    if key not in cache:
        # Observational routing can reuse values without extending autograd.
        training_key = (module, name, True)
        if not grad and training_key in cache:
            value = cache[training_key]
            cache[key] = tuple(v.detach() for v in value) if isinstance(value, tuple) else value.detach()
        else:
            cache[key] = compute()
    return cache[key]


def direction(x, safe_zero):
    # F.normalize and the zero-direction guard used to calculate this norm twice.
    norm = x.norm(dim=-1, keepdim=True)
    value = x / norm.clamp_min(1e-12).expand_as(x)
    return torch.where(norm > 1e-12, value, torch.zeros_like(value)) if safe_zero else value


class RuntimeMinKron(MinKron):
    def forward(self, x, state=None, backend='chunk', reset_each=False):
        if not enabled() or x.shape[1] != 1:
            return super().forward(x, state, backend, reset_each)
        if backend not in ('step', 'chunk'):
            raise ValueError('backend must be chunk or step')
        b, t, _ = x.shape
        if state is not None and state.shape != (b, self.H, self.K, self.V):
            raise ValueError('Wrong minKron state shape.')
        qkv, erase, write, forget, gate = self.projection(x).split(self.splits, -1)
        q, k, v = F.silu(qkv).split([self.H*self.K, self.H*self.K, self.H*self.V], -1)
        q, k, v = [direction(a.reshape(b, self.H, d), self.safe_zero)
                   for a, d in ((q, self.K), (k, self.K), (v, self.V))]
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        f = self.forget_projection(forget).to(acc) + self.dt_bias.to(acc)
        decay = cached(self, ('decay', acc), lambda: self.A_log.to(acc).exp())
        g = -decay[None, :, None]*F.softplus(f).reshape(b, self.H, self.K)
        # The event-major engine always supplies one timestep. Avoid the
        # sequence transposes and repeated slice/select backward nodes.
        q, k, v = (a.to(acc) for a in (q, k, v))
        previous = q.new_zeros(b, self.H, self.K, self.V) if state is None or reset_each else state.to(acc)
        decayed = previous*g.exp().unsqueeze(-1)
        norm = v.square().sum(-1)
        inv = norm.clamp_min(1e-18).reciprocal()
        score = ((k.unsqueeze(-1)*decayed)*v.unsqueeze(-2)).sum((-2, -1))
        u = write.to(acc).sigmoid().squeeze(1)*norm*inv-erase.sigmoid().to(acc).squeeze(1)*inv*score
        final = decayed+u[..., None, None]*k.unsqueeze(-1)*v.unsqueeze(-2)
        out = (q.unsqueeze(-1)*final).sum(-2)*self.K**-.5
        z = out.to(x.dtype).unsqueeze(1).to(acc)
        z = z*torch.rsqrt(z.square().mean(-1, keepdim=True)+1e-5)*self.norm_weight.to(acc)
        z = z*(gate+self.output_gate_bias).reshape(b, t, self.H, self.V).sigmoid()
        return self.output(z.reshape(b, t, self.H*self.V).to(x.dtype)), final


class RuntimeSymbolLevel(SymbolLevel):
    def __init__(self, cfg, safe_zero=False):
        # Match reference construction order, names and random draws exactly.
        nn.Module.__init__(self)
        self.cfg, self.safe_zero = cfg, safe_zero
        self.norm = nn.LayerNorm(cfg.width)
        self.memory = RuntimeMinKron(cfg.width, cfg.kron_heads, cfg.kron_key,
                                    cfg.kron_value, cfg.chunk_size, safe_zero=safe_zero)
        self.query = nn.Linear(cfg.width, cfg.address_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(cfg.hat_symbols, cfg.address_dim))
        self.values = nn.Parameter(torch.randn(cfg.hat_symbols, cfg.width)*.02)

    def single(self, x, state, backend='step', memory_enabled=True):
        context, final = self.memory(self.norm(x), state, backend, not memory_enabled)
        h = x+context
        q = direction(self.query(self.norm(h)), self.safe_zero)
        keys = cached(self, 'keys', lambda: F.normalize(self.keys, dim=-1))
        scores = 4*F.linear(q, keys)
        probabilities = scores.softmax(-1)
        ids = scores.topk(self.cfg.hat_active_symbols, -1).indices
        weights = probabilities.gather(-1, ids)*self.cfg.hat_symbols/self.cfg.hat_active_symbols
        symbol = (self.values[ids]*weights[..., None]).sum(-2)/math.sqrt(self.cfg.hat_active_symbols)
        return h+symbol, final, dict(ids=ids, weights=weights, balance=switch_balance(probabilities, ids))

    def forward(self, x, links, states, backend='chunk', memory_enabled=True):
        if enabled() and x.shape[1] == 1 and bool((links == links[0, 0]).all()):
            link = int(links[0, 0])
            previous = states.get(link)
            if previous is None:
                previous = self.memory.initial_state(x.shape[0], x.device, x.dtype)
            value, final, record = self.single(x, previous, backend, memory_enabled)
            return value, {**states, link: final}, record
        return super().forward(x, links, states, backend, memory_enabled)


class RuntimeHAT(HAT):
    def __init__(self, cfg, safe_zero=False):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.byte_embedding = nn.Embedding(256, cfg.width)
        self.link_embedding = nn.Embedding(cfg.max_links, cfg.width)
        self.slot_embedding = nn.Embedding(cfg.max_slots, cfg.width)
        self.binding = nn.Linear(3*cfg.width, cfg.width, bias=False)
        self.levels = nn.ModuleList(RuntimeSymbolLevel(cfg, safe_zero) for _ in range(cfg.hat_levels))
        self.final_norm = nn.LayerNorm(cfg.width)
        for m in (self.byte_embedding, self.link_embedding, self.slot_embedding, self.binding):
            nn.init.normal_(m.weight, std=.02)


class RuntimeLaceController(LaceController):
    def forward(self, evidence, context, enabled=True, permutation=None, utility_bias=None,
                observe_only=False):
        if _cache.get() is None or not enabled or permutation is not None:
            return super().forward(evidence, context, enabled, permutation, utility_bias)
        c = self.c
        templates = cached(self, 'templates', lambda: self.template_logits.softmax(-1))
        base_shape = evidence.shape
        e = evidence.reshape(-1, c.layers, c.groups)
        ctx = self.context_query(context).reshape(-1, c.laces)
        if utility_bias is not None and (utility_bias.shape != (c.layers, c.laces)
                                        or not bool(torch.isfinite(utility_bias).all())):
            raise ValueError('Lace utility bias must be finite [layers,laces].')
        layer_ids = cached(self, 'layer_ids', lambda: torch.arange(c.layers, device=e.device)[None, :, None])
        current = e
        for iteration in range(c.refinement_rounds):
            agreement = torch.einsum('nlg,mlg->nml', current.softmax(-1), templates)
            support = agreement.clamp_min(1e-8).log()
            external = (ctx[:, :, None]+support.sum(-1, keepdim=True)-support).transpose(1, 2)
            if utility_bias is not None:
                external = external+utility_bias
            scores, ids = external.topk(c.active_laces, -1)
            if observe_only and iteration == c.refinement_rounds-1:
                return None, {'lace_ids': ids.reshape(*base_shape[:2], c.layers, c.active_laces)}
            weight = scores.softmax(-1)
            prior = (templates[ids, layer_ids]*weight[..., None]).sum(-2)
            if iteration < c.refinement_rounds-1:
                message = prior.clamp_min(1e-8).log()
                current = e+c.lace_gain*(message-message.mean(-1, keepdim=True))
        return prior.reshape(base_shape), dict(
            lace_ids=ids.reshape(*base_shape[:2], c.layers, c.active_laces),
            lace_weights=weight.reshape(*base_shape[:2], c.layers, c.active_laces),
            balance=switch_balance(external.softmax(-1), ids))


class RuntimeRouter(StreamingRouter):
    def pilot(self, x):
        if not enabled():
            return super().pilot(x)
        prototypes = cached(self, 'prototypes', lambda: normalize(self.prototypes))
        return 4*F.linear(normalize(self.query(x)), prototypes)

    def forward(self, x, prior):
        if not enabled():
            return super().forward(x, prior)
        c = self.c
        prototypes = cached(self, 'prototypes', lambda: normalize(self.prototypes))
        query = normalize(self.query(x)+c.lace_gain*self.message(prior@prototypes))
        scores = 4*F.linear(query, prototypes)
        message = prior.clamp_min(1e-8).log()
        scores = scores+c.lace_gain*(message-message.mean(-1, keepdim=True))
        prob = scores.softmax(-1)
        groups = scores.topk(c.active_groups, -1).indices
        candidates = self.membership[groups]
        valid = candidates >= 0
        if x.shape[0]*x.shape[1]*c.active_groups >= c.groups:
            # These are small routing addresses, not expert MLP weights. When
            # the batch already requests at least one table's worth of rows,
            # normalize the table once and reuse it across bytes and rounds.
            table = cached(self, 'member_keys', lambda: normalize(prototypes[:, None, :]
                +.35*torch.tanh(self.expert_residual[self.membership.clamp_min(0)])))
            keys = table[groups]
        else:
            keys = normalize(prototypes[groups][..., None, :]
                             +.35*torch.tanh(self.expert_residual[candidates.clamp_min(0)]))
        member_scores = (query[..., None, None, :]*keys).sum(-1)*8
        member_scores = member_scores.masked_fill(~valid, -torch.inf)
        member_prob = member_scores.softmax(-1)
        picked = member_scores.topk(c.experts_per_group, -1).indices
        ids = candidates.gather(-1, picked)
        eg = member_prob.gather(-1, picked)*valid.sum(-1, keepdim=True)/c.experts_per_group
        gg = prob.gather(-1, groups)*c.groups/c.active_groups
        return dict(groups=groups, group_scores=scores, group_prob=prob, experts=ids.flatten(-2),
                    gates=(eg*gg[..., None]).flatten(-2), aux=switch_balance(prob, groups))


class RuntimeBlock(RecurrentBlock):
    def __init__(self, c, cfg, layer, count, shared):
        nn.Module.__init__(self)
        self.norm1, self.norm2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.memory = RuntimeMinKron(c.width, cfg.kron_heads, cfg.kron_key, cfg.kron_value, cfg.chunk_size)
        self.router = RuntimeRouter(c, layer, count)
        self.experts = SplitExpertBank(c, shared, count-shared.up.shape[0])


class RuntimeChild(RecurrentChild):
    def __init__(self, c, cfg, shared, with_decoder=False):
        nn.Module.__init__(self)
        self.cfg = c
        self.blocks = nn.ModuleList(RuntimeBlock(c, cfg, l, n, shared[l]) for l, n in enumerate(c.expert_counts))
        self.lace = RuntimeLaceController(c)
        self.final_norm = nn.LayerNorm(c.width)
        self.decoder = nn.Linear(c.width, 256, bias=False) if with_decoder else None
        if self.decoder is not None:
            nn.init.normal_(self.decoder.weight, std=.02)

    def execute_node(self, payload, prior, states, backend='chunk', memory_enabled=True, trace=False, decode=True):
        if not enabled():
            return super().execute(payload, prior, states, backend, memory_enabled, trace, decode)
        x, aux = payload, payload.new_zeros(())
        actual, final, details, semantic, group_ids, expert_ids = [], [], [], [], [], []
        for layer, block in enumerate(self.blocks):
            x, memory, route = block(x, prior[:, :, layer], states[layer], backend, memory_enabled)
            final.append(memory)
            aux = aux+route['aux']
            actual.append(route['group_scores'])
            semantic.append(route['group_prob'])
            group_ids.append(route['groups'])
            if trace:
                expert_ids.append(route['experts'])
                details.append({k: v.detach() for k, v in route.items()})
        latent = self.final_norm(x)
        if decode and self.decoder is None:
            raise ValueError('This reusable node has no byte decoder; use decode=False.')
        with torch.no_grad():
            _, lace = self.lace(torch.stack(actual, -2).detach(),
                                F.layer_norm(payload.detach(), (self.cfg.width,)), observe_only=True)
        return dict(logits=self.decoder(latent) if decode else None, state=tuple(final), aux=aux/len(self.blocks),
                    latent=latent, semantic=torch.stack(semantic, -2), group_ids=torch.stack(group_ids, -2),
                    expert_ids=torch.stack(expert_ids, -2) if trace else None,
                    feedback_lace_ids=lace['lace_ids'], trace=details)
