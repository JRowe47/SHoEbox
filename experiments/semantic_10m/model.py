"""Variable-depth SHoE nanoGPT with context-resolved semantic laces.

Pure PyTorch reference implementation. Expert matmuls and attention reads are
conditional; embeddings, QKV projections, prototype scoring and AdamW are dense.
No dense T x T attention score or mask is built. This is an LSA-inspired stable
prototype index, not a reproduction of Ward clustering / LSA tree search.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 65
    context: int = 128
    width: int = 100
    layers: int = 3
    heads: int = 4
    experts: int = 1024  # total, divided 342 / 341 / 341
    expert_hidden: int = 4
    address_dim: int = 16
    groups: int = 32
    active_groups: int = 2
    experts_per_group: int = 8
    laces: int = 64
    active_laces: int = 2
    refinement_rounds: int = 2
    lace_gain: float = 0.75
    local_window: int = 16
    keys_per_group: int = 8
    seed: int = 20261103

    def validate(self):
        assert self.layers >= 2, 'A lace connects at least two layers.'
        assert self.width % self.heads == 0
        assert self.context >= self.local_window > 0
        assert 1 <= self.active_groups <= self.groups
        assert 1 <= self.active_laces <= self.laces
        assert self.refinement_rounds >= 1 and self.keys_per_group > 0
        assert min(self.expert_counts) // self.groups >= self.experts_per_group

    @classmethod
    def scaled_10m(cls, **kwargs):
        """Six layers, 4,096 experts, 64 groups/layer and 256 laces."""
        return cls(width=180, layers=6, heads=6, experts=4096,
                   expert_hidden=6, address_dim=32, groups=64, laces=256,
                   **kwargs)

    @property
    def expert_counts(self):
        return tuple(self.experts // self.layers + (i < self.experts % self.layers)
                     for i in range(self.layers))


def normalize(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def prefix_mean(x):
    return x.cumsum(1) / torch.arange(1, x.shape[1]+1, device=x.device)[None, :, None]


def switch_balance(prob, choices):
    count = prob.shape[-1]
    load = torch.bincount(choices.reshape(-1), minlength=count).to(prob.dtype)
    load = load / choices.numel()
    return count * (prob.reshape(-1, count).mean(0) * load).sum()


class LaceController(nn.Module):
    """Learned depth-way factors with sparse, bidirectional routing rounds.

For target layer l, the evidence contributed by l is left out of the factor
score. This is an extrinsic message from other layers, not its own score echoed
back. Only the selected lace factors contribute to each message.
"""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.context_query = nn.Linear(c.width, c.laces, bias=False)
        template = torch.full((c.laces, c.layers, c.groups), -1.0)
        gen = torch.Generator().manual_seed(c.seed + 51)
        for layer in range(c.layers):
            ids = torch.cat([torch.randperm(c.groups, generator=gen)
                             for _ in range(math.ceil(c.laces/c.groups))])[:c.laces]
            template[torch.arange(c.laces), layer, ids] = 2.0
        template += .05 * torch.randn(template.shape, generator=gen)
        self.template_logits = nn.Parameter(template)

    def forward(self, evidence, context, enabled=True, permutation=None):
        # evidence: B,T,L,G, independently organized group axes at every layer.
        c = self.c
        templates = self.template_logits.softmax(-1)
        if permutation is not None:
            # Diagnostic intervention: scramble one layer's relation endpoints.
            middle = c.layers // 2
            templates = torch.stack([
                templates[:, layer, permutation] if layer == middle else templates[:, layer]
                for layer in range(c.layers)], 1)
        base_shape = evidence.shape
        e = evidence.reshape(-1, c.layers, c.groups)
        ctx = self.context_query(context).reshape(-1, c.laces)
        if not enabled:
            prior = torch.full_like(e, 1/c.groups)
            return prior.reshape(base_shape), {'lace_ids': None, 'balance': e.new_zeros(())}
        current = e
        for _ in range(c.refinement_rounds):
            agreement = torch.einsum('nlg,mlg->nml', current.softmax(-1), templates)
            support = agreement.clamp_min(1e-8).log()
            # N,L,M; each target sees context and all OTHER layers' evidence.
            external = (ctx[:, :, None] + support.sum(-1, keepdim=True) - support).transpose(1, 2)
            scores, ids = external.topk(c.active_laces, -1)
            weight = scores.softmax(-1)
            layer_ids = torch.arange(c.layers, device=e.device)[None, :, None]
            chosen = templates[ids, layer_ids]  # N,L,K,G
            prior = (chosen * weight[..., None]).sum(-2)
            message = prior.clamp_min(1e-8).log()
            message = message - message.mean(-1, keepdim=True)
            current = e + c.lace_gain * message
        balance = switch_balance(external.softmax(-1), ids)
        return prior.reshape(base_shape), {
            'lace_ids': ids.reshape(*base_shape[:2], c.layers, c.active_laces),
            'lace_weights': weight.reshape(*base_shape[:2], c.layers, c.active_laces),
            'balance': balance,
        }


def semantic_reads(assignments, selected_groups, window, per_group):
    """Bounded causal postings lookup, with no query x historical-key scan.

Each key has one prototype assignment. Cumulative counts provide the end of
each query's causal prefix in each posting list, so future writes cannot leak.
Recent local keys are excluded from remote reads to avoid double counting.
"""
    b, t = assignments.shape
    g = int(assignments.max()) + 1
    g = max(g, int(selected_groups.max()) + 1)
    device = assignments.device
    counts = F.one_hot(assignments, g).cumsum(1)
    ranks = counts.gather(-1, assignments[..., None]).squeeze(-1) - 1
    postings = torch.full((b, g, t), -1, dtype=torch.long, device=device)
    batch = torch.arange(b, device=device)[:, None]
    positions = torch.arange(t, device=device)
    postings[batch, assignments, ranks] = positions[None]
    # Remote lookup ends BEFORE the local window. Recent same-group keys must
    # not consume the remote budget and hide otherwise available older keys.
    remote_end = (positions-window).clamp_min(0)
    old_counts = counts[:, remote_end] * (positions >= window)[None, :, None]
    end = old_counts.gather(-1, selected_groups)
    read_rank = end[..., None] - 1 - torch.arange(per_group, device=device)
    remote = postings[torch.arange(b, device=device)[:, None, None, None],
                      selected_groups[..., None], read_rank.clamp_min(0)]
    valid_remote = ((read_rank >= 0) & (remote >= 0)
                    & (remote < positions[None, :, None, None] - window + 1))
    local = positions[:, None] - torch.arange(window, device=device)
    local = local[None].expand(b, -1, -1)
    reads = torch.cat([local, remote.flatten(-2)], -1)
    valid = torch.cat([local >= 0, valid_remote.flatten(-2)], -1)
    return reads.clamp_min(0), valid


class SemanticRouter(nn.Module):
    def __init__(self, c, layer, expert_count):
        super().__init__()
        self.c, self.expert_count = c, expert_count
        self.query = nn.Linear(c.width, c.address_dim, bias=False)
        self.key = nn.Linear(c.width, c.address_dim, bias=False)
        self.message = nn.Linear(c.address_dim, c.address_dim, bias=False)
        self.prototypes = nn.Parameter(torch.randn(c.groups, c.address_dim))
        self.expert_residual = nn.Parameter(torch.randn(expert_count, c.address_dim)*.2)
        # Independent, balanced IDs per layer. Experts learn inside stable groups.
        gen = torch.Generator().manual_seed(c.seed + 101*(layer+1))
        perm = torch.randperm(expert_count, generator=gen)
        slots = math.ceil(expert_count/c.groups)
        membership = torch.full((c.groups, slots), -1, dtype=torch.long)
        for group in range(c.groups):
            ids = perm[group::c.groups]
            membership[group, :len(ids)] = ids
        self.register_buffer('membership', membership)

    def pilot(self, x):
        return 4 * F.linear(normalize(self.query(x)), normalize(self.prototypes))

    def forward(self, x, prior, lace_enabled=True):
        c = self.c
        prototypes = normalize(self.prototypes)
        query = self.query(x + .25*prefix_mean(x))
        if lace_enabled:
            query = query + c.lace_gain*self.message(prior @ prototypes)
        query = normalize(query)
        scores = 4 * F.linear(query, prototypes)
        if lace_enabled:
            message = prior.clamp_min(1e-8).log()
            scores = scores + c.lace_gain*(message-message.mean(-1, keepdim=True))
        group_prob = scores.softmax(-1)
        groups = scores.topk(c.active_groups, -1).indices
        candidates = self.membership[groups]
        valid = candidates >= 0
        keys = normalize(prototypes[groups][..., None, :] +
                         .35*torch.tanh(self.expert_residual[candidates.clamp_min(0)]))
        member_scores = (query[..., None, None, :]*keys).sum(-1)*8
        member_scores = member_scores.masked_fill(~valid, -torch.inf)
        member_prob = member_scores.softmax(-1)
        picked = member_scores.topk(c.experts_per_group, -1).indices
        experts = candidates.gather(-1, picked)
        # Full candidate normalization keeps task gradients into unchosen members.
        eg = member_prob.gather(-1, picked)*valid.sum(-1, keepdim=True)/c.experts_per_group
        gg = group_prob.gather(-1, groups)*c.groups/c.active_groups
        gates = (eg*gg[..., None]).flatten(-2)
        key_address = normalize(self.key(x))
        key_score = F.linear(key_address, prototypes)
        assignments = key_score.argmax(-1)
        alignment = 1-key_score.gather(-1, assignments[..., None]).mean()
        balance = switch_balance(group_prob, groups) + switch_balance((key_score*4).softmax(-1), assignments)
        return {
            'query': query, 'key': key_address, 'groups': groups,
            'group_scores': scores, 'group_prob': group_prob,
            'assignments': assignments, 'experts': experts.flatten(-2),
            'gates': gates, 'aux': balance + .1*alignment,
        }


class SparseAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.width, 3*c.width, bias=False)
        self.output = nn.Linear(c.width, c.width, bias=False)

    def forward(self, x, route):
        c = self.c
        b, t, d = x.shape
        h, hd = c.heads, d//c.heads
        q, k, v = self.qkv(x).reshape(b, t, 3, h, hd).unbind(2)
        reads, valid = semantic_reads(route['assignments'], route['groups'],
                                      c.local_window, c.keys_per_group)
        batch = torch.arange(b, device=x.device)[:, None, None]
        selected_k, selected_v = k[batch, reads], v[batch, reads]
        score = (q[:, :, None]*selected_k).sum(-1)/math.sqrt(hd)
        selected_address = route['key'][batch, reads]
        semantic_score = (route['query'][:, :, None]*selected_address).sum(-1)
        # Every selected key receives a differentiable prototype score as well
        # as exact QK and address similarity. Membership itself is discrete.
        key_groups = route['assignments'][batch, reads]
        group_bias = route['group_scores'].gather(-1, key_groups)
        score = score + .25*semantic_score[..., None] + .1*group_bias[..., None]
        score = score.masked_fill(~valid[..., None], -torch.inf)
        weights = score.softmax(-2)
        y = (weights[..., None]*selected_v).sum(2).reshape(b, t, d)
        return self.output(y), {'reads': reads, 'valid': valid}


class ExpertBank(nn.Module):
    def __init__(self, c, count):
        super().__init__()
        self.up = nn.Parameter(torch.randn(count, c.expert_hidden, c.width)*.1)
        self.down = nn.Parameter(torch.randn(count, c.expert_hidden, c.width)*.02)

    def forward(self, x, ids, gates):
        # Only chosen weights enter expert arithmetic. PyTorch's gather backward
        # still allocates a dense bank gradient; this is not a fused sparse kernel.
        b, t, d = x.shape
        chosen_up, chosen_down = self.up[ids], self.down[ids]
        hidden = F.gelu((chosen_up*x[..., None, None, :]).sum(-1))
        y = (hidden[..., None]*chosen_down).sum(-2)
        # sqrt(K) aggregation maintains scale across many tiny specialists.
        return (y*gates[..., None]).sum(-2)/math.sqrt(ids.shape[-1])


class Block(nn.Module):
    def __init__(self, c, layer, count):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.router = SemanticRouter(c, layer, count)
        self.attention = SparseAttention(c)
        self.experts = ExpertBank(c, count)

    def forward(self, x, prior, lace_enabled=True):
        normalized = self.norm1(x)
        route = self.router(normalized, prior, lace_enabled)
        attended, memory = self.attention(normalized, route)
        x = x + attended
        x = x + self.experts(self.norm2(x), route['experts'], route['gates'])
        return x, route, memory


class SHoEGPT(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        self.cfg = c or Config()
        self.cfg.validate()
        c = self.cfg
        self.token_embedding = nn.Embedding(c.vocab_size, c.width)
        self.position_embedding = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c, i, n) for i, n in enumerate(c.expert_counts)])
        self.lace = LaceController(c)
        self.final_norm = nn.LayerNorm(c.width)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=.02)

    def forward(self, tokens, targets=None, lace_enabled=True, trace=False, permutation=None):
        b, t = tokens.shape
        if t > self.cfg.context:
            raise ValueError('Input exceeds configured context; crop or reconfigure.')
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(t, device=tokens.device))
        context = F.layer_norm(x+.25*prefix_mean(x), (self.cfg.width,))
        evidence = torch.stack([block.router.pilot(context) for block in self.blocks], -2)
        prior, lace_trace = self.lace(evidence, context, lace_enabled, permutation)
        aux, details = x.new_zeros(()), []
        for layer, block in enumerate(self.blocks):
            x, route, memory = block(x, prior[:, :, layer], lace_enabled)
            aux = aux + route['aux']
            if trace:
                details.append({k: v.detach() for k, v in {**route, **memory}.items()})
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1)) if targets is not None else None
        aux = aux/self.cfg.layers + .1*lace_trace['balance']
        return {'logits': logits, 'ce': ce, 'aux': aux,
                'trace': details if trace else None,
                'lace_ids': lace_trace['lace_ids'].detach() if trace and lace_trace['lace_ids'] is not None else None}

    def parameter_summary(self):
        parts = {'expert_weights': 0, 'expert_addresses': 0, 'attention': 0,
                 'routing_and_laces': 0, 'embeddings': 0, 'norms': 0}
        for name, p in self.named_parameters():
            if '.experts.' in name: key = 'expert_weights'
            elif 'expert_residual' in name: key = 'expert_addresses'
            elif '.attention.' in name: key = 'attention'
            elif 'embedding' in name: key = 'embeddings'
            elif 'norm' in name: key = 'norms'
            else: key = 'routing_and_laces'
            parts[key] += p.numel()
        return {'total': sum(parts.values()), 'components': parts,
                'expert_counts': list(self.cfg.expert_counts),
                'active_experts_per_layer': self.cfg.active_groups*self.cfg.experts_per_group,
                'active_expert_weight_parameters_per_token': self.cfg.layers*self.cfg.active_groups*self.cfg.experts_per_group*2*self.cfg.width*self.cfg.expert_hidden}

    @torch.no_grad()
    def generate(self, tokens, count=200, temperature=.8, seed=123, lace_enabled=True):
        gen = torch.Generator(device=tokens.device).manual_seed(seed)
        for _ in range(count):
            logits = self(tokens[:, -self.cfg.context:], lace_enabled=lace_enabled)['logits'][:, -1]/temperature
            sample = torch.multinomial(logits.softmax(-1), 1, generator=gen)
            tokens = torch.cat([tokens, sample], 1)
        return tokens


if __name__ == '__main__':
    import json
    model = SHoEGPT()
    print(json.dumps({'config': asdict(model.cfg), **model.parameter_summary()}, indent=2))
