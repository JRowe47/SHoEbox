"""Cortical-only reductions of work that the publication path does not consume.

The minKron state equations follow experiments.hat_minkron.kron and retain
its attribution. No archived experiment or native extension is modified.
"""
import math
from contextvars import ContextVar
from functools import wraps
import torch
from torch.nn import functional as F
from experiments.hat_minkron.kron import normalize_direction
from experiments.hat_minkron.semantic_core import switch_balance, normalize
from .fused_kron import fused_final
from .fused_experts import expert_forward


_cache = ContextVar('cortical_forward_cache',default=None)


def forward_scope(function):
    @wraps(function)
    def run(self,*args,**kwargs):
        token = _cache.set({} if self.execution_backend in ('optimized','fused') else None)
        try:
            return function(self,*args,**kwargs)
        finally:
            _cache.reset(token)
    return run


def cached(module,name,compute):
    cache = _cache.get()
    if cache is None:
        return compute()
    key = (module,name,torch.is_grad_enabled())
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def compact_final(k, v, g, beta, write, state):
    """Exact chunk final state; skip all unconsumed intermediate query reads."""
    acc = torch.float64 if k.dtype == torch.float64 else torch.float32
    k, v, g, beta, write = [a.transpose(1, 2).to(acc) for a in (k, v, g, beta, write)]
    b,h,t,d = k.shape
    s = k.new_zeros(b,h,d,v.shape[-1]) if state is None else state.to(acc)
    if t == 1:
        s = s*g[:, :, 0].exp().unsqueeze(-1)
        norm = v[:, :, 0].square().sum(-1)
        inv = norm.clamp_min(1e-18).reciprocal()
        score = ((k[:, :, 0].unsqueeze(-1)*s)*v[:, :, 0].unsqueeze(-2)).sum((-2,-1))
        u = write[:, :, 0]*norm*inv-beta[:, :, 0]*inv*score
        return s+u[...,None,None]*k[:, :, 0, :, None]*v[:, :, 0, None, :]
    gc = g.cumsum(-2)
    gam = gc.exp()
    mask = torch.ones(t,t,dtype=torch.bool,device=k.device).tril()
    decay = (gc.unsqueeze(-2)-gc.unsqueeze(-3)).masked_fill(~mask.unsqueeze(-1),0).exp()
    kk = (k.unsqueeze(-2)*k.unsqueeze(-3)*decay).sum(-1)
    norm = v.square().sum(-1)
    inv = norm.clamp_min(1e-18).reciprocal()
    strength = beta*inv
    matrix = (strength.unsqueeze(-1)*kk*(v@v.transpose(-1,-2))).tril(-1)
    matrix = matrix+torch.eye(t,dtype=acc,device=k.device)
    old = (((k*gam)@s)*v).sum(-1)
    rhs = write*norm*inv-strength*old
    u = torch.linalg.solve_triangular(matrix,rhs.unsqueeze(-1),upper=False,unitriangular=True)
    tail = k*(gc[..., -1:, :]-gc).exp()
    return gam[..., -1, :, None]*s+tail.transpose(-1,-2)@(u*v)


def minkron_last(memory, x, state, *, fused=False):
    """Read only the final position, while retaining every state write and gradient."""
    b,t,_ = x.shape
    h,kd,vd = memory.H,memory.K,memory.V
    qkv,erase,write,forget,gate = memory.projection(x).split(memory.splits,-1)
    q,k,v = qkv.split([h*kd,h*kd,h*vd],-1)
    q = normalize_direction(F.silu(q[:, -1:]).reshape(b,1,h,kd),safe_zero=memory.safe_zero)
    k = normalize_direction(F.silu(k).reshape(b,t,h,kd),safe_zero=memory.safe_zero)
    v = normalize_direction(F.silu(v).reshape(b,t,h,vd),safe_zero=memory.safe_zero)
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    f = memory.forget_projection(forget).to(acc)+memory.dt_bias.to(acc)
    rate = cached(memory,('rate',acc),lambda:memory.A_log.to(acc).exp())
    g = -rate[None,None,:,None]*F.softplus(f).reshape(b,t,h,kd)
    erase,write = erase.sigmoid(),write.to(acc).sigmoid()
    final = fused_final(k.to(acc),v.to(acc),g,erase.to(acc),write,state) if fused else None
    if final is None:
        final = state
        for start in range(0,t,memory.chunk_size):
            end = start+memory.chunk_size
            final = compact_final(k[:,start:end],v[:,start:end],g[:,start:end],
                                  erase[:,start:end],write[:,start:end],final)
    z = (q[:,0].to(acc).unsqueeze(-1)*final).sum(-2)*kd**-.5
    z = z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5)*memory.norm_weight.to(acc)
    z = z*(gate[:, -1]+memory.output_gate_bias).reshape(b,h,vd).sigmoid()
    return memory.output(z.reshape(b,1,h*vd).to(x.dtype)),final


def symbol_single(level, x, previous, *, terminal, optimized, fused=False):
    """The cortical HAT always has one joint, ordered source sequence."""
    z = level.norm(x)
    if terminal and optimized:
        context,final = minkron_last(level.memory,z,previous,fused=fused)
        h = x[:, -1:]+context
    else:
        context,final = level.memory(z,previous,backend='chunk')
        h = x+context
        if terminal:
            h = h[:, -1:]
    q = normalize_direction(level.query(level.norm(h)),safe_zero=level.safe_zero)
    keys = cached(level,'keys',lambda:F.normalize(level.keys,dim=-1))
    scores = 4*F.linear(q,keys)
    probabilities = scores.softmax(-1)
    ids = scores.topk(level.cfg.hat_active_symbols,-1).indices
    weights = probabilities.gather(-1,ids)*level.cfg.hat_symbols/level.cfg.hat_active_symbols
    symbol = (level.values[ids]*weights[...,None]).sum(-2)/math.sqrt(level.cfg.hat_active_symbols)
    return h+symbol,final,switch_balance(probabilities,ids)


def pilot(router,x):
    prototypes = cached(router,'prototypes',lambda:normalize(router.prototypes))
    return 4*F.linear(normalize(router.query(x)),prototypes)


def route(router,x,prior):
    c = router.c
    prototypes = cached(router,'prototypes',lambda:normalize(router.prototypes))
    query = normalize(router.query(x)+c.lace_gain*router.message(prior@prototypes))
    scores = 4*F.linear(query,prototypes)
    message = prior.clamp_min(1e-8).log()
    scores = scores+c.lace_gain*(message-message.mean(-1,keepdim=True))
    probability = scores.softmax(-1)
    groups = scores.topk(c.active_groups,-1).indices
    candidates = router.membership[groups]
    valid = candidates >= 0
    keys = cached(router,'member_keys',lambda:normalize(prototypes[:,None,:]+
                   .35*torch.tanh(router.expert_residual[router.membership.clamp_min(0)])))
    member_scores = (query[...,None,None,:]*keys[groups]).sum(-1)*8
    member_scores = member_scores.masked_fill(~valid,-torch.inf)
    member_probability = member_scores.softmax(-1)
    picked = member_scores.topk(c.experts_per_group,-1).indices
    ids = candidates.gather(-1,picked)
    eg = member_probability.gather(-1,picked)*valid.sum(-1,keepdim=True)/c.experts_per_group
    gg = probability.gather(-1,groups)*c.groups/c.active_groups
    return {'experts':ids.flatten(-2),'gates':(eg*gg[...,None]).flatten(-2),
            'aux':switch_balance(probability,groups)}


def lace_forward(lace,evidence,context):
    c = lace.c
    templates = cached(lace,'templates',lambda:lace.template_logits.softmax(-1))
    shape = evidence.shape
    e = evidence.reshape(-1,c.layers,c.groups)
    ctx = lace.context_query(context).reshape(-1,c.laces)
    layer_ids = cached(lace,'layer_ids',lambda:torch.arange(c.layers,device=e.device)[None,:,None])
    current = e
    for iteration in range(c.refinement_rounds):
        agreement = torch.einsum('nlg,mlg->nml',current.softmax(-1),templates)
        support = agreement.clamp_min(1e-8).log()
        external = (ctx[:,:,None]+support.sum(-1,keepdim=True)-support).transpose(1,2)
        scores,ids = external.topk(c.active_laces,-1)
        weights = scores.softmax(-1)
        prior = (templates[ids,layer_ids]*weights[...,None]).sum(-2)
        if iteration+1 < c.refinement_rounds:
            message = prior.clamp_min(1e-8).log()
            current = e+c.lace_gain*(message-message.mean(-1,keepdim=True))
    return prior.reshape(shape),{'balance':switch_balance(external.softmax(-1),ids)}


def execute_private(core, x, prior, states, *, fused=False):
    """Skip shared-bank dispatch and an unused second lace calculation."""
    aux,final,expert_ids = x.new_zeros(()),[],[]
    for index,block in enumerate(core.blocks):
        memory,state = (minkron_last(block.memory,block.norm1(x),states[index],fused=True) if fused else
                        block.memory(block.norm1(x),states[index],backend='step'))
        x = x+memory
        z = block.norm2(x)
        routed = route(block.router,z,prior[:, :, index])
        if block.experts.shared_count:
            raise ValueError('Private cortical execution cannot contain shared banks.')
        bank = block.experts.private
        update = expert_forward(bank,z,routed['experts'],routed['gates'],cached) if fused else None
        if update is None:
            update = bank(z,routed['experts'],routed['gates'])
        x = x+update
        final.append(state)
        aux = aux+routed['aux']
        expert_ids.append(routed['experts'])
    return {'latent':core.final_norm(x),'state':tuple(final),'aux':aux/len(core.blocks),
            'expert_ids':torch.stack(expert_ids,-2)}
