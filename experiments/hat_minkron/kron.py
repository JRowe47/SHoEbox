"""Compact Kron recurrence, adapted from the verified October 21 implementation.
Bounded chunks retain only a fixed FP32 matrix per head between calls.
See references/ for original source, authors' reference and MIT notices.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize_direction(x, dim=-1, eps=1e-12, safe_zero=False):
    """A missing direction has zero value and, optionally, zero derivative.

    The ordinary normalization Jacobian at zero is 1/eps. Composing memory
    and symbol queries can amplify that arbitrary derivative for absent inputs.
    Opt-in zero-direction semantics leave nonzero normalization unchanged.
    """
    value = F.normalize(x, dim=dim, eps=eps)
    if safe_zero:
        present = x.norm(dim=dim, keepdim=True) > eps
        value = torch.where(present, value, torch.zeros_like(value))
    return value


def compact_chunk(q,k,v,g,beta,c,state=None):
    """Exact one-chunk recurrence; supports arbitrary head sizes, FP32/FP64."""
    orig=q.dtype;acc=torch.float64 if orig==torch.float64 else torch.float32
    q,k,v,g,beta,c=[x.transpose(1,2).to(acc) for x in (q,k,v,g,beta,c)]
    B,H,T,K=q.shape;V=v.shape[-1]
    S=q.new_zeros(B,H,K,V) if state is None else state.to(acc)
    if T==1:
        S=S*g[:,:,0].exp().unsqueeze(-1)
        norm=v[:,:,0].square().sum(-1)
        inv=norm.clamp_min(1e-18).reciprocal()
        score=((k[:,:,0].unsqueeze(-1)*S)*v[:,:,0].unsqueeze(-2)).sum((-2,-1))
        u=c[:,:,0]*norm*inv-beta[:,:,0]*inv*score
        S=S+u[...,None,None]*k[:,:,0,:,None]*v[:,:,0,None,:]
        o=(q[:,:,0,:,None]*S).sum(-2)*K**-.5
        return o.unsqueeze(1).to(orig),S
    gc=g.cumsum(-2);gam=gc.exp()
    mask=torch.ones(T,T,dtype=torch.bool,device=q.device).tril()
    L=(gc.unsqueeze(-2)-gc.unsqueeze(-3)).masked_fill(~mask.unsqueeze(-1),0).exp()
    Akk=(k.unsqueeze(-2)*k.unsqueeze(-3)*L).sum(-1)
    Aqk=(q.unsqueeze(-2)*k.unsqueeze(-3)*L).sum(-1)*mask*K**-.5
    vn=v.square().sum(-1);inv=vn.clamp_min(1e-18).reciprocal();bs=beta*inv
    M=(bs.unsqueeze(-1)*Akk*(v@v.transpose(-1,-2))).tril(-1)
    M=M+torch.eye(T,dtype=acc,device=q.device)
    p=(((k*gam)@S)*v).sum(-1)
    rhs=c*vn*inv-bs*p
    u=torch.linalg.solve_triangular(M,rhs.unsqueeze(-1),upper=False,unitriangular=True)
    uv=u*v
    out=K**-.5*((q*gam)@S)+Aqk@uv
    tail=k*(gc[...,-1:,:]-gc).exp()
    final=gam[...,-1,:,None]*S+tail.transpose(-1,-2)@uv
    return out.transpose(1,2).contiguous().to(orig),final


class MinKron(nn.Module):
    """Permanent packed projection; same normalized, gated compact recurrence.

    chunk_size bounds temporary pairwise work. Persistent state is [B,H,K,V].
    reset_each is a frozen-model intervention that removes historical memory
    while retaining the current event's write/read computation.
    """
    def __init__(self, width, heads=4, key_dim=8, value_dim=8, chunk_size=32, safe_zero=False):
        super().__init__()
        self.width, self.H, self.K, self.V = width, heads, key_dim, value_dim
        self.chunk_size = chunk_size
        self.safe_zero = safe_zero
        if min(width, heads, key_dim, value_dim, chunk_size) < 1:
            raise ValueError('All minKron dimensions must be positive.')
        self.splits = [heads*(2*key_dim+value_dim), heads, heads, value_dim, heads*value_dim]
        self.projection = nn.Linear(width, sum(self.splits), bias=False)
        self.forget_projection = nn.Linear(value_dim, heads*key_dim, bias=False)
        self.A_log = nn.Parameter(torch.empty(heads))
        self.dt_bias = nn.Parameter(torch.empty(heads*key_dim))
        self.output_gate_bias = nn.Parameter(torch.full((heads*value_dim,), 3.))
        self.norm_weight = nn.Parameter(torch.ones(value_dim))
        self.output = nn.Linear(heads*value_dim, width, bias=False)
        with torch.no_grad():
            self.A_log.uniform_(0, 16).clamp_min_(1e-6).log_()
            dt = torch.exp(torch.rand(heads*key_dim)*(math.log(.1)-math.log(.001))+math.log(.001))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.projection.weight[-heads*value_dim:].zero_()
            nn.init.normal_(self.output.weight, std=.02)
        for p in (self.A_log, self.dt_bias, self.output_gate_bias, self.norm_weight):
            p._no_weight_decay = True

    def initial_state(self, batch, device, dtype=torch.float32):
        acc = torch.float64 if dtype == torch.float64 else torch.float32
        return torch.zeros(batch, self.H, self.K, self.V, device=device, dtype=acc)

    def forward(self, x, state=None, backend='chunk', reset_each=False):
        b,t,_ = x.shape
        if t == 0: raise ValueError('minKron requires at least one event.')
        if state is not None and state.shape != (b,self.H,self.K,self.V):
            raise ValueError('Wrong minKron state shape.')
        qkv, erase, write, forget, gate = self.projection(x).split(self.splits, -1)
        q,k,v = F.silu(qkv).split([self.H*self.K,self.H*self.K,self.H*self.V], -1)
        q,k,v = [normalize_direction(a.reshape(b,t,self.H,d),safe_zero=self.safe_zero)
                 for a,d in ((q,self.K),(k,self.K),(v,self.V))]
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        f = self.forget_projection(forget).to(acc) + self.dt_bias.to(acc)
        g = -self.A_log.to(acc).exp()[None,None,:,None]*F.softplus(f).reshape(b,t,self.H,self.K)
        args = (q,k,v,g,erase.sigmoid(),write.to(acc).sigmoid())
        if reset_each:
            flat = [a.reshape(b*t,1,*a.shape[2:]) for a in args]
            out, final = compact_chunk(*flat)
            out = out.reshape(b,t,self.H,self.V)
            final = final.reshape(b,t,self.H,self.K,self.V)[:,-1].clone()
        else:
            size = 1 if backend == 'step' else self.chunk_size
            if backend not in ('step','chunk'): raise ValueError('backend must be chunk or step')
            pieces, final = [], state
            for start in range(0,t,size):
                out, final = compact_chunk(*(a[:,start:start+size] for a in args), final)
                pieces.append(out)
            out = torch.cat(pieces, 1)
        # RMS normalization over each head's value coordinates, then output gate.
        z = out.to(acc)
        z = z * torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5) * self.norm_weight.to(acc)
        z = z * (gate+self.output_gate_bias).reshape(b,t,self.H,self.V).sigmoid()
        return self.output(z.reshape(b,t,self.H*self.V).to(x.dtype)), final

