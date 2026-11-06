"""Compact expert derivatives and an analytical single-event Kron backward."""
import torch
from torch import Tensor
from typing import Tuple
from .native_runtime import for_tensor
from torch.nn import functional as F


class DenseBankGradient(torch.autograd.Function):
    """One boundary per bank/event: compact row gradients become dense here.

    Parameter gradients remain ordinary dense tensors, including zero rows,
    so AdamW moments, decay, hooks and autograd.grad retain their semantics.
    No backward writes directly to Parameter.grad or to a shared mutable buffer.
    """
    @staticmethod
    def forward(ctx, weight):
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, gradient):
        if gradient.is_sparse and not torch.is_grad_enabled():
            native = for_tensor(gradient)
            if native is not None:
                return native.dense_rows(gradient._indices()[0].contiguous(),
                                         gradient._values().contiguous(), gradient.shape[0])
        return gradient.to_dense() if gradient.is_sparse else gradient


class SelectedRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, ids):
        ctx.save_for_backward(ids)
        ctx.shape = weight.shape
        return weight.index_select(0, ids)

    @staticmethod
    def backward(ctx, gradient):
        ids, = ctx.saved_tensors
        return torch.sparse_coo_tensor(ids[None], gradient, ctx.shape,
                                      device=gradient.device, dtype=gradient.dtype,
                                      check_invariants=False), None


class SelectedExpert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, gates):
        pre = (up*x[:, None]).sum(-1)
        hidden = F.gelu(pre)
        raw = (hidden[..., None]*down).sum(-2)
        ctx.save_for_backward(x, up, down, gates, pre, hidden, raw)
        return raw*gates[:, None]

    @staticmethod
    def backward(ctx, gradient):
        x, up, down, gates, pre, hidden, raw = ctx.saved_tensors
        if torch.is_grad_enabled():
            inputs = (x, up, down, gates)
            requested = [v for v in inputs if v.requires_grad]
            hidden = F.gelu((up*x[:, None]).sum(-1))
            output = (hidden[..., None]*down).sum(-2)*gates[:, None]
            derivatives = iter(torch.autograd.grad(output, requested, gradient, create_graph=True))
            return tuple(next(derivatives) if v.requires_grad else None for v in inputs)
        native = for_tensor(x)
        if native is not None:
            return tuple(native.expert_backward(*(v.contiguous() for v in (gradient, *ctx.saved_tensors))))
        delta = gradient*gates[:, None]
        dz = (delta[:, None]*down).sum(-1)*(.5*(1+torch.erf(pre/2**.5))+
            pre*torch.exp(-.5*pre.square())/(2*torch.pi)**.5)
        return (dz[..., None]*up).sum(-2), dz[..., None]*x[:, None], delta[:, None]*hidden[..., None], (gradient*raw).sum(-1)


@torch.jit.script
def recurrence_forward(q: Tensor, k: Tensor, v: Tensor, g: Tensor,
                       beta: Tensor, write: Tensor, previous: Tensor):
    decay = g.exp()
    decayed = previous*decay.unsqueeze(-1)
    norm = v.square().sum(-1)
    inv = norm.clamp_min(1e-18).reciprocal()
    score = ((k.unsqueeze(-1)*decayed)*v.unsqueeze(-2)).sum((-2, -1))
    u = write*norm*inv-beta*inv*score
    final = decayed+u[..., None, None]*k.unsqueeze(-1)*v.unsqueeze(-2)
    out = (q.unsqueeze(-1)*final).sum(-2)*q.shape[-1]**-.5
    return out, final, decay, decayed, norm, inv, score, u


@torch.jit.script
def recurrence_backward(do: Tensor, ds: Tensor, q: Tensor, k: Tensor, v: Tensor,
                        beta: Tensor, write: Tensor, final: Tensor, decay: Tensor,
                        decayed: Tensor, norm: Tensor, inv: Tensor, score: Tensor, u: Tensor, previous: Tensor):
    scale = q.shape[-1]**-.5
    read = do.unsqueeze(-2)*scale
    dq = (read*final).sum(-1)
    total = ds+q.unsqueeze(-1)*read
    weighted = (total*v.unsqueeze(-2)).sum(-1)
    du = (weighted*k).sum(-1)
    dscore = -du*(beta*inv)
    score_direction = dscore[..., None, None]*v.unsqueeze(-2)
    dk = weighted*u.unsqueeze(-1)+(score_direction*decayed).sum(-1)
    dv = (total*(u[..., None, None]*k.unsqueeze(-1))).sum(-2)
    dv = dv+(dscore[..., None, None]*(decayed*k.unsqueeze(-1))).sum(-2)
    dinv = du*(write*norm)+(-du*score)*beta
    dnorm = (du*inv)*write-dinv*inv.square()*(norm >= 1e-18)
    dv = dv+2*v*dnorm.unsqueeze(-1)
    ddecayed = total+score_direction*k.unsqueeze(-1)
    dg = (ddecayed*previous).sum(-1)*decay
    return dq, dk, dv, dg, (-du*score)*inv, (du*inv)*norm, ddecayed*decay.unsqueeze(-1)


class KronStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, write, previous):
        original = (q, k, v, g, beta, write, previous)
        ctx.native = for_tensor(q)
        out, final, decay, decayed, norm, inv, score, u = recurrence_forward(q, k, v, g, beta, write, previous)
        ctx.save_for_backward(*original, q, k, v, beta, write, final, decay, decayed, norm, inv, score, u, previous)
        return out, final

    @staticmethod
    def backward(ctx, do, ds):
        if torch.is_grad_enabled():
            # Higher derivatives use the differentiable formula, never detached
            # native intermediates. Ordinary training takes the combined path.
            inputs = ctx.saved_tensors[:7]
            requested = [x for x in inputs if x.requires_grad]
            out, final, _, _, _, _, _, _ = recurrence_forward(*inputs)
            derivatives = iter(torch.autograd.grad((out, final), requested, (do, ds), create_graph=True))
            return tuple(next(derivatives) if x.requires_grad else None for x in inputs)
        if ctx.native is not None:
            return tuple(ctx.native.backward(*(x.contiguous() for x in (do, ds, *ctx.saved_tensors[7:]))))
        return recurrence_backward(do, ds, *ctx.saved_tensors[7:])


@torch.jit.ignore
def kron_step(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
              write: Tensor, previous: Tensor) -> Tuple[Tensor, Tensor]:
    return KronStep.apply(q, k, v, g, beta, write, previous)
