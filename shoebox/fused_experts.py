"""Indexed CPU experts with bounded groups of dense gradient accumulation.

Only selected rows enter arithmetic. Autograd combines compact sparse row
derivatives at the boundary, which materializes a normal dense parameter gradient.
No backward mutates Parameter.grad, so hooks, autograd.grad, retained graphs,
gradient accumulation and dense AdamW keep their ordinary semantics.
"""
import os
import torch
from torch.autograd.function import once_differentiable
from .fused_kron import for_tensor


GRADIENT_GROUP_CALLS = 8


def expert_status():
    """Requested expert mode; native_status separately records availability."""
    return {'requested': os.environ.get('SHOEBOX_EXPERTS', 'auto') != '0',
            'gradient_group_calls': GRADIENT_GROUP_CALLS}


class _DenseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, native, slots):
        ctx.native = native
        ctx.set_materialize_grads(False)
        return tuple(weight.view_as(weight) for _ in range(slots))

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        gradients = [g for g in gradients if g is not None]
        return (ctx.native.dense_row_grads(gradients) if gradients else None), None, None


class _WeightViews:
    """Each use owns an autograd edge; backward never merges sparse tensors."""
    def __init__(self, weight, native):
        self.weight, self.native, self.views = weight, native, iter(())

    def take(self):
        view = next(self.views, None)
        if view is None:
            self.views = iter(_DenseGradient.apply(self.weight, self.native, GRADIENT_GROUP_CALLS))
            view = next(self.views)
        return view


class _IndexedExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, ids, gates, native):
        result, pre = native.experts_forward(x, up, down, ids, gates, True)
        ctx.native = native
        ctx.save_for_backward(x, up, down, ids, gates, pre)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, up, down, ids, gates, pre = ctx.saved_tensors
        dx, du, dd, dg = ctx.native.experts_backward(
            gradient.contiguous(), x, up, down, ids, gates, pre)
        rows = ids.reshape(1, -1)
        def sparse(values, shape):
            return torch.sparse_coo_tensor(rows, values, shape, device=values.device,
                                           dtype=values.dtype, check_invariants=False)
        return dx, sparse(du, up.shape), sparse(dd, down.shape), None, dg, None


def expert_forward(bank, x, ids, gates, cache):
    """Return fused output or None for the caller's unchanged portable path.

The cache is scoped to one model.forward, never to parameter lifetime. Reusing
    its view pool connects each group of uses to one autograd boundary per bank.
    This grouping bounds pending row gradients without merging sparse tensors;
    it never detaches temporal gradients. Separate forwards remain composable.
"""
    if os.environ.get('SHOEBOX_EXPERTS', 'auto') == '0' or torch.is_autocast_enabled('cpu'):
        return None
    native = for_tensor(x)
    if native is None or any(t.device != x.device or t.dtype != x.dtype
                             for t in (bank.up, bank.down, gates)):
        return None
    up, down = bank.up, bank.down
    if torch.is_grad_enabled():
        up = cache(bank, 'expert_up', lambda: _WeightViews(bank.up, native)).take()
        down = cache(bank, 'expert_down', lambda: _WeightViews(bank.down, native)).take()
    shape = x.shape
    args = (x.reshape(-1, shape[-1]).contiguous(), up.contiguous(), down.contiguous(),
            ids.reshape(-1, ids.shape[-1]).contiguous(),
            gates.reshape(-1, gates.shape[-1]).contiguous())
    if torch.is_grad_enabled() and any(t.requires_grad for t in args):
        result = _IndexedExperts.apply(*args, native)
    else:
        result = native.experts_forward(*args, False)[0]
    return result.reshape(shape)
