"""Exact hard-byte transport with the original straight-through derivative."""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class _HardEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, soft, weight):
        ctx.save_for_backward(ids, weight)
        return F.embedding(ids, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ids, weight = ctx.saved_tensors
        grad_soft = F.linear(gradient, weight) if ctx.needs_input_grad[1] else None
        grad_weight = None
        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight).index_add(0, ids.reshape(-1), gradient.reshape(-1, weight.shape[1]))
        return None, grad_soft, grad_weight


def hard_embedding(ids, soft, weight):
    return _HardEmbedding.apply(ids, soft, weight)
