"""Fold frozen normalization and affine corrections into an inference module."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .emission_adapter import EmissionAdapter
from .readout_diagnostic import ByteDecoder


class CompiledReadout(nn.Module):
    """Feature-only frozen output head; no query/source labels or graph state."""

    def __init__(self, weight, bias, *, w1=None, b1=None, w2=None):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] != 256 or bias.shape != (256,):
            raise ValueError('Expected a 256-byte affine head.')
        nonlinear = w1 is not None
        if nonlinear:
            if (w1.ndim != 2 or w1.shape[0] != weight.shape[0] or b1 is None or w2 is None
                    or b1.shape != (w1.shape[1],) or w2.shape != (w1.shape[1], 256)):
                raise ValueError('Invalid GELU correction dimensions.')
        elif b1 is not None or w2 is not None:
            raise ValueError('Incomplete GELU correction.')
        for name, value in (('weight', weight), ('bias', bias), ('w1', w1), ('b1', b1), ('w2', w2)):
            if value is not None and (not value.is_floating_point() or not torch.isfinite(value).all()
                    or value.dtype != weight.dtype or value.device != weight.device):
                raise ValueError('Expected finite buffers with one dtype and device.')
            self.register_buffer(name, None if value is None else value.detach().clone())

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.weight.shape[0] or not torch.isfinite(features).all():
            raise ValueError('Expected finite features of the compiled width.')
        x = features.to(device=self.weight.device, dtype=self.weight.dtype)
        z = x @ self.weight + self.bias
        return z if self.w1 is None else z + F.gelu(x @ self.w1 + self.b1) @ self.w2

    def logits(self, features):
        return self(features)

    def predict(self, features):
        return self(features).argmax(-1)

    def cost(self):
        d = self.weight.shape[0]; h = 0 if self.w1 is None else self.w1.shape[1]
        return {'stored_scalars': sum(t.numel() for t in self.buffers()),
            'stored_bytes': sum(t.numel()*t.element_size() for t in self.buffers()),
            'matrix_multiply_accumulates_per_case': d*256 + (d*h+h*256 if h else 0),
            'matrix_multiplications': 3 if h else 1, 'gelu_elements_per_case': h}


def affine_parts(decoder):
    d = len(decoder.mean)
    if (decoder.mean.shape != (d,) or decoder.scale.shape != (d,) or
            decoder.weight.shape != (d+1, 256) or not (decoder.scale > 0).all()
            or any(not torch.isfinite(t).all() for t in (decoder.mean, decoder.scale, decoder.weight))):
        raise ValueError('Invalid source decoder.')
    weight = decoder.weight[:-1]/decoder.scale[:, None]
    return weight, decoder.weight[-1]-decoder.mean @ weight


def compile_readout(head):
    """Return an independent float64 module, folding all affine work offline.

Identity residuals collapse into the base. GELU keeps its nonlinearity but
absorbs normalization, strength, and the output bias into its frozen buffers.
Floating-point reassociation may change last bits; audit predictions on the
intended inputs when exporting a head.
"""
    if isinstance(head, ByteDecoder):
        return CompiledReadout(*affine_parts(head)).eval()
    if not all(hasattr(head, n) for n in ('base', 'correction', 'strength')):
        raise TypeError('Expected a byte decoder or a scaled frozen correction.')
    if not math.isfinite(head.strength) or not 0 <= head.strength <= 1:
        raise ValueError('Correction strength must be in [0,1].')
    weight, bias = affine_parts(head.base)
    if head.strength == 0:
        return CompiledReadout(weight, bias).eval()
    c = head.correction; strength = head.strength
    if isinstance(c, ByteDecoder):
        dw, db = affine_parts(c)
        return CompiledReadout(weight+strength*dw, bias+strength*db).eval()
    if not isinstance(c, EmissionAdapter) or c.activation not in ('identity', 'gelu'):
        raise TypeError('Unsupported correction.')
    if not (c.scale > 0).all():
        raise ValueError('Invalid correction normalization.')
    w1 = c.w1/c.scale[:, None]; b1 = c.b1-c.mean @ w1
    w2 = strength*c.w2; bias = bias+strength*c.b2
    if c.activation == 'identity':
        return CompiledReadout(weight+w1 @ w2, bias+b1 @ w2).eval()
    return CompiledReadout(weight, bias, w1=w1, b1=b1, w2=w2).eval()


def restore_compiled(record):
    """Restore a weights_only-safe state dictionary, including architecture."""
    if set(record) not in ({'weight', 'bias'}, {'weight', 'bias', 'w1', 'b1', 'w2'}):
        raise ValueError('Invalid compiled buffer inventory.')
    return CompiledReadout(**record).eval()
