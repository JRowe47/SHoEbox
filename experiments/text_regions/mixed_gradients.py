"""Preserve ordinary clipping, recovering finite FP32 norm overflow in FP64."""
import math
import torch


def clip_mixed_gradients(parameters, maximum=1.):
    parameters = list(parameters)
    try:
        # The finite path is exactly the preceding trainer's arithmetic. Reject
        # before mutation so an overflow cannot zero or poison finite gradients.
        return torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True), False
    except RuntimeError:
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Nonfinite mixed training gradient entries.')
        norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
        if not math.isfinite(norm):
            raise RuntimeError('Nonfinite mixed training gradient norm in FP64.')
        scale = min(1., maximum/(norm+1e-6))
        with torch.no_grad():
            for gradient in gradients:
                gradient.mul_(scale)
        return torch.tensor(norm, dtype=torch.float64), True
