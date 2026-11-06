"""Tensor-only one-event kernels for the regional recurrence."""
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
from .runtime_autograd import kron_step, recurrence_forward


@torch.jit.script
def balance(probabilities: Tensor, choices: Tensor) -> Tensor:
    count = probabilities.shape[-1]
    load = torch.bincount(choices.reshape(-1), minlength=count).to(probabilities.dtype)
    load = load/choices.numel()
    return count*(probabilities.reshape(-1, count).mean(0)*load).sum()


@torch.jit.script
def direction(x: Tensor, safe_zero: bool) -> Tensor:
    norm = x.norm(p=2, dim=-1, keepdim=True)
    value = x/norm.clamp_min(1e-12)
    if safe_zero:
        value = torch.where(norm > 1e-12, value, torch.zeros_like(value))
    return value


def event(x: Tensor, projection: Tensor, forget_weight: Tensor, decay: Tensor,
          dt_bias: Tensor, norm_weight: Tensor, gate_bias: Tensor, output: Tensor,
          state: Optional[Tensor], heads: int, key: int, value: int,
          safe_zero: bool) -> Tuple[Tensor, Tensor]:
    packed = F.linear(x, projection)
    forget = packed[..., heads*(2*key+value)+2*heads:heads*(2*key+value)+2*heads+value]
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    f = F.linear(forget, forget_weight).to(acc)+dt_bias.to(acc)
    z, final = projected_event(packed, f, decay[None], norm_weight.to(acc), gate_bias,
                               state, heads, key, value, safe_zero)
    return F.linear(z, output), final


def projected_event(packed: Tensor, f: Tensor, decay: Tensor, norm_weight: Tensor,
                    gate_bias: Tensor, state: Optional[Tensor], heads: int, key: int,
                    value: int, safe_zero: bool) -> Tuple[Tensor, Tensor]:
    b = packed.shape[0]
    qkv, erase, write, forget, gate = packed.split(
        [heads*(2*key+value), heads, heads, value, heads*value], -1)
    if key == value:
        q, k, v = direction(F.silu(qkv).reshape(b, 3, heads, key), safe_zero).unbind(1)
    else:
        q0, k0, v0 = F.silu(qkv).split([heads*key, heads*key, heads*value], -1)
        q = direction(q0.reshape(b, heads, key), safe_zero)
        k = direction(k0.reshape(b, heads, key), safe_zero)
        v = direction(v0.reshape(b, heads, value), safe_zero)
    acc = torch.float64 if packed.dtype == torch.float64 else torch.float32
    g = -decay[:, :, None]*F.softplus(f).reshape(b, heads, key)
    q, k, v = q.to(acc), k.to(acc), v.to(acc)
    previous = q.new_zeros((b, heads, key, value)) if state is None else state.to(acc)
    beta, writing = erase.sigmoid().to(acc).squeeze(1), write.to(acc).sigmoid().squeeze(1)
    if torch.is_grad_enabled():
        out, final = kron_step(q, k, v, g, beta, writing, previous)
    else:
        out, final, _, _, _, _, _, _ = recurrence_forward(q, k, v, g, beta, writing, previous)
    z = out.to(packed.dtype).unsqueeze(1).to(acc)
    z = z*torch.rsqrt(z.square().mean(-1, keepdim=True)+1e-5)*norm_weight.to(acc)
    z = z*(gate+gate_bias).reshape(b, 1, heads, value).sigmoid()
    return z.reshape(b, 1, heads*value).to(packed.dtype), final


projected_event = torch.jit.script(projected_event)
event = torch.jit.script(event)


# Flat scripted inference avoids the training autograd boundary and dispatch.
@torch.jit.script
def inference_event(x: Tensor, projection: Tensor, forget_weight: Tensor, decay: Tensor,
          dt_bias: Tensor, norm_weight: Tensor, gate_bias: Tensor, output: Tensor,
          state: Optional[Tensor], heads: int, key: int, value: int,
          safe_zero: bool) -> Tuple[Tensor, Tensor]:
    b = x.shape[0]
    qkv, erase, write, forget, gate = F.linear(x, projection).split(
        [heads*(2*key+value), heads, heads, value, heads*value], -1)
    if key == value:
        q, k, v = direction(F.silu(qkv).reshape(b, 3, heads, key), safe_zero).unbind(1)
    else:
        q0, k0, v0 = F.silu(qkv).split([heads*key, heads*key, heads*value], -1)
        q = direction(q0.reshape(b, heads, key), safe_zero)
        k = direction(k0.reshape(b, heads, key), safe_zero)
        v = direction(v0.reshape(b, heads, value), safe_zero)
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    f = F.linear(forget, forget_weight).to(acc)+dt_bias.to(acc)
    g = -decay[None, :, None]*F.softplus(f).reshape(b, heads, key)
    q, k, v = q.to(acc), k.to(acc), v.to(acc)
    previous = q.new_zeros((b, heads, key, value)) if state is None else state.to(acc)
    decayed = previous*g.exp().unsqueeze(-1)
    norm = v.square().sum(-1)
    inv = norm.clamp_min(1e-18).reciprocal()
    score = ((k.unsqueeze(-1)*decayed)*v.unsqueeze(-2)).sum((-2, -1))
    u = write.to(acc).sigmoid().squeeze(1)*norm*inv-erase.sigmoid().to(acc).squeeze(1)*inv*score
    final = decayed+u[..., None, None]*k.unsqueeze(-1)*v.unsqueeze(-2)
    out = (q.unsqueeze(-1)*final).sum(-2)*key**-.5
    z = out.to(x.dtype).unsqueeze(1).to(acc)
    z = z*torch.rsqrt(z.square().mean(-1, keepdim=True)+1e-5)*norm_weight.to(acc)
    z = z*(gate+gate_bias).reshape(b, 1, heads, value).sigmoid()
    return F.linear(z.reshape(b, 1, heads*value).to(x.dtype), output), final


@torch.jit.script
def symbol_read(h: Tensor, norm_weight: Tensor, norm_bias: Tensor, query: Tensor,
                keys: Tensor, values: Tensor, active: int, safe_zero: bool
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    q = direction(F.linear(F.layer_norm(h, [h.shape[-1]], norm_weight, norm_bias), query), safe_zero)
    scores = 4*F.linear(q, keys)
    probabilities = scores.softmax(-1)
    ids = scores.topk(active, -1).indices
    weights = probabilities.gather(-1, ids)*keys.shape[0]/active
    symbol = (values[ids]*weights[..., None]).sum(-2)/float(active)**.5
    return h+symbol, ids, weights, balance(probabilities, ids)


@torch.jit.script
def expert_bank(x: Tensor, ids: Tensor, gates: Tensor, shared_up: Tensor,
                shared_down: Tensor, private_up: Tensor, private_down: Tensor) -> Tensor:
    b, t, d = x.shape
    k = ids.shape[-1]
    shared_count = shared_up.shape[0]
    flat_ids, flat_gates = ids.reshape(-1), gates.reshape(-1)
    flat_x = x.reshape(-1, d)
    out = x.new_zeros((b*t, d))
    for shared, up, down in ((True, shared_up, shared_down), (False, private_up, private_down)):
        selected = ((flat_ids < shared_count) == shared).nonzero().flatten()
        if selected.numel() > 0:
            local = flat_ids[selected] - (0 if shared else shared_count)
            owners = selected.div(k, rounding_mode='floor')
            chosen_up, chosen_down = up.index_select(0, local), down.index_select(0, local)
            values = flat_x.index_select(0, owners)
            hidden = F.gelu((chosen_up*values[:, None]).sum(-1))
            values = (hidden[..., None]*chosen_down).sum(-2)*flat_gates[selected, None]
            out = out.index_add(0, owners, values)
    return out.reshape(b, t, d)/float(k)**.5
