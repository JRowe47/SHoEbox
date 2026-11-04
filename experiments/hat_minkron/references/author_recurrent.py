# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""PyTorch references for KronLSTM: the token-by-token recurrence and its chunkwise form.

`README.md` in this directory states the recurrence, its variants and the chunkwise derivation.
"""

import torch
import torch.nn.functional as F


# floor on ||v||^2 in the division of `beta_max = s / ||v||^2`:
# below it a pair is stored at the fraction ||v||^2 / V_NORM_FLOOR of what `beta` asks for,
# and the cap `s / V_NORM_FLOOR` stays far inside the fp32 range; the Triton kernels hard-code the same value
V_NORM_FLOOR = 1e-18


def _gate_rank(g, K: int) -> int:
    """Return 0 for no gate, 3 for a per-head gate `[B, T, H]` and 4 for a per-key-dimension gate `[B, T, H, K]`.

    A rank-4 gate whose last axis is not `K` is refused rather than broadcast:
    a per-head gate with a stray trailing axis, `[B, T, H, 1]`, would otherwise run the per-key-dimension path.
    """
    if g is None:
        return 0
    if g.dim() == 3:
        return 3
    if g.dim() != 4:
        raise ValueError(
            f"`g` must be [B, T, H] (per-head scalar decay) or [B, T, H, K] (per-key-dim "
            f"decay), got rank {g.dim()} of shape {tuple(g.shape)}.",
        )
    if g.shape[-1] != K:
        raise ValueError(
            f"a per-key-dim decay gate needs one entry per key dimension: expected "
            f"`g.shape[-1] == {K}`, got {g.shape[-1]} (full shape {tuple(g.shape)}). A per-head "
            f"scalar gate belongs in [B, T, H] form, without a trailing axis.",
        )
    return 4


def naive_recurrent_kronlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    allow_neg_eigval: bool = False,
    target_gate: torch.Tensor | None = None,
    decouple_erase: bool = True,
):
    """Token-by-token KronLSTM, differentiable by autograd; the ground truth of the other implementations.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`, assumed to be unit-norm.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            natural-log decay, `[B, T, H]` (one rate per head) or `[B, T, H, K]` (one rate per key dimension),
            or `None` for no decay.
        beta (torch.Tensor):
            erase gate (the step size of the Kronecker delta rule) of shape `[B, T, H]`, in `[0, 1]`.
        scale (float, Optional):
            read-out scale. Default: `K ** -0.5`.
        initial_state (torch.Tensor, Optional):
            initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            whether to return the final state. Default: `False`.
        allow_neg_eigval (bool, Optional):
            whether `beta* = s * beta / ||v||^2` may reach the negative-eigenvalue range, `s = 2`; else `s = 1`.
            Default: `False`.
        target_gate (torch.Tensor, Optional):
            target gate `c` of shape `[B, T, H]`, after a sigmoid; the write target becomes `c * ||v||^2`.
            Default: `None`.
        decouple_erase (bool, Optional):
            whether `beta` gates the erase only: the write keeps the cap `s / ||v||^2`,
            so `target_gate` alone gates it. Requires `target_gate`. Default: `True`.

    Returns:
        o (torch.Tensor):
            outputs of shape `[B, T, H, V]`, in `q`'s dtype.
        final_state (torch.Tensor):
            final state of shape `[B, H, K, V]` in the accumulation dtype if `output_final_state`, else `None`.
    """
    if decouple_erase and target_gate is None:
        raise ValueError("`decouple_erase=True` requires `target_gate`.")
    beta_max_scaler = 2.0 if allow_neg_eigval else 1.0
    orig_dtype = q.dtype
    # accumulate in fp32, or in float64 for float64 inputs
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    g_rank = _gate_rank(g, k.shape[-1])
    q, k, v, beta = (x.transpose(1, 2).contiguous().to(acc) for x in (q, k, v, beta))
    target_gate = target_gate.transpose(1, 2).contiguous().to(acc) if target_gate is not None else None
    if g is not None:
        g = g.transpose(1, 2).contiguous().to(acc)
    B, H, T, K, V = *k.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    o = torch.zeros(B, H, T, V, dtype=acc, device=v.device)
    S = torch.zeros(B, H, K, V, dtype=acc, device=v.device)
    if initial_state is not None:
        S = S + initial_state.to(acc)

    # the floor applies to the division only; the target below uses the unclamped ||v||^2
    v_norm = v.norm(dim=-1) ** 2
    beta_max = beta_max_scaler / v_norm.clamp_min(V_NORM_FLOOR)
    beta_star = beta_max * beta

    for t in range(T):
        b_q, b_k, b_v = q[:, :, t], k[:, :, t], v[:, :, t]
        if g is not None:
            # rank 4 decays each state row (key dimension) at its own rate, rank 3 the whole state at one
            b_g = g[:, :, t].exp()
            S = S * (b_g.unsqueeze(-1) if g_rank == 4 else b_g[..., None, None])
        # y = k^T S v, a full contraction over K and V
        b_y = torch.einsum('bhk,bhkv,bhv->bh', b_k, S, b_v)
        # u = w * target - beta* y, with w = beta_max under `decouple_erase` and beta* otherwise;
        # kept factored as beta* (target - y) without `decouple_erase`,
        # so that `target_gate == 1` reproduces the unit-target write exactly
        if decouple_erase:
            b_u = (beta_max[:, :, t] * (v_norm[:, :, t] * target_gate[:, :, t])
                   - beta_star[:, :, t] * b_y)
        elif target_gate is not None:
            b_u = beta_star[:, :, t] * (v_norm[:, :, t] * target_gate[:, :, t] - b_y)
        else:
            b_u = beta_star[:, :, t] * (v_norm[:, :, t] - b_y)
        S = S + b_u[..., None, None] * b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, t] = torch.einsum('bhk,bhkv->bhv', b_q * scale, S)

    o = o.transpose(1, 2).contiguous().to(orig_dtype)
    return o, (S if output_final_state else None)

