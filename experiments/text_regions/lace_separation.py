"""Query-conditioned lace bias and bounded first-order, head-local credit.

Capture is opt-in optimizer bookkeeping. No targets enter a model forward, no
examples persist, and only the final refinement's task-connected scores count.
Keep these controls ablatable: specialization may need more capacity/exposure.
"""
from dataclasses import dataclass
import math
import time
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LaceSeparationConfig:
    installed: bool = False
    enabled: bool = False
    rank: int = 4
    cap: float = .5
    alpha: float = .10
    beta: float = .05
    max_extra_ratio: float = .25
    margin: float = 1.

    def validate(self):
        if type(self.installed) is not bool or type(self.enabled) is not bool:
            raise ValueError('Lace switches must be boolean.')
        if self.enabled and not self.installed:
            raise ValueError('Enabled lace bias requires installed heads.')
        if type(self.rank) is not int or self.rank < 1 or not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError('Positive rank and finite cap required.')
        if any(not math.isfinite(x) or x < 0 for x in
               (self.alpha, self.beta, self.max_extra_ratio, self.margin)):
            raise ValueError('Credit coefficients must be finite and nonnegative.')


class LaceBiasHead(nn.Module):
    def __init__(self, features, layers, laces, config, seed=0):
        super().__init__()
        self.config, self.layers, self.laces = config, layers, laces
        # Private generator: installing even a bypassed head leaves RNG intact.
        generator = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(.02*torch.randn(config.rank, features, generator=generator))
        self.V = nn.Parameter(torch.zeros(layers*laces, config.rank))

    def forward(self, z, capture=False):
        h = F.linear(z, self.U)
        bias = self.config.cap*(F.linear(h, self.V)/self.config.cap).tanh()
        cache = (z.detach(), h.detach(), bias.detach(), self.V.detach().clone()) if capture else None
        return bias.reshape(-1, self.layers, self.laces), cache


def local_derivative(v, z, h, forward_v):
    """Exact local derivative with z fixed; no second backward or graph."""
    return (v @ forward_v).T @ z, v.T @ h


def extra_score_gradient(ids, gradients, eligible, alpha, beta):
    """Inputs [pairs,2,layers,k]; only aligned controller/layer IDs compare."""
    a = gradients.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    usable = torch.isfinite(a) & (a > 0)
    g = torch.nan_to_num(gradients/(a+torch.finfo(gradients.dtype).eps)).clamp(-2, 2)
    matches = ids[:, 0, :, :, None].eq(ids[:, 1, :, None, :])
    shared = torch.stack((matches.any(-1), matches.any(-2)), 1)
    opposing = matches & (g[:, 0, :, :, None]*g[:, 1, :, None, :] < -.01)
    conflict = torch.stack((opposing.any(-1), opposing.any(-2)), 1)
    extra = torch.nan_to_num(a)*(-alpha*(~shared)*F.relu(-g) + beta*shared*conflict*F.relu(g))
    extra = torch.where(usable & eligible[:, None, None, None], extra, 0.)
    return extra, shared, conflict


class LaceCredit:
    def __init__(self, event, config):
        self.event, self.config = event, config
        self.records = {}
        self.eligible = self.targets = None

    def supervise(self, logits, targets):
        """Training-only gate; targets never cross the forward interface."""
        self.targets = targets.detach().reshape(-1, 2)
        correct = logits.detach().gather(1, targets[:, None]).flatten()
        other = logits.detach().gather(1, targets.reshape(-1, 2).flip(1).reshape(-1, 1)).flatten()
        self.eligible = self.targets[:, 0].ne(self.targets[:, 1]) & ((correct-other).reshape(-1, 2).amin(-1) < self.config.margin)

    def record(self, key, head, rows, valid, lace, cache):
        scores = lace['selected_scores']
        if not scores.requires_grad:
            return
        if key in self.records:
            raise RuntimeError('Duplicate answer decision capture (recompute registered twice).')
        record = dict(head=head, rows=rows.detach(), valid=valid.detach(),
                      ids=lace['lace_ids'][:, 0].detach(),
                      weights=lace['lace_weights'][:, 0].detach(), cache=cache)
        # selected_scores bypasses the direct balance branch on external scores.
        scores.register_hook(lambda g: record.update(gradient=g.detach()))
        self.records[key] = record

    @torch.no_grad()
    def apply(self):
        started = time.perf_counter()
        if self.eligible is None:
            raise RuntimeError('Missing paired supervision gate.')
        additions, heads = {}, {}
        stats = dict(records=len(self.records), pair_decisions=0, eligible_decisions=0,
                     updated_decisions=0, opposing_laces=0, helpful_unique=0,
                     shared_laces=0, selected_laces=0, same_target_decisions=0,
                     same_target_opposing=0, same_target_updated=0)
        samples = {k: [] for k in ('hard_overlap','soft_l1','shared_mass','unique_mass',
                                   'input_rms_difference','hidden_rms_difference','bias_rms','bias_saturation')}
        for record in self.records.values():
            if 'gradient' not in record:
                continue
            g = record['gradient']
            if g.dtype != torch.float32:
                raise ValueError('Extra credit currently requires verified unscaled FP32 gradients.')
            valid_rows = {int(row): i for i, row in enumerate(record['rows']) if record['valid'][i]}
            pair_ids = [i for i in range(len(self.eligible)) if 2*i in valid_rows and 2*i+1 in valid_rows]
            if not pair_ids:
                continue
            ix = torch.tensor([[valid_rows[2*i],valid_rows[2*i+1]] for i in pair_ids],device=g.device)
            eligible = self.eligible[pair_ids]
            same = self.targets[pair_ids, 0].eq(self.targets[pair_ids, 1])
            ids, grad = record['ids'][ix], g[ix]
            extra, shared, conflict = extra_score_gradient(ids, grad, eligible, self.config.alpha, self.config.beta)
            updated = extra.ne(0).flatten(1).any(-1)
            stats['pair_decisions'] += len(pair_ids)
            stats['eligible_decisions'] += int(eligible.sum())
            stats['updated_decisions'] += int(updated.sum())
            stats['opposing_laces'] += int(conflict[~same].sum())
            stats['helpful_unique'] += int(((~shared)&(grad<0))[~same].sum())
            stats['shared_laces'] += int(shared.sum()); stats['selected_laces'] += shared.numel()
            stats['same_target_decisions'] += int(same.sum())
            stats['same_target_opposing'] += int(conflict[same].sum())
            stats['same_target_updated'] += int(updated[same].sum())
            weights = record['weights'][ix]
            samples['hard_overlap'].append(float(shared.float().mean()))
            samples['shared_mass'].append(float((weights*shared).sum(-1).mean()))
            samples['unique_mass'].append(float((weights*(~shared)).sum(-1).mean()))
            # Dense only over tiny lace dictionaries, never over expert banks.
            size = record['head'].laces if record['head'] is not None else int(ids.max())+1
            soft = weights.new_zeros(*ids.shape[:-1],size).scatter(-1,ids,weights)
            samples['soft_l1'].append(float((soft[:,0]-soft[:,1]).abs().sum(-1).mean()))
            if record['cache'] is None:
                continue
            z,h,bias,forward_v = record['cache']; head=record['head']
            samples['input_rms_difference'].append(float((z[ix[:,0]]-z[ix[:,1]]).square().mean().sqrt()))
            samples['hidden_rms_difference'].append(float((h[ix[:,0]]-h[ix[:,1]]).square().mean().sqrt()))
            samples['bias_rms'].append(float(bias.square().mean().sqrt()))
            samples['bias_saturation'].append(float((bias.abs()>=.95*head.config.cap).float().mean()))
            delta = bias.new_zeros(len(z),head.layers,head.laces)
            local = extra.new_zeros(g.shape).index_copy(0,ix.flatten(),extra.flatten(0,1))
            delta.scatter_add_(-1,record['ids'],local)
            v = delta.flatten(1)*(1-(bias/head.config.cap).square())
            dU,dV=local_derivative(v,z,h,forward_v)
            for parameter, addition in ((head.U,dU),(head.V,dV)):
                ident=id(parameter);heads[ident]=parameter
                additions[ident]=additions.get(ident,0.)+addition
        ordinary = math.sqrt(sum(float(p.grad.square().sum()) for p in heads.values() if p.grad is not None))
        extra_norm = math.sqrt(sum(float(x.square().sum()) for x in additions.values()))
        factor = min(1.,self.config.max_extra_ratio*ordinary/extra_norm) if extra_norm and math.isfinite(extra_norm) else 0.
        for ident, addition in additions.items():
            if factor:
                parameter=heads[ident]
                if parameter.grad is None:
                    parameter.grad=torch.zeros_like(parameter)
                parameter.grad.add_(addition,alpha=factor)
        stats.update(ordinary_head_norm=ordinary,uncapped_extra_norm=extra_norm,
                     extra_head_norm=extra_norm*factor if math.isfinite(extra_norm) else 0.,norm_cap_factor=factor)
        stats.update({k:sum(v)/len(v) if v else 0. for k,v in samples.items()})
        stats['seconds']=time.perf_counter()-started
        self.records.clear()
        return stats
