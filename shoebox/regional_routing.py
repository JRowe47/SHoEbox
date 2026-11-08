"""Learned hard regional/edge selection over the permitted cortical graph."""
import torch
from torch import nn
from torch.nn import functional as F
from .config import NAMES, SPINE, ROUTING_GROUPS


def compile_waves(specs):
    levels = {}
    for spec in specs:
        levels[spec.name] = 1+max((levels[e.source] for e in spec.inputs if not e.delayed), default=-1)
    return tuple(tuple(s for s in specs if levels[s.name] == level)
                 for level in range(max(levels.values())+1))


def balance(probability, selected):
    return probability.shape[-1]*(probability.mean(0)*selected.float().mean(0)/
                                   selected.sum(-1).float().mean().clamp_min(1)).sum()


class RegionalRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.routing_dim
        size = sum(cfg.for_region(n).width for n in ('A1','B1','Y'))
        self.query = nn.Linear(size, d, bias=False)
        self.region_keys = nn.Parameter(torch.randn(len(ROUTING_GROUPS), d)*.1)
        self.region_bias = nn.Parameter(torch.zeros(len(ROUTING_GROUPS)))
        self.edge_keys = nn.ParameterDict({s.name: nn.Parameter(torch.randn(len(s.inputs), d)*.1)
                                          for s in cfg.topology() if s.inputs})
        self.edge_bias = nn.ParameterDict({s.name: nn.Parameter(torch.zeros(len(s.inputs)))
                                          for s in cfg.topology() if s.inputs})

    def perturbed(self, scores):
        if self.training and self.cfg.routing_noise:
            uniform = torch.rand_like(scores).clamp(1e-6, 1-1e-6)
            return scores-self.cfg.routing_noise*(-uniform.log()).log()
        return scores

    def regions(self, latents, batch):
        a,b = latents['A1'],latents['B1']
        y = latents.get('Y', a.new_zeros(batch,self.cfg.for_region('Y').width))
        query = F.normalize(F.silu(self.query(torch.cat((a,b,y),-1))),dim=-1)
        scores = 4*F.linear(query,F.normalize(self.region_keys,dim=-1))+self.region_bias
        probabilities = scores.softmax(-1)
        ids = self.perturbed(scores).topk(self.cfg.region_topk,-1).indices
        chosen = torch.zeros_like(scores,dtype=torch.bool).scatter(1,ids,True)
        active = torch.zeros(batch,len(NAMES),dtype=torch.bool,device=a.device)
        gates = a.new_ones(batch,len(NAMES))
        for name in SPINE:
            active[:,NAMES.index(name)] = True
        for index,names in enumerate(ROUTING_GROUPS):
            for name in names:
                active[:,NAMES.index(name)] = chosen[:,index]
                gates[:,NAMES.index(name)] = 1+(probabilities[:,index]-probabilities[:,index].detach())
        return query, active, gates, balance(probabilities,chosen), probabilities

    def edges(self, spec, query, available):
        scores = 4*F.linear(query,F.normalize(self.edge_keys[spec.name],dim=-1))+self.edge_bias[spec.name]
        # A finite floor keeps the all-unavailable cold-start row well defined.
        probabilities = scores.masked_fill(~available,-1e9).softmax(-1)
        ranked = self.perturbed(scores).masked_fill(~available,-torch.inf)
        chosen = torch.zeros_like(available)
        current = torch.tensor([not e.delayed for e in spec.inputs],device=query.device)
        required = available & current[None]
        if spec.name == 'Y':
            required = available & torch.tensor([e.source=='S2' for e in spec.inputs],device=query.device)[None]
        first = ranked.masked_fill(~required,-torch.inf).argmax(-1,keepdim=True)
        chosen.scatter_(1,first,required.any(-1,keepdim=True))
        # Fill the remaining slots, retaining a current source when one exists.
        for _ in range(self.cfg.edge_topk):
            need = chosen.sum(-1) < self.cfg.edge_topk
            candidate = ranked.masked_fill(chosen,-torch.inf).argmax(-1,keepdim=True)
            valid = available.gather(1,candidate) & ~chosen.gather(1,candidate) & need[:,None]
            chosen.scatter_(1,candidate,chosen.gather(1,candidate)|valid)
        gates = 1+(probabilities-probabilities.detach())
        return chosen, gates, balance(probabilities,chosen)
