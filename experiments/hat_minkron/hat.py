"""Hierarchical byte symbol construction with shared, source-indexed recurrence."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from kron import MinKron
from semantic_core import switch_balance


@dataclass
class HATState:
    batch_size: int
    levels: tuple
    clocks: dict

    def detach(self):
        return HATState(self.batch_size,
                        tuple({k:v.detach() for k,v in layer.items()} for layer in self.levels),
                        {k:v.detach() for k,v in self.clocks.items()})

    def reset(self, rows):
        def clear(v):
            mask=(~rows).reshape(-1,*([1]*(v.ndim-1)))
            return v*mask
        return HATState(self.batch_size,
                        tuple({k:clear(v) for k,v in layer.items()} for layer in self.levels),
                        {k:clear(v) for k,v in self.clocks.items()})

    def tensors(self):
        return [v for layer in self.levels for v in layer.values()]+list(self.clocks.values())


class SymbolLevel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.norm=nn.LayerNorm(cfg.width)
        self.memory=MinKron(cfg.width,cfg.kron_heads,cfg.kron_key,cfg.kron_value,cfg.chunk_size)
        self.query=nn.Linear(cfg.width,cfg.address_dim,bias=False)
        self.keys=nn.Parameter(torch.randn(cfg.hat_symbols,cfg.address_dim))
        self.values=nn.Parameter(torch.randn(cfg.hat_symbols,cfg.width)*.02)

    def forward(self,x,links,states,backend='chunk',memory_enabled=True):
        b,t,d=x.shape
        context=torch.zeros_like(x)
        new_states=dict(states)
        z=self.norm(x)
        # Only arriving links run a memory update. State coordinates never depend
        # on routing rank, packet scheduling, or a transient top-k position.
        for link in links.unique().tolist():
            mask=links==link
            previous=states.get(link)
            if previous is None:
                previous=self.memory.initial_state(b,x.device,x.dtype)
            if previous.shape[0] != b: raise ValueError('State batch size mismatch.')
            if bool((mask==mask[:1]).all()):
                indices=mask[0].nonzero().flatten()
                values,final=self.memory(z[:,indices],previous,backend,not memory_enabled)
                context=context.index_copy(1,indices,values)
            else:
                final=previous
                for row in range(b):
                    indices=mask[row].nonzero().flatten()
                    if indices.numel()==0: continue
                    values,end=self.memory(z[row:row+1,indices],previous[row:row+1],backend,not memory_enabled)
                    flat_ids=row*t+indices
                    context=context.reshape(b*t,d).index_copy(0,flat_ids,values[0]).reshape(b,t,d)
                    final=final.index_copy(0,torch.tensor([row],device=x.device),end)
            new_states[link]=final
        h=x+context
        q=F.normalize(self.query(self.norm(h)),dim=-1)
        scores=4*F.linear(q,F.normalize(self.keys,dim=-1))
        probabilities=scores.softmax(-1)
        ids=scores.topk(self.cfg.hat_active_symbols,-1).indices
        # Normalize over all candidates before gathering; unchosen scores retain
        # gradients through the denominator, as in the SHoE expert router.
        weights=probabilities.gather(-1,ids)*self.cfg.hat_symbols/self.cfg.hat_active_symbols
        symbol=(self.values[ids]*weights[...,None]).sum(-2)/math.sqrt(self.cfg.hat_active_symbols)
        return h+symbol,new_states,{'ids':ids,'weights':weights,
                                    'balance':switch_balance(probabilities,ids)}


class HAT(nn.Module):
    """One byte alphabet, one binding encoder, shared learned symbol dictionaries.

    v1 emits one contextual symbol occurrence per byte. The hierarchy composes
    representations; it does not yet merge events or change their sequence rate.
    Literal bytes and exact provenance remain available beside sparse symbols.
    """
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.byte_embedding=nn.Embedding(256,cfg.width)
        self.link_embedding=nn.Embedding(cfg.max_links,cfg.width)
        self.slot_embedding=nn.Embedding(cfg.max_slots,cfg.width)
        self.binding=nn.Linear(3*cfg.width,cfg.width,bias=False)
        self.levels=nn.ModuleList([SymbolLevel(cfg) for _ in range(cfg.hat_levels)])
        self.final_norm=nn.LayerNorm(cfg.width)
        for m in (self.byte_embedding,self.link_embedding,self.slot_embedding,self.binding):
            nn.init.normal_(m.weight,std=.02)

    def initial_state(self,batch):
        return HATState(batch,tuple({} for _ in self.levels),{})

    def forward(self,events,state=None,backend='chunk',memory_enabled=True,trace=False):
        events.validate(self.cfg.max_links,self.cfg.max_slots)
        b,t=events.values.shape
        state=state or self.initial_state(b)
        if state.batch_size!=b or len(state.levels)!=len(self.levels):
            raise ValueError('HAT state schema/batch does not match the model.')
        value=self.byte_embedding(events.values)
        address=torch.cat((value,self.link_embedding(events.links),self.slot_embedding(events.slots)),-1)
        x=value+self.binding(address)
        levels,records=[],[]
        aux=x.new_zeros(())
        for layer,previous in zip(self.levels,state.levels):
            x,new,record=layer(x,events.links,previous,backend,memory_enabled)
            levels.append(new);records.append(record)
            aux=aux+record['balance']/len(self.levels)
        clocks=dict(state.clocks)
        positions=torch.zeros_like(events.values)
        for link in events.links.unique().tolist():
            mask=events.links==link
            old=clocks.get(link,torch.zeros(b,dtype=torch.long,device=x.device))
            local=old[:,None]+mask.long().cumsum(1)-1
            positions=torch.where(mask,local,positions)
            clocks[link]=old+mask.long().sum(1)
        return {'payload':self.final_norm(x),'state':HATState(b,tuple(levels),clocks),'aux':aux,
                'symbol_ids':torch.stack([r['ids'] for r in records],-2),
                'symbol_weights':torch.stack([r['weights'] for r in records],-2),
                'links':events.links,'slots':events.slots,'source_positions':positions,
                'literal_values':events.values}
