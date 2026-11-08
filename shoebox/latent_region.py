"""Continuous, source-framed occurrences for the ordinary HAT-SHoE primitive."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from experiments.hat_minkron.hat import SymbolLevel
from experiments.hat_minkron.model import RecurrentChild
from experiments.hat_minkron.semantic_core import ExpertBank
from .region import RegionState
from .runtime import symbol_single, execute_private, pilot, lace_forward


SOURCES = ('external', 'A1', 'B1', 'L2', 'S2', 'L3', 'joint', 'Y')
SOURCE_IDS = {name: i for i, name in enumerate(SOURCES)}


@dataclass(frozen=True)
class LatentFrame:
    source: str
    value: torch.Tensor  # [B,D], one occurrence, never D recurrent steps
    start: int
    end: int
    occurrence: int

    def trace(self, event):
        return dict(source=self.source, start=self.start, end=self.end,
                    occurrence=self.occurrence, age=event-self.end)


class LatentHAT(nn.Module):
    def __init__(self, cfg, message_width):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(message_width, cfg.width, bias=False)
        self.source = nn.Embedding(len(SOURCES), cfg.width)
        self.position = nn.Embedding(16, cfg.width)
        self.span = nn.Linear(3, cfg.width, bias=False)
        self.context = nn.Linear(message_width+2, cfg.width, bias=False)
        self.level = SymbolLevel(cfg, safe_zero=True)
        self.norm = nn.LayerNorm(cfg.width)
        nn.init.normal_(self.source.weight, std=.02)
        nn.init.normal_(self.position.weight, std=.02)

    def encode(self, frames, event, context=None):
        if not frames or len(frames) > self.position.num_embeddings:
            raise ValueError('Expected one to 16 complete occurrences.')
        if any(f.source not in SOURCE_IDS or not 0 <= f.start <= f.end <= event for f in frames):
            raise ValueError('Frames must identify a source and a completed causal span.')
        if any(a.end > b.end for a, b in zip(frames, frames[1:])):
            raise ValueError('Occurrences must be in chronological order.')
        values = torch.stack([f.value for f in frames], 1)
        ids = torch.tensor([SOURCE_IDS[f.source] for f in frames], device=values.device)
        # Relative start/end and span length encode unbounded age without an
        # absolute-position table. Raw endpoints/occurrence IDs remain in state.
        timing = values.new_tensor([[math.log1p(event-f.start), math.log1p(event-f.end),
                                    math.log1p(f.end-f.start+1)] for f in frames])/math.log(129)
        x = self.input(values)+self.source(ids)+self.position.weight[:len(frames)]+self.span(timing)
        if context is not None:
            if context.end >= frames[0].start:
                raise ValueError('Top-down context must be committed before this input span.')
            age = math.log1p(event-context.end)/math.log(129)
            metadata = values.new_tensor([age, 1.]).expand(values.shape[0], -1)
            # Context conditions each NEW occurrence. It is not replayed as a
            # fictitious observation and does not add recurrent microsteps.
            x = x+self.context(torch.cat((context.value, metadata), -1))[:, None]
        return x

    def forward(self, frames, previous, event, context=None, fused=True):
        x = self.encode(frames, event, context)
        x, state, aux = symbol_single(self.level, x, previous[0], terminal=True,
                                      optimized=True, fused=fused)
        return self.norm(x), (state,), aux


class LatentRegion(nn.Module):
    """Private HAT and three sparse recurrent expert layers; no byte-code port."""
    def __init__(self, cfg, name):
        super().__init__()
        hc = cfg.hat_config(name)
        self.hat = LatentHAT(hc, cfg.message_width)
        c = hc.child_config()
        self.core = RecurrentChild(c, hc, [ExpertBank(c, 0) for _ in range(3)], with_decoder=False)
        for block in self.core.blocks:
            block.memory.safe_zero = True
        if name == 'B1':
            with torch.no_grad():
                for memory in (self.hat.level.memory, *(b.memory for b in self.core.blocks)):
                    memory.A_log.sub_(math.log(4))
        self.publish = nn.Linear(hc.width, cfg.message_width, bias=False)

    def initial_state(self, batch, device, dtype):
        return RegionState((self.hat.level.memory.initial_state(batch, device, dtype),),
                           tuple(b.memory.initial_state(batch, device, dtype) for b in self.core.blocks))

    def forward(self, frames, state, event, *, context=None, fused=True):
        x, hat, hat_aux = self.hat(frames, state.hat, event, context, fused)
        evidence = torch.stack([pilot(b.router, x) for b in self.core.blocks], -2)
        prior, lace = lace_forward(self.core.lace, evidence, x)
        result = execute_private(self.core, x, prior, state.blocks, fused=fused)
        latent = result['latent'][:, 0]
        return dict(value=self.publish(latent), latent=latent,
                    state=RegionState(hat, result['state'], state.clock+1, state.bytes_seen+len(frames)),
                    aux=hat_aux+result['aux']+.1*lace['balance'],
                    experts=result['expert_ids'][:, 0].detach())
