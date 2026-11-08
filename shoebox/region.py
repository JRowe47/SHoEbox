"""The same ordinary framed HAT + sparse SHoE primitive for every region."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.hat_minkron.hat import SymbolLevel
from experiments.hat_minkron.model import RecurrentChild
from experiments.hat_minkron.semantic_core import ExpertBank
from .config import SOURCE_IDS
from .transport import byte_frame
from .runtime import symbol_single, execute_private, cached, pilot, lace_forward


@dataclass
class RegionState:
    hat: tuple
    blocks: tuple
    clock: int = 0
    bytes_seen: int = 0


class FramedHAT(nn.Module):
    """Source-tagged, ordered bytes enter one joint local symbol sequence.

    Frame index, byte offset, port width and original producer survive until
    the HAT has interacted with them. No averaging precedes sequence learning.
    The ordinary minKron state spans all arriving sources in recipe order.
    """
    def __init__(self, cfg, max_port_bytes=4, terminal=False):
        super().__init__()
        self.cfg = cfg
        self.terminal = terminal
        self.optimized = False
        self.fused = False
        self.byte = nn.Embedding(256, cfg.width)
        self.source = nn.Embedding(len(SOURCE_IDS), cfg.width)
        self.origin = nn.Embedding(len(SOURCE_IDS), cfg.width)
        self.offset = nn.Embedding(max_port_bytes, cfg.width)
        self.frame_index = nn.Embedding(24, cfg.width)
        self.port_width = nn.Embedding(max_port_bytes+1, cfg.width)
        self.binding = nn.Linear(5*cfg.width, cfg.width, bias=False)
        self.levels = nn.ModuleList(SymbolLevel(cfg, safe_zero=True) for _ in range(cfg.hat_levels))
        self.norm = nn.LayerNorm(cfg.width)
        for item in (self.byte, self.source, self.origin, self.offset, self.frame_index, self.port_width, self.binding):
            nn.init.normal_(item.weight, std=.02)

    def forward(self, frames, states):
        if not 1 <= len(frames) <= self.frame_index.num_embeddings:
            raise ValueError('Expected one to 24 complete input frames.')
        parts = []
        if self.optimized:
            value_w,source_w,origin_w,position_w,boundary_w = self.binding.weight.split(self.cfg.width,-1)
            # Source/position metadata is constant across batch rows. Project
            # its small tables once per invocation, before broadcasting.
            sources,origins,positions,boundaries,widths = cached(self,'metadata',lambda:(
                F.linear(self.source.weight,source_w),F.linear(self.origin.weight,origin_w),
                F.linear(self.offset.weight,position_w),F.linear(self.frame_index.weight,boundary_w),
                F.linear(self.port_width.weight,boundary_w)))
            payloads,metadata = [],[]
        for number, frame in enumerate(frames):
            if frame.source not in SOURCE_IDS or not 1 <= frame.width <= self.offset.num_embeddings:
                raise ValueError('Unknown source or invalid frame width.')
            values = (self.byte(frame.values) if frame.distribution is None
                      else frame.distribution @ self.byte.weight)
            if frame.strength is not None:
                values = values*frame.strength[:, None, None]
            b, t, _ = values.shape
            if self.optimized:
                payloads.append(values)
                metadata.append(sources[SOURCE_IDS[frame.source]]+
                                origins[SOURCE_IDS[frame.origin_source or frame.source]]+
                                positions[:t]+boundaries[number]+widths[t])
                continue
            source = self.source.weight[SOURCE_IDS[frame.source]].expand(b, t, -1)
            origin = self.origin.weight[SOURCE_IDS[frame.origin_source or frame.source]].expand(b, t, -1)
            position = self.offset.weight[:t].expand(b, -1, -1)
            boundary = (self.frame_index.weight[number] + self.port_width.weight[t]).expand(b, t, -1)
            parts.append(values + self.binding(torch.cat((values, source, origin, position, boundary), -1)))
        if self.optimized:
            values = torch.cat(payloads,1)
            x = values+F.linear(values,value_w)+torch.cat(metadata,0).unsqueeze(0)
        else:
            x = torch.cat(parts, 1)
        links = None if self.optimized or self.terminal else torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        final, aux = [], x.new_zeros(())
        for index,(layer, previous) in enumerate(zip(self.levels, states)):
            terminal = self.terminal and index == len(self.levels)-1
            if self.optimized or self.terminal:
                x,updated,balance = symbol_single(layer,x,previous,terminal=terminal,optimized=self.optimized,fused=self.fused)
            else:
                x,states_out,record = layer(x,links,{0:previous},backend='chunk')
                updated,balance = states_out[0],record['balance']
            final.append(updated)
            aux = aux + balance/len(self.levels)
        return self.norm(x[:, -1:]), tuple(final), aux


class CorticalRegion(nn.Module):
    def __init__(self, cfg, index, widths, temporal_scale=1., input_scale=1.):
        super().__init__()
        hc = cfg.hat_config(index)
        self.hat = FramedHAT(hc,cfg.max_port_bytes,cfg.terminal_hat)
        self.optimized = False
        self.fused = False
        c = hc.child_config()
        # Private banks by default: sharing weights never accidentally shares state.
        empty_banks = [ExpertBank(c, 0) for _ in range(3)]
        self.core = RecurrentChild(c, hc, empty_banks, with_decoder=False)
        for block in self.core.blocks:
            block.memory.safe_zero = True
        # Wider transport presents more internal byte steps per observation.
        # Compensate that initial HAT clock expansion; rates remain learnable.
        if input_scale != 1:
            with torch.no_grad():
                for level in self.hat.levels:
                    level.memory.A_log.sub_(math.log(input_scale))
        # B1 has a broader initial time constant. All rates remain learned.
        if temporal_scale != 1:
            with torch.no_grad():
                for level in self.hat.levels:
                    level.memory.A_log.sub_(math.log(temporal_scale))
                for block in self.core.blocks:
                    block.memory.A_log.sub_(math.log(temporal_scale))
        self.outputs = nn.ModuleDict({str(width): nn.Linear(cfg.width, width*256, bias=False)
                                      for width in sorted(widths)})
        for head in self.outputs.values():
            nn.init.normal_(head.weight, std=.02)

    def initial_state(self, batch, device, dtype):
        return RegionState(tuple(level.memory.initial_state(batch, device, dtype) for level in self.hat.levels),
                           tuple(block.memory.initial_state(batch, device, dtype) for block in self.core.blocks))

    def forward(self, name, frames, state, event, revision, observe=False):
        x, hat_state, hat_aux = self.hat(frames, state.hat)
        evidence = torch.stack([pilot(block.router,x) if self.optimized else block.router.pilot(x)
                                for block in self.core.blocks], -2)
        prior, lace = (lace_forward(self.core.lace,evidence,x) if self.optimized else self.core.lace(evidence,x))
        output = (execute_private(self.core,x,prior,state.blocks,fused=self.fused) if self.optimized else
                  self.core.execute(x, prior, state.blocks, decode=False, backend='step'))
        latent = output['latent'][:, 0]
        logits = {int(width): head(latent).reshape(latent.shape[0], int(width), 256)
                  for width, head in self.outputs.items()}
        probabilities = {width:value.softmax(-1) for width,value in logits.items()} if observe else {}
        frames_out = {width: byte_frame(name, event, revision, value,probabilities.get(width)) for width, value in logits.items()}
        return {'frames': frames_out, 'logits': logits, 'probabilities':probabilities,'latent': latent,
                'state': RegionState(hat_state, output['state'], state.clock+1,
                                     state.bytes_seen+sum(f.width for f in frames)),
                'aux': hat_aux+output['aux']+.1*lace['balance'],
                'experts': output['expert_ids'][:, 0].detach()}
