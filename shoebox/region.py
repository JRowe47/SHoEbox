"""The same ordinary framed HAT + sparse SHoE primitive for every region."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from experiments.hat_minkron.hat import SymbolLevel
from experiments.hat_minkron.model import RecurrentChild
from experiments.hat_minkron.semantic_core import ExpertBank
from .config import SOURCE_IDS
from .transport import byte_frame


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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.byte = nn.Embedding(256, cfg.width)
        self.source = nn.Embedding(len(SOURCE_IDS), cfg.width)
        self.origin = nn.Embedding(len(SOURCE_IDS), cfg.width)
        self.offset = nn.Embedding(4, cfg.width)
        self.frame_index = nn.Embedding(24, cfg.width)
        self.port_width = nn.Embedding(5, cfg.width)
        self.binding = nn.Linear(5*cfg.width, cfg.width, bias=False)
        self.levels = nn.ModuleList(SymbolLevel(cfg, safe_zero=True) for _ in range(cfg.hat_levels))
        self.norm = nn.LayerNorm(cfg.width)
        for item in (self.byte, self.source, self.origin, self.offset, self.frame_index, self.port_width, self.binding):
            nn.init.normal_(item.weight, std=.02)

    def forward(self, frames, states):
        if not 1 <= len(frames) <= self.frame_index.num_embeddings:
            raise ValueError('Expected one to 24 complete input frames.')
        parts = []
        for number, frame in enumerate(frames):
            if frame.source not in SOURCE_IDS or not 1 <= frame.width <= 4:
                raise ValueError('Unknown source or invalid frame width.')
            values = (self.byte(frame.values) if frame.distribution is None
                      else frame.distribution @ self.byte.weight)
            if frame.strength is not None:
                values = values*frame.strength[:, None, None]
            b, t, _ = values.shape
            source = self.source.weight[SOURCE_IDS[frame.source]].expand(b, t, -1)
            origin = self.origin.weight[SOURCE_IDS[frame.origin_source or frame.source]].expand(b, t, -1)
            position = self.offset.weight[:t].expand(b, -1, -1)
            boundary = (self.frame_index.weight[number] + self.port_width.weight[t]).expand(b, t, -1)
            parts.append(values + self.binding(torch.cat((values, source, origin, position, boundary), -1)))
        x = torch.cat(parts, 1)
        links = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        final, aux = [], x.new_zeros(())
        for layer, previous in zip(self.levels, states):
            x, updated, record = layer(x, links, {0: previous}, backend='chunk')
            final.append(updated[0])
            aux = aux + record['balance']/len(self.levels)
        return self.norm(x[:, -1:]), tuple(final), aux


class CorticalRegion(nn.Module):
    def __init__(self, cfg, index, widths, temporal_scale=1.):
        super().__init__()
        hc = cfg.hat_config(index)
        self.hat = FramedHAT(hc)
        c = hc.child_config()
        # Private banks by default: sharing weights never accidentally shares state.
        empty_banks = [ExpertBank(c, 0) for _ in range(3)]
        self.core = RecurrentChild(c, hc, empty_banks, with_decoder=False)
        for block in self.core.blocks:
            block.memory.safe_zero = True
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

    def forward(self, name, frames, state, event, revision):
        x, hat_state, hat_aux = self.hat(frames, state.hat)
        evidence = torch.stack([block.router.pilot(x) for block in self.core.blocks], -2)
        prior, lace = self.core.lace(evidence, x)
        output = self.core.execute(x, prior, state.blocks, decode=False, backend='step')
        latent = output['latent'][:, 0]
        logits = {int(width): head(latent).reshape(latent.shape[0], int(width), 256)
                  for width, head in self.outputs.items()}
        frames_out = {width: byte_frame(name, event, revision, value) for width, value in logits.items()}
        return {'frames': frames_out, 'logits': logits, 'latent': latent,
                'state': RegionState(hat_state, output['state'], state.clock+1,
                                     state.bytes_seen+sum(f.width for f in frames)),
                'aux': hat_aux+output['aux']+.1*lace['balance'],
                'experts': output['expert_ids'][:, 0].detach()}
