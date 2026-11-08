"""A small shared DG/association/readback boundary, owned outside cortex.

Only EC-C/EC-X enter or leave this subsystem. One stored pair per observation,
bounded capacity, sparse addresses, distinct top-k frames and a learned null.
There are no cortical owner callbacks, capsules, question lanes or replay.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from .transport import Frame


@dataclass(frozen=True)
class Episode:
    event: int
    content: Frame
    context: Frame
    features: torch.Tensor


@dataclass(frozen=True)
class EpisodicState:
    episodes: tuple = ()
    writes: int = 0


class Hippocampus(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.address = nn.Linear(2*cfg.width, cfg.episode_address, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(1.))
        self.null = nn.Parameter(torch.tensor(0.))

    def encode(self, features):
        z = self.address(features)
        selected = z.abs().topk(self.cfg.episode_active, -1).indices
        sparse = torch.zeros_like(z).scatter(-1, selected, z.gather(-1, selected))
        return F.normalize(sparse, dim=-1, eps=1e-6)

    def read(self, content, context, state, event, revision):
        if not state.episodes:
            return {'EC-C': [], 'EC-X': []}
        # Re-encode bounded, detached stored features with current weights.
        features = torch.cat((content, context), -1)
        query = self.encode(features)
        keys = self.encode(torch.stack([e.features for e in state.episodes], 1))
        scores = (query[:, None]*keys).sum(-1)*self.log_scale.clamp(-3, 4).exp()
        count = min(self.cfg.episode_reads, len(state.episodes))
        selected_scores, indices = scores.topk(count, -1)
        weights = torch.cat((selected_scores, self.null.expand(scores.shape[0], 1)), -1).softmax(-1)
        row = torch.arange(features.shape[0], device=features.device)
        events = torch.tensor([e.event for e in state.episodes], device=features.device)
        returns = {}
        for owner, field, source in [('EC-C', 'content', 'HC-C'), ('EC-X', 'context', 'HC-X')]:
            values = torch.stack([getattr(e, field).values for e in state.episodes], 1)
            revisions = torch.tensor([getattr(e, field).revision for e in state.episodes], device=features.device)
            frames = []
            for k in range(count):
                ids = indices[:, k]
                frames.append(Frame(source, event, revision, values[row, ids],
                                    strength=weights[:, k], origin_source=owner,
                                    episode_events=events[ids], episode_revisions=revisions[ids]))
            returns[owner] = frames
        return returns

    def write(self, content, context, content_latent, context_latent, state):
        if content.source != 'EC-C' or context.source != 'EC-X' or content.event != context.event:
            raise ValueError('Hippocampus accepts aligned EC publications only.')
        if state.episodes and content.event <= state.episodes[-1].event:
            raise ValueError('An occurrence may enter episodic storage only once, in order.')
        episode = Episode(content.event,
                          replace(content.detach(), distribution=None),
                          replace(context.detach(), distribution=None),
                          torch.cat((content_latent, context_latent), -1).detach())
        return EpisodicState((*state.episodes, episode)[-self.cfg.episode_capacity:], state.writes+1)
