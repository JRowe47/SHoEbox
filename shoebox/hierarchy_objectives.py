"""One-step future-input prediction, using only current-window publications.

Slot order is time-major, source-minor. Row/document identities and absolute
boundary times are carried separately, without enabling the routing trace.
Nothing in this module runs during sampling. Phase A detaches labels only;
current representations still carry joint upstream credit.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .hierarchy_config import CADENCES
from .latent_region import SOURCE_IDS

RECIPES = {'L2': ('A1',), 'S2': ('A1', 'B1'), 'L3': ('L2', 'S2')}
PRODUCERS = ('A1', 'B1', 'L2', 'S2', 'L3')


@dataclass
class Publications:
    values: torch.Tensor  # [batch, publications, channels], live graph
    positions: torch.Tensor  # absolute byte boundaries
    valid: torch.Tensor  # whole source spans inside this window/document
    documents: torch.Tensor
    streams: torch.Tensor
    source_id: int


def publication_records(values, positions, *, start, documents, valid):
    """Vectorized validity checks include every byte of a publication's span."""
    batch, length = documents.shape
    result = {}
    for source, entries in values.items():
        if not entries:
            continue
        z = torch.stack(entries, 1)
        times = torch.tensor(positions[source], device=z.device, dtype=torch.long)
        relative = times-start
        span = relative[:, None]-torch.arange(CADENCES[source]-1, -1, -1, device=z.device)
        inside = (span >= 0).all(-1)
        indices = span.clamp(0, length-1)
        doc = documents[:, relative]
        ok = inside[None] & valid[:, indices].all(-1)
        ok &= (documents[:, indices] == doc[:, :, None]).all(-1) & (doc >= 0)
        result[source] = Publications(z, times, ok, doc,
                torch.arange(batch, device=z.device), SOURCE_IDS[source])
    return result


def future_pairs(records, destination):
    """Return predictor inputs, real future bundles and persistence bundles.

    Only complete current and future bundles wholly within this call are used.
    The validity mask requires the same stream and document for all slots.
    A new call starts new records, so no graph survives an optimizer update.
    """
    current = records.get(destination)
    sources = RECIPES[destination]
    if current is None or any(s not in records for s in sources):
        return None
    stride = CADENCES[sources[0]]
    future_times = current.positions[:, None]+stride*torch.arange(1, 5, device=current.values.device)
    prior_times = current.positions[:, None]-stride*torch.arange(3, -1, -1, device=current.values.device)
    ok = current.valid.clone()
    future, prior = [], []
    for source in sources:
        record = records[source]
        if not torch.equal(record.streams, current.streams):
            raise ValueError('Publication stream identities disagree.')
        selections = []
        for times in (future_times, prior_times):
            index = torch.searchsorted(record.positions, times)
            in_range = index < record.positions.numel()
            index = index.clamp_max(record.positions.numel()-1)
            exact = in_range & (record.positions[index] == times)
            ok &= exact.all(-1)[None] & record.valid[:, index].all(-1)
            ok &= (record.documents[:, index] == current.documents[:, :, None]).all(-1)
            selections.append(record.values[:, index])
        future.append(selections[0]); prior.append(selections[1])
    # [valid examples, four times, source roles, channels]
    return dict(input=current.values[ok], target=torch.stack(future, -2)[ok].detach(),
                persistence=torch.stack(prior, -2)[ok].detach(),
                mask=ok, positions=current.positions, sources=sources,
                possible=int(current.valid.sum()))


class SourceNormalizer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('variance', torch.ones(width))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def observe(self, values, rate):
        if values.shape[0] < 2:
            return
        values = values.detach()
        variance, mean = torch.var_mean(values, dim=0, unbiased=False)
        if not int(self.count):
            self.mean.copy_(mean); self.variance.copy_(variance)
        else:
            delta = mean-self.mean
            self.variance.mul_(1-rate).add_(variance, alpha=rate).add_(delta.square(), alpha=rate*(1-rate))
            self.mean.lerp_(mean, rate)
        self.count.add_(values.shape[0])

    def normalize(self, values, floor):
        return (values.detach()-self.mean)/self.variance.clamp_min(floor).sqrt()


def diversity_loss(records, target_std):
    """Bounded variance and redundancy penalties on LIVE source publications."""
    variance, covariance, health = [], [], {}
    for source, record in records.items():
        x = record.values[record.valid]
        if x.shape[0] < 2:
            continue
        centered = x-x.mean(0)
        var = centered.square().mean(0)
        std = (var+1e-6).sqrt()
        variance.append(F.relu(target_std-std).mean()/target_std)
        scaled = centered/std.detach().clamp_min(target_std)
        cov = scaled.T@scaled/x.shape[0]
        off = ~torch.eye(x.shape[-1], device=x.device, dtype=torch.bool)
        redundancy = cov[off].square()
        covariance.append((redundancy/(1+redundancy)).mean() if redundancy.numel() else x.sum()*0)
        health[source] = dict(examples=x.shape[0], mean_std=float(std.detach().mean()),
                min_std=float(std.detach().min()), fraction_below_floor=float((std < target_std).float().mean()))
    zero = next(iter(records.values())).values.sum()*0
    return (torch.stack(variance).mean() if variance else zero,
            torch.stack(covariance).mean() if covariance else zero, health)


class RegionalObjectives(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.config = config
        # Isolated deterministic initialization leaves shared model and RNG intact.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.head_seed)
            self.predictors = nn.ModuleDict({n: nn.Linear(width, 4*len(s)*width) for n, s in RECIPES.items()})
        self.normalizers = nn.ModuleDict({n: SourceNormalizer(width) for n in PRODUCERS[:-1]})

    def normalized_bundle(self, values, sources):
        return torch.stack([self.normalizers[s].normalize(values[:, :, i], self.config.normalizer_variance_floor)
                            for i, s in enumerate(sources)], -2)

    def forward(self, records):
        cfg = self.config
        zero = next(self.parameters()).new_zeros(())
        if not records:
            return dict(loss=zero, losses={}, pairs={}, metrics={}, variance=zero, covariance=zero, health={})
        if self.training:
            for source, normalizer in self.normalizers.items():
                if source in records:
                    r = records[source]
                    normalizer.observe(r.values[r.valid], cfg.normalizer_rate)
        losses, counts, metrics = {}, {}, {}
        for name in RECIPES:
            pair = future_pairs(records, name)
            count = 0 if pair is None else pair['input'].shape[0]
            counts[name] = count
            if not count:
                losses[name] = zero
                metrics[name] = dict(pairs=0, possible=0 if pair is None else pair['possible'])
                continue
            target = self.normalized_bundle(pair['target'], pair['sources'])
            previous = self.normalized_bundle(pair['persistence'], pair['sources'])
            prediction = self.predictors[name](pair['input']).reshape_as(target)
            per_role = F.smooth_l1_loss(prediction, target, reduction='none').mean((0, 1, 3))
            losses[name] = per_role.mean()  # explicit equal source-role weighting
            with torch.no_grad():
                # Rotate complete bundles, retaining their internal source/time order.
                shuffled = target.roll(max(1, count//2), 0)
                metrics[name] = dict(pairs=count, possible=pair['possible'],
                    huber=float(losses[name]), persistence=float(F.smooth_l1_loss(previous, target)),
                    shuffled_future=float(F.smooth_l1_loss(prediction, shuffled)),
                    mse=float(F.mse_loss(prediction, target)),
                    persistence_mse=float(F.mse_loss(previous, target)),
                    shuffled_future_mse=float(F.mse_loss(prediction, shuffled)),
                    source_huber={s: float(per_role[i]) for i, s in enumerate(pair['sources'])},
                    shuffled_examples=count if count > 1 else 0)
        variance, covariance, health = diversity_loss(records, cfg.target_std)
        prediction_loss = sum(getattr(cfg, n.lower()+'_weight')*losses[n] for n in RECIPES)
        return dict(loss=prediction_loss+cfg.variance_weight*variance+cfg.covariance_weight*covariance,
                    losses=losses, pairs=counts, metrics=metrics,
                    variance=variance, covariance=covariance, health=health)
