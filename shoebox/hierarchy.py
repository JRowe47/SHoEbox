"""Causal 1/4/16-byte hierarchy with one explicit observed-prefix contract.

At position t, values[:, t] is observed and logits[:, t] predicts its successor.
Targets never enter state. Sampling feeds the actually sampled byte back through
this same API. There is no previous-argmax input or obligatory second round.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from .hierarchy_config import HierarchyConfig, ObjectiveConfig, CADENCES
from .hierarchy_objectives import RegionalObjectives, PRODUCERS, publication_records
from .latent_region import LatentFrame, LatentRegion
from .region import RegionState
from .runtime import forward_scope


@dataclass
class HierarchyState:
    batch: int
    nodes: dict
    latest: dict
    pending_local: tuple
    pending_mid: tuple
    event: int
    signature: dict

    def _map(self, fn):
        def visit(x):
            if torch.is_tensor(x):
                return fn(x)
            if is_dataclass(x):
                return type(x)(**{f.name: visit(getattr(x, f.name)) for f in fields(x)})
            if isinstance(x, dict):
                return {k: visit(v) for k, v in x.items()}
            if isinstance(x, tuple):
                return tuple(visit(v) for v in x)
            return x
        return visit(self)

    def detach(self):
        return self._map(lambda t: t.detach())

    def to(self, device):
        return self._map(lambda t: t.to(device))

    def storage_bytes(self):
        seen = set()
        size = 0
        def count(t):
            nonlocal size
            key = (str(t.device), t.untyped_storage().data_ptr())
            if key not in seen:
                seen.add(key)
                size += t.untyped_storage().nbytes()
            return t
        self._map(count)
        return size

    def to_dict(self):
        def pack(x):
            if is_dataclass(x):
                return {'_kind': type(x).__name__, **{f.name: pack(getattr(x, f.name)) for f in fields(x)}}
            if isinstance(x, dict):
                return {k: pack(v) for k, v in x.items()}
            if isinstance(x, tuple):
                return tuple(pack(v) for v in x)
            return x
        return pack(self)

    @classmethod
    def from_dict(cls, data):
        kinds = {c.__name__: c for c in (cls, LatentFrame, RegionState)}
        def unpack(x):
            if isinstance(x, dict):
                if '_kind' in x:
                    return kinds[x['_kind']](**{k: unpack(v) for k, v in x.items() if k != '_kind'})
                return {k: unpack(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                return tuple(unpack(v) for v in x)
            return x
        return unpack(data)


class JointDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, w = cfg.message_width, cfg.decoder_width
        self.local = nn.Linear(3*d, w)
        self.mid = nn.Linear(2*d, w, bias=False)
        self.high = nn.Linear(d, w, bias=False)
        self.interaction = nn.Sequential(nn.Linear(5*w+6, 2*w), nn.SiLU(), nn.Linear(2*w, w))
        self.message = nn.Linear(w, d, bias=False)
        self.readout = nn.LayerNorm(w)
        self.head = nn.Linear(w, 256, bias=False)
        nn.init.normal_(self.head.weight, std=.02)

    def combine(self, observed, latest, event, permutation=None, local_scale=1.):
        zero = torch.zeros_like(observed)
        local = self.local(torch.cat((observed, latest['A1'].value,
                                      latest['B1'].value if 'B1' in latest else zero), -1))
        if local_scale != 1.:
            # Weaken only the decoder shortcut, including its final residual.
            # The evidence entering A1/B1 and every upper region stays intact.
            local = local*local_scale
        def value(name):
            frame = latest.get(name)
            v = zero if frame is None else frame.value
            return v if permutation is None else v[permutation]
        mid = self.mid(torch.cat((value('L2'), value('S2')), -1))
        high = self.high(value('L3'))
        metadata = observed.new_tensor([
            v for name in ('L2', 'S2', 'L3')
            for v in ((math.log1p(event-latest[name].end)/math.log(129), 1.)
                      if name in latest else (0., 0.))]).expand(observed.shape[0], -1)
        joint = local+self.interaction(torch.cat((local, mid, high, local*mid, local*high, metadata), -1))
        return local, joint


def future_span_loss(head, latents, positions, targets, horizon):
    """Average over valid (boundary, stream, offset) targets, including tails.

targets[:, t] is x[t+1]. An optional extra H-1 target tail supervises the
last boundary without ever feeding that future tail to the recurrent model.
"""
    if not latents or targets is None:
        return next(head.parameters()).new_zeros(()), 0
    z = torch.stack(latents, 1)
    offsets = torch.tensor(positions, device=z.device)[:, None]+torch.arange(horizon, device=z.device)
    valid = offsets < targets.shape[1]
    labels = targets[:, offsets.clamp_max(targets.shape[1]-1)].masked_fill(~valid[None], -100)
    count = int((labels != -100).sum())
    logits = head(z).reshape(*z.shape[:2], horizon, 256)
    return (F.cross_entropy(logits.flatten(0, 2), labels.flatten()) if count else logits.sum()*0), count


class HierarchyModel(nn.Module):
    def __init__(self, cfg=None, objective=None):
        super().__init__()
        self.cfg = cfg or HierarchyConfig()
        self.cfg.validate()
        self.execution_backend = 'cpu'
        self.cpu_workers = 1
        self.embedding = nn.Embedding(256, self.cfg.message_width)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.regions = nn.ModuleDict({name: LatentRegion(self.cfg, name) for name in self.cfg.names})
        self.decoder = JointDecoder(self.cfg)
        if self.cfg.local_only:
            # The control contains exactly one recurrent HAT-SHoE. Use the same
            # observed-byte/local projection and final head, without idle banks.
            del self.decoder.mid, self.decoder.high, self.decoder.interaction, self.decoder.message
            self.future_heads = nn.ModuleDict()
        else:
            self.future_heads = nn.ModuleDict({
                'mid': nn.Linear(2*self.cfg.message_width, 4*256),
                'high': nn.Linear(self.cfg.message_width, 16*256)})
        self.configure_objective(objective or ObjectiveConfig())

    def configure_objective(self, objective):
        if objective.mode == 'latent' and self.cfg.local_only:
            raise ValueError('Regional latent objectives require the full hierarchy.')
        self.objective_config = objective
        if objective.mode == 'latent':
            if not hasattr(self, 'regional_objectives'):
                parameter = next(self.parameters())
                self.regional_objectives = RegionalObjectives(self.cfg.message_width, objective).to(parameter)
            self.regional_objectives.config = objective
        elif hasattr(self, 'regional_objectives'):
            del self.regional_objectives
        return self

    def set_execution_backend(self, backend):
        if backend not in ('cpu', 'optimized'):
            raise ValueError('Hierarchy backend must be cpu or optimized.')
        self.execution_backend = backend
        return self

    def initial_state(self, batch, device=None):
        if type(batch) is not int or batch < 1:
            raise ValueError('Batch size must be positive.')
        parameter = next(self.parameters())
        device = device or parameter.device
        nodes = {n: r.initial_state(batch, device, parameter.dtype) for n, r in self.regions.items()}
        return HierarchyState(batch, nodes, {}, (), (), 0, self.cfg.to_dict())

    @forward_scope
    def forward(self, values, targets=None, state=None, *, future_targets=None,
                trace=False, context_permutation=None, feedback_permutation=None,
                decoder_local_scale=1., document_ids=None, collect_messages=False):
        if (values.ndim != 2 or min(values.shape) < 1 or values.dtype not in (torch.uint8, torch.int32, torch.int64)
                or bool(((values < 0) | (values > 255)).any())):
            raise ValueError('Expected nonempty integer observations [B,T] in [0,255].')
        if values.device != next(self.parameters()).device:
            raise ValueError('Input and model must share a device.')
        batch, length = values.shape
        if document_ids is not None and (document_ids.shape != values.shape or
                document_ids.dtype != torch.long or document_ids.device != values.device):
            raise ValueError('Document IDs must be int64 [B,T] on the input device.')
        for labels in (targets, future_targets):
            if labels is not None and (labels.ndim != 2 or labels.shape[0] != batch or
                    labels.dtype != torch.long or labels.device != values.device or
                    bool(((labels != -100) & ((labels < 0) | (labels > 255))).any())):
                raise ValueError('Targets must be int64 [B,T] bytes or -100 on the input device.')
        if targets is not None and (targets.shape != values.shape or not bool((targets != -100).any())):
            raise ValueError('Next-byte targets must match observations and include a valid target.')
        if future_targets is not None and (targets is None or future_targets.shape[1] < length or
                                          not torch.equal(future_targets[:, :length], targets)):
            raise ValueError('Future target prefix must equal next-byte targets.')
        for permutation in (context_permutation, feedback_permutation):
            if permutation is not None and (permutation.dtype != torch.long or
                    permutation.device != values.device or permutation.shape != (batch,) or
                    not torch.equal(permutation.sort().values, torch.arange(batch, device=values.device))):
                raise ValueError('Context intervention must be a permutation of batch rows.')
        if not math.isfinite(decoder_local_scale) or not 0 <= decoder_local_scale <= 1:
            raise ValueError('Decoder local scale must be in [0,1].')
        if self.cfg.local_only and decoder_local_scale != 1.:
            raise ValueError('Local attenuation requires the full hierarchy.')
        state = state or self.initial_state(batch)
        if state.batch != batch or state.signature != self.cfg.to_dict():
            raise ValueError('State/configuration/batch mismatch.')
        if any(s.hat[0].device != values.device for s in state.nodes.values()):
            raise ValueError('State and input must share a device.')
        nodes, latest = dict(state.nodes), dict(state.latest)
        local_pending, mid_pending = list(state.pending_local), list(state.pending_mid)
        outputs, aux, records = [], [], []
        spans = {'mid': ([], []), 'high': ([], [])}
        latent_objective = targets is not None and self.objective_config.mode == 'latent'
        collect = latent_objective or collect_messages
        publications = {n: [] for n in PRODUCERS} if collect else None
        positions = {n: [] for n in PRODUCERS} if collect else None
        counts = {n: 0 for n in self.cfg.names}
        occurrences = {n: 0 for n in self.cfg.names}
        expert_rows = {n: [] for n in self.cfg.names} if trace else None
        observed = self.embedding(values.long())

        def run(name, frames, event, context=None):
            if context is not None and feedback_permutation is not None:
                context = replace(context, value=context.value[feedback_permutation])
            out = self.regions[name](frames, nodes[name], event, context=context,
                                     fused=self.execution_backend == 'cpu')
            nodes[name] = out['state']
            frame = LatentFrame(name, out['value'], frames[0].start, frames[-1].end, nodes[name].clock)
            latest[name] = frame
            counts[name] += batch
            occurrences[name] += batch*len(frames)
            aux.append(out['aux'])
            if collect and name in publications:
                publications[name].append(frame.value)
                positions[name].append(event)
            if trace:
                records.append(dict(region=name, event=event, inputs=[f.trace(event) for f in frames],
                                    context=None if context is None else context.trace(event)))
                expert_rows[name].append(out['experts'])
            return frame, out['latent']

        for t in range(length):
            event = state.event+t
            frame = LatentFrame('external', observed[:, t], event, event, event)
            # The entire span uses context published before its first event.
            # A1/B1 can use the last completed o2 span; o2 can use prior L3.
            a, _ = run('A1', [frame], event, latest.get('L2') if self.cfg.top_down else None)
            if self.cfg.local_only:
                local = self.decoder.local(torch.cat((observed[:, t], a.value, torch.zeros_like(a.value)), -1))
                outputs.append(self.decoder.head(self.decoder.readout(local)))
                continue
            b, _ = run('B1', [frame], event, latest.get('S2') if self.cfg.top_down else None)
            local_pending.append((a, b))
            if (event+1) % 4 == 0:
                if len(local_pending) != 4:
                    raise ValueError('Incomplete first-order source span.')
                context = latest.get('L3') if self.cfg.top_down else None
                l2, _ = run('L2', [p[0] for p in local_pending], event, context)
                s2, _ = run('S2', [f for pair in local_pending for f in pair], event, context)
                mid_pending.append((l2, s2))
                local_pending.clear()
                if targets is not None and not latent_objective:
                    spans['mid'][0].append(torch.cat((l2.value, s2.value), -1))
                    spans['mid'][1].append(t)
            if (event+1) % 16 == 0:
                if len(mid_pending) != 4:
                    raise ValueError('Incomplete second-order source span.')
                high, _ = run('L3', [f for pair in mid_pending for f in pair], event)
                mid_pending.clear()
                if targets is not None and not latent_objective:
                    spans['high'][0].append(high.value)
                    spans['high'][1].append(t)
            local, joint = self.decoder.combine(observed[:, t], latest, event, context_permutation,
                                                decoder_local_scale)
            # Y remains an ordinary small HAT-SHoE. Its one new occurrence is
            # the nonlinear conjunction of separate local/mid/high slots.
            y_frame = LatentFrame('joint', self.decoder.message(joint), event, event, event)
            _, y = run('Y', [y_frame], event)
            outputs.append(self.decoder.head(self.decoder.readout(local+y)))
        logits = torch.stack(outputs, 1)
        ce = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        balance = torch.stack(aux).mean()
        losses, target_counts = {}, {}
        for scale, horizon in (('mid', 4), ('high', 16)):
            losses[scale], target_counts[scale] = (future_span_loss(self.future_heads[scale], *spans[scale],
                    future_targets if future_targets is not None else targets, horizon)
                    if scale in self.future_heads else (logits.new_zeros(()), 0))
        loss = None if ce is None else (ce+self.cfg.mid_loss_weight*losses['mid']+
                self.cfg.high_loss_weight*losses['high']+self.cfg.expert_balance_weight*balance)
        messages, regional = {}, None
        if collect:
            if document_ids is None:
                # One uninterrupted document per row is the standalone API default.
                document_ids = torch.arange(batch, device=values.device)[:, None].expand_as(values)
            valid = torch.ones_like(values, dtype=torch.bool) if targets is None else targets != -100
            messages = publication_records(publications, positions, start=state.event,
                                           documents=document_ids, valid=valid)
        if latent_objective:
            regional = self.regional_objectives(messages)
            loss = ce+self.cfg.expert_balance_weight*balance+regional['loss']
        return dict(logits=logits, ce=ce, loss=loss, aux=balance, span_losses=losses,
                    span_targets=target_counts, trace=records, regional=regional,
                    publications=messages,
                    experts={n: torch.stack(rows) for n, rows in expert_rows.items() if rows} if trace else {},
                    work=dict(region_rows=counts, occurrence_rows=occurrences, external_bytes=values.numel()),
                    state=HierarchyState(batch, nodes, latest, tuple(local_pending), tuple(mid_pending),
                                         state.event+length, state.signature))

    def parameter_cost(self):
        """Scheduled parameter touches per external byte, not claimed FLOPs.

        Counts every small router/recurrent/dense parameter on a scheduled
        invocation and only k/E expert weights. AdamW still updates dense banks.
        """
        regions, active = {}, 0.
        for name, region in self.regions.items():
            total = sum(p.numel() for p in region.parameters())
            banks = sum(p.numel() for b in region.core.blocks for p in b.experts.private.parameters())
            selected = self.cfg.active_groups*self.cfg.experts_per_group
            used = total-banks+banks*selected/self.cfg.experts_per_layer
            active += used/CADENCES[name]
            regions[name] = dict(parameters=total, active_per_invocation=used, cadence=CADENCES[name])
        decoder = sum(p.numel() for p in self.decoder.parameters())
        future = sum(p.numel() for p in self.future_heads.parameters())
        regional = sum(p.numel() for p in self.regional_objectives.parameters()) if hasattr(self, 'regional_objectives') else 0
        return dict(total=sum(p.numel() for p in self.parameters()), regions=regions,
                    decoder_parameters=decoder, training_head_parameters=future+regional,
                    latent_head_parameters=regional,
                    inference_parameters=sum(p.numel() for p in self.parameters())-future-regional,
                    scheduled_inference_parameter_touches_per_byte=active+decoder+self.cfg.message_width,
                    region_executions_per_byte=sum(1/CADENCES[n] for n in self.cfg.names),
                    note='Parameter-touch estimate, not FLOPs or wall time; optimizer banks are dense.')
