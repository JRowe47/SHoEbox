"""Event-major cortical graph with committed feedback and an EC-only boundary."""
from dataclasses import dataclass, fields, is_dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from .config import SkeletonConfig, REGIONS, NAMES
from .transport import Frame, map_tensors, detach_state, publish, recipe_frames
from .region import CorticalRegion, RegionState
from .hippocampus import Hippocampus, EpisodicState, Episode, SparseMemory
from .runtime import forward_scope


@dataclass
class CorticalState:
    batch: int
    nodes: dict
    ports: dict
    memory: EpisodicState
    latents: dict
    output: Frame | None = None
    event: int = 0
    signature: dict | None = None

    def detach(self, compact=False):
        return detach_state(self, compact)

    def to(self, device):
        return map_tensors(self, lambda t: t.to(device))

    def storage_bytes(self):
        seen, size = set(), 0
        def count(t):
            nonlocal size
            key = (str(t.device), t.untyped_storage().data_ptr())
            if key not in seen:
                seen.add(key)
                size += t.untyped_storage().nbytes()
            return t
        map_tensors(self, count)
        return size

    def to_dict(self):
        def pack(x):
            if is_dataclass(x):
                return {'_kind': type(x).__name__, **{f.name: pack(getattr(x, f.name)) for f in fields(x)}}
            if isinstance(x, dict):
                return {k: pack(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return tuple(pack(v) for v in x)
            return x
        return pack(self)

    @classmethod
    def from_dict(cls, data):
        kinds = {c.__name__: c for c in (cls, Frame, RegionState, EpisodicState, Episode, SparseMemory)}
        def unpack(x):
            if isinstance(x, dict):
                if '_kind' in x:
                    return kinds[x['_kind']](**{k: unpack(v) for k, v in x.items() if k != '_kind'})
                return {k: unpack(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return tuple(unpack(v) for v in x)
            return x
        return unpack(data)


class CorticalModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or SkeletonConfig()
        self.cfg.validate()
        self.specs = self.cfg.topology()
        self.execution_backend = 'reference'
        widths = {name: set() for name in NAMES}
        for region in self.specs:
            for edge in region.inputs:
                widths[edge.source].add(edge.width)
        widths['EC-C'].add(self.cfg.max_port_bytes)
        widths['EC-X'].add(self.cfg.max_port_bytes)
        widths['Y'].add(1)
        self.regions = nn.ModuleDict({spec.name: CorticalRegion(self.cfg.for_region(spec.name), i, widths[spec.name],
                                                              temporal_scale=4. if spec.name == 'B1' else 1.,
                                                              input_scale=self.cfg.port_scale if spec.order>1 else 1.)
                                      for i, spec in enumerate(self.specs)})
        self.hippocampus = Hippocampus(self.cfg)
        self.cpu_workers = 1
        if self.cfg.region_topk:
            from .regional_routing import RegionalRouter, compile_waves
            self.graph_router = RegionalRouter(self.cfg)
            self.execution_waves = compile_waves(self.specs)

    def set_execution_backend(self, backend):
        if backend not in ('reference', 'optimized', 'fused', 'cpu'):
            raise ValueError('Backend must be reference, optimized, fused or cpu.')
        self.execution_backend = backend
        for region in self.regions.values():
            region.optimized = region.hat.optimized = backend != 'reference'
            region.fused = region.hat.fused = backend in ('fused', 'cpu')
            region.compact_transport = region.hat.compact_transport = backend == 'cpu'
        return self

    def set_cpu_workers(self, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError('CPU worker count must be positive.')
        self.cpu_workers = workers
        return self

    def initial_state(self, batch, device=None):
        if type(batch) is not int or batch < 1:
            raise ValueError('Batch size must be positive.')
        parameter = next(self.parameters())
        device = device or parameter.device
        nodes = {name: region.initial_state(batch, device, parameter.dtype) for name, region in self.regions.items()}
        memory = EpisodicState()
        if self.cfg.region_topk:
            memory = SparseMemory(tuple(EpisodicState() for _ in range(batch)))
            for node in nodes.values():
                node.clock = torch.zeros(batch, dtype=torch.long, device=device)
                node.bytes_seen = torch.zeros_like(node.clock)
        return CorticalState(batch, nodes, {}, memory, {}, signature=self.cfg.to_dict())

    @forward_scope
    def forward(self, values, targets=None, state=None, *, source=0, trace=False,
                continuation=None, memory_enabled=None, observer=None):
        """Predict each next byte from observations and preceding emitted output.

        values is integer [B,T] external input. continuation optionally supplies
        actual preceding output bytes [B,T] (or -1 for no output). It never
        becomes external observation. Without it, Y receives its previous
        argmax emission, rather than the current target.
        """
        if not torch.is_tensor(values) or values.dtype not in (torch.uint8, torch.int32, torch.int64):
            raise ValueError('Observed bytes must be an integer tensor.')
        if values.ndim != 2 or min(values.shape) < 1 or bool(((values < 0) | (values > 255)).any()):
            raise ValueError('Expected nonempty observed bytes [B,T] in [0,255].')
        if type(source) is not int or not 0 <= source < 16:
            raise ValueError('External source must be an integer from 0 to 15.')
        if values.device != next(self.parameters()).device:
            raise ValueError('Input and model must share a device.')
        b, t = values.shape
        if targets is not None:
            if targets.dtype != torch.long or targets.shape != values.shape or targets.device != values.device:
                raise ValueError('Targets must be int64 [B,T] on the input device.')
            if bool(((targets != -100) & ((targets < 0) | (targets > 255))).any()) or not bool((targets != -100).any()):
                raise ValueError('Targets need at least one valid byte; only -100 is ignored.')
        if continuation is not None:
            if (continuation.dtype != torch.long or continuation.shape != values.shape or
                    continuation.device != values.device or bool(((continuation < -1) | (continuation > 255)).any())):
                raise ValueError('Continuation must be int64 [B,T] bytes or -1.')
            if not bool(((continuation == -1) == (continuation[:1] == -1)).all()):
                raise ValueError('Continuation presence must agree across batch rows.')
        state = state or self.initial_state(b, values.device)
        if state.batch != b or state.signature != self.cfg.to_dict():
            raise ValueError('State does not match this model configuration/batch.')
        if any(v.hat[0].device != values.device for v in state.nodes.values()):
            raise ValueError('State and input must share a device; use state.to(device).')
        memory_on = self.cfg.memory_enabled if memory_enabled is None else bool(memory_enabled)
        losses, traces, outputs, routing = [], [], [], []
        current_state = state
        for step in range(t):
            result = self._event(values[:, step:step+1].long(), current_state, source, trace,
                                 None if continuation is None else continuation[:, step:step+1], memory_on, observer)
            current_state = result['state']
            outputs.append(result['logits'])
            losses.append(result['aux'])
            traces.extend(result['trace'])
            if 'routing' in result:
                routing.append(result['routing'])
        logits = torch.stack(outputs, 1)
        aux = torch.stack(losses).mean()
        ce = None if targets is None else F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        return {'logits': logits, 'state': current_state, 'ce': ce, 'aux': aux,
                'loss': None if ce is None else ce+self.cfg.auxiliary_weight*aux, 'trace': traces, 'routing': routing}

    def _event(self, values, state, source, trace, continuation, memory_on, observer=None):
        if self.cfg.region_topk:
            from .sparse_executor import sparse_event
            return sparse_event(self, values, state, source, trace, continuation, memory_on, observer)
        event = state.event
        nodes, ports, latents = dict(state.nodes), dict(state.ports), dict(state.latents)
        observed = Frame(f'external:{source}', event, 0, values)
        previous_output = state.output
        if continuation is not None:
            previous_output = (None if bool((continuation == -1).all()) else
                               Frame('Y-output', event-1, 0, continuation))
        aux, records = [], []
        zero_c = next(self.parameters()).new_zeros(state.batch, self.cfg.for_region('EC-C').width)
        zero_x = next(self.parameters()).new_zeros(state.batch, self.cfg.for_region('EC-X').width)
        for revision in range(self.cfg.rounds):
            snapshot = ports
            returns = (self.hippocampus.read(latents.get('EC-C', zero_c), latents.get('EC-X', zero_x),
                                            state.memory, event, revision-1)
                       if memory_on else {'EC-C': [], 'EC-X': []})
            for spec in self.specs:
                if spec.order == 1 and revision:
                    continue
                frames = [observed] if spec.order == 1 else recipe_frames(spec, ports, snapshot)
                frames += returns.get(spec.name, [])
                if spec.name == 'Y' and previous_output is not None:
                    frames = [*frames, previous_output]
                collect = observer is not None and (spec.order == 1 or revision == self.cfg.rounds-1)
                result = self.regions[spec.name](spec.name, frames, nodes[spec.name], event, revision,observe=collect)
                if observer is not None and (spec.order == 1 or revision == self.cfg.rounds-1):
                    observer(spec.name, event, revision, result)
                nodes[spec.name], latents[spec.name] = result['state'], result['latent']
                ports = publish(ports, spec.name, result['frames'])
                aux.append(result['aux'])
                if trace:
                    records.append({'region': spec.name, 'event': event, 'round': revision,
                                    'inputs': [f.trace() for f in frames],
                                    'experts': result['experts'].cpu().tolist()})
                if spec.name == 'Y':
                    logits = result['logits'][1][:, 0]
                    emission = replace(result['frames'][1], source='Y-output', origin_source='Y')
        memory = state.memory
        if memory_on:
            memory = self.hippocampus.write(ports[f'EC-C:{self.cfg.max_port_bytes}'][-1],
                                           ports[f'EC-X:{self.cfg.max_port_bytes}'][-1],
                                           latents['EC-C'], latents['EC-X'], memory)
        return {'logits': logits, 'aux': torch.stack(aux).mean(), 'trace': records,
                'state': CorticalState(state.batch, nodes, ports, memory, latents, emission,
                                       event+1, state.signature)}
