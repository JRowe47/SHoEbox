"""Task-driven owned memory on the ordinary byte stream.

The producer is one shared regional program executed on independent owner states.
Its sparse experts construct payloads and semantic headers. A current-context query
reads committed publications before a composer executes; new publications commit
only after the prediction. No task labels or field addresses enter this module.
"""
from dataclasses import asdict, dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from ..hat_minkron.events import ByteBatch
from ..hat_minkron.hat import HATState
from ..hat_minkron.semantic_core import ExpertBank
from .config import GraphConfig, NodeSpec
from .node import RecurrentNode
from .ports import Publication, SemanticHeader, publish, map_tensors, plain_dict, tensor_list
from .state import NodeState
from .runtime import RuntimeHAT, forward_scope, next_event
from .task_payload import PayloadAdapter
from .task_state_commit import commit_producer_state


@dataclass(frozen=True)
class TaskMemoryConfig:
    width: int = 80
    owners: int = 4
    policy_features: int = 8
    experts_per_block: int = 128
    shared_per_block: int = 64
    expert_hidden: int = 4
    groups: int = 8
    active_groups: int = 2
    experts_per_group: int = 2
    address_dim: int = 16
    laces: int = 32
    active_laces: int = 2
    hat_levels: int = 1
    hat_symbols: int = 32
    hat_active_symbols: int = 2
    kron_heads: int = 2
    kron_key: int = 4
    kron_value: int = 4
    seed: int = 1
    write_mode: str = 'independent'
    memory_aware_write: bool = False
    bootstrap_empty: bool = False
    payload_rank: int = 0
    state_commit: str = 'always'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def graph_config(self):
        values = self.to_dict()
        for key in ('owners', 'policy_features', 'write_mode', 'memory_aware_write', 'bootstrap_empty', 'payload_rank', 'state_commit'):
            values.pop(key)
        if self.state_commit not in ('always', 'publication'):
            raise ValueError('State commit must be always or publication.')
        if not 0 <= self.payload_rank <= self.width:
            raise ValueError('Payload rank must be between zero and payload width.')
        if self.owners < 1 or self.policy_features < 1:
            raise ValueError('Positive owner and policy feature counts required.')
        if self.write_mode not in ('independent', 'competitive'):
            raise ValueError('Write mode must be independent or competitive.')
        return GraphConfig(**values, nodes=(NodeSpec('M', 'memory', 'ordinary'),
                                            NodeSpec('C', 'composer')),
                           edges=(), readout=('C',), context=32)


@dataclass(frozen=True)
class TaskMemoryState:
    ingress: HATState
    producer: NodeState
    composer: NodeState
    publications: dict[str, Publication]
    events_seen: torch.Tensor
    epochs: torch.Tensor

    def detach(self):
        return map_tensors(self, lambda x: x.detach())

    def clone(self):
        return map_tensors(self, lambda x: x.clone())

    def to(self, device):
        return map_tensors(self, lambda x: x.to(device))

    def reset(self, rows):
        if rows.dtype != torch.bool or rows.shape != self.events_seen.shape:
            raise ValueError('Reset must be bool [B].')
        return replace(self, ingress=self.ingress.reset(rows),
                       producer=self.producer.reset(rows.repeat_interleave(len(self.publications))),
                       composer=self.composer.reset(rows),
                       publications={k: p.reset(rows) for k, p in self.publications.items()},
                       events_seen=self.events_seen*(~rows), epochs=self.epochs+rows.long())

    def memory_tensor(self):
        return torch.stack([torch.cat((p.payload, p.header.memberships.flatten(1)), -1)
                            for p in self.publications.values()], 1)

    def with_memory_tensor(self, memory):
        if memory.shape != self.memory_tensor().shape:
            raise ValueError('Memory tensor shape mismatch.')
        pubs = {}
        for i, (name, pub) in enumerate(self.publications.items()):
            d = pub.payload.shape[-1]
            pubs[name] = replace(pub, payload=memory[:, i, :d],
                                header=replace(pub.header, memberships=memory[:, i, d:].reshape_as(pub.header.memberships)))
        return replace(self, publications=pubs)

    def tensors(self):
        return tensor_list(self)

    def storage_bytes(self):
        unique = {(str(x.device), x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
                  for x in self.tensors()}
        return sum(unique.values())

    def to_dict(self):
        return plain_dict(self.detach())

    @classmethod
    def from_dict(cls, data):
        def node(d):
            return NodeState(tuple(d['contextualizer']), tuple(d['blocks']), d['clock'])
        return cls(HATState(**data['ingress']), node(data['producer']), node(data['composer']),
                   {k: Publication.from_dict(v) for k, v in data['publications'].items()},
                   data['events_seen'], data['epochs'])


class TaskMemoryModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or TaskMemoryConfig()
        c = self.cfg.graph_config()
        c.validate()
        self.graph_cfg = c
        self.execution_backend = 'optimized'
        self._large_expert_banks = False
        h = c.hat_config()
        self.hat = RuntimeHAT(h, safe_zero=True)
        self.shared_banks = nn.ModuleList(ExpertBank(h.child_config(), c.shared_per_block) for _ in range(3))
        self.producer = RecurrentNode(c.nodes[0], c, self.shared_banks, 0)
        self.composer = RecurrentNode(c.nodes[1], c, self.shared_banks, 1)
        d, m, f, s = c.width, self.cfg.owners, self.cfg.policy_features, 3*c.groups
        self.owner_embedding = nn.Parameter(torch.randn(m, d)*.2)
        self.policy_features = nn.Linear(d, f)
        self.write_policy = nn.Parameter(torch.randn(m, f+1)*.15)
        with torch.no_grad():
            self.write_policy[:, -1].copy_(torch.linspace(-.15, .15, m))
        self.query = nn.Linear(d, c.address_dim, bias=False)
        self.query_semantic = nn.Linear(s, c.address_dim, bias=False)
        self.key = nn.Linear(d+s, c.address_dim, bias=False)
        self.null_read = nn.Linear(d, 1)
        nn.init.zeros_(self.null_read.weight)
        nn.init.zeros_(self.null_read.bias)
        self.read_value = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.read_value.weight)
        self.read_context = nn.Linear(s, s, bias=False)
        nn.init.eye_(self.read_context.weight)
        self.ingress_value = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.ingress_value.weight)
        self.decoder = nn.Linear(d, 256)
        nn.init.normal_(self.decoder.weight, std=.02)
        nn.init.zeros_(self.decoder.bias)
        self.owner_names = tuple(f'M{i}' for i in range(m))
        if self.cfg.memory_aware_write:
            # Preserve common initialization and the caller's random stream.
            # Zero initialization also preserves initial policy logits exactly.
            with torch.random.fork_rng(devices=[]):
                self.memory_policy_features = nn.Linear(d+s+4, f, bias=False)
            nn.init.zeros_(self.memory_policy_features.weight)
        if self.cfg.payload_rank:
            self.payload_adapter = PayloadAdapter(m, d, self.cfg.payload_rank)

    @property
    def memory_dim(self):
        return self.cfg.width+3*self.cfg.groups

    def initial_state(self, batch, device=None):
        device = device or self.write_policy.device
        dtype = self.write_policy.dtype
        d, c = self.cfg.width, self.cfg
        pubs = {}
        for name in self.owner_names:
            zero = torch.zeros(batch, dtype=torch.long, device=device)
            header = SemanticHeader(name, torch.zeros(batch, 3, c.groups, device=device, dtype=dtype),
                                    torch.zeros(batch, 3, c.active_groups, dtype=torch.long, device=device),
                                    torch.zeros(batch, 3, c.active_laces, dtype=torch.long, device=device))
            pubs[name] = Publication(name, 'latent', torch.zeros(batch, d, device=device, dtype=dtype),
                                    torch.zeros(batch, dtype=torch.bool, device=device),
                                    zero, zero, zero, zero, header)
        return TaskMemoryState(self.hat.initial_state(batch),
                               self.producer.initial_state(batch*c.owners, device, dtype),
                               self.composer.initial_state(batch, device, dtype), pubs,
                               torch.zeros(batch, dtype=torch.long, device=device),
                               torch.zeros(batch, dtype=torch.long, device=device))

    @forward_scope
    def forward(self, inputs, state=None, reset=None, control='intact', write_override=None, capture=True):
        if inputs.ndim != 2 or inputs.dtype != torch.long or inputs.shape[1] < 1:
            raise ValueError('Inputs must be nonempty int64 [B,T] bytes.')
        if bool(((inputs < 0) | (inputs > 255)).any()):
            raise ValueError('Inputs must use the byte vocabulary.')
        if control not in ('intact', 'blocked', 'shuffle', 'fresh'):
            raise ValueError('Unknown memory control.')
        if write_override not in (None, 'hold', 'always'):
            raise ValueError('Write override must be hold or always.')
        b, t = inputs.shape
        c, m, d = self.cfg, self.cfg.owners, self.cfg.width
        state = state or self.initial_state(b, inputs.device)
        if state.events_seen.shape != (b,) or state.events_seen.device != inputs.device:
            raise ValueError('State batch/device mismatch.')
        if reset is not None and (reset.dtype != torch.bool or reset.shape not in ((b,), (b, t)) or reset.device != inputs.device):
            raise ValueError('Reset must be bool [B] or [B,T].')
        logits, records = [], {k: [] for k in (
            'read_weights', 'read_versions', 'read_valid', 'read_is_current',
            'write_probs', 'write_actions', 'write_features', 'state_actions',
            'memory_before', 'memory_candidates', 'memory_versions', 'memory_valid', 'write_eligible')}
        bootstrap_masks = []
        state_masks = []
        if c.payload_rank:
            records['payload_features'] = []
        aux = self.write_policy.new_zeros(())
        hat_aux = aux
        active = torch.ones(b, dtype=torch.bool, device=inputs.device)
        producer_active = active.repeat_interleave(m)
        for index in range(t):
            next_event()
            if reset is not None and (reset.ndim == 2 or index == 0):
                rows = reset[:, index] if reset.ndim == 2 else reset
                if bool(rows.any()):
                    state = state.reset(rows)
            ingress = self.hat(ByteBatch.static(inputs[:, index:index+1]), state.ingress, backend='step')
            x = ingress['payload'][:, 0]
            before = state.memory_tensor()
            memberships = x.new_zeros(b*m, 3, c.groups)
            producer = self.producer((x[:, None]+self.owner_embedding[None]).reshape(b*m, d),
                                     memberships, state.producer, producer_active)
            raw_payload = producer['latent'].reshape(b, m, d)
            candidate_payload = raw_payload
            if c.payload_rank:
                candidate_payload, payload_features = self.payload_adapter(raw_payload)
            candidate_header = producer['header'].memberships.reshape(b, m, 3*c.groups)
            candidates = torch.cat((candidate_payload, candidate_header), -1)
            query = F.normalize(self.query(x)+self.query_semantic(candidate_header.mean(1)), dim=-1)
            memory = before
            owned_valid = torch.stack([p.valid for p in state.publications.values()], 1)
            valid = owned_valid
            versions = torch.stack([p.version for p in state.publications.values()], 1)
            if control == 'shuffle':
                memory, valid, versions = memory.roll(1, 0), valid.roll(1, 0), versions.roll(1, 0)
            elif control == 'fresh':
                # Same producer/composer compute, with current expert outputs
                # replacing persistent publications at the read interface.
                memory = candidates
                valid = torch.ones_like(owned_valid)
                versions = torch.full_like(versions, -1)
            keys = F.normalize(self.key(memory), dim=-1)
            scores = 4*(query[:, None]*keys).sum(-1)
            scores = scores.masked_fill(~valid, -1e4)
            weights = torch.cat((scores, self.null_read(x)), -1).softmax(-1)
            effective = weights[:, :m] if control != 'blocked' else weights[:, :m]*0
            retrieved = (effective[:, :, None]*memory).sum(1)
            composer = self.composer(self.ingress_value(x)+self.read_value(retrieved[:, :d]),
                                     self.read_context(retrieved[:, d:]).reshape(b, 3, c.groups),
                                     state.composer, active)
            logits.append(self.decoder(composer['latent']))
            # Admission features stay upstream of the payload-only adapter.
            feature_input = self.policy_features(raw_payload)
            if c.memory_aware_write:
                old = before.detach()
                observed = candidates.detach()
                occurrence = torch.stack([p.occurrence for p in state.publications.values()], 1)
                age = (state.events_seen[:, None]-occurrence).clamp_min(0)*owned_valid
                statistics = torch.stack((owned_valid.to(old.dtype), age.to(old.dtype).log1p(),
                    F.cosine_similarity(observed[:, :, :d], old[:, :, :d], dim=-1),
                    F.cosine_similarity(observed[:, :, d:], old[:, :, d:], dim=-1)), -1).detach()
                feature_input = feature_input+self.memory_policy_features(torch.cat((old, statistics), -1))
            features = feature_input.tanh()
            write_logits = (features*self.write_policy[None, :, :-1]).sum(-1)+self.write_policy[None, :, -1]
            if c.write_mode == 'competitive':
                scores = torch.cat((write_logits, write_logits.new_zeros(b, 1)), -1)
                probabilities = scores.softmax(-1)[:, :m]
                actions = F.one_hot(scores.argmax(-1), m+1)[:, :m].bool()
            else:
                probabilities = write_logits.sigmoid()
                actions = probabilities.ge(.5)
            bootstrap = torch.zeros(b, dtype=torch.bool, device=inputs.device)
            if write_override is not None:
                actions = torch.full_like(actions, write_override == 'always')
            elif c.bootstrap_empty:
                bootstrap = (~owned_valid).any(-1)
                first_empty = (~owned_valid).long().argmax(-1)
                recruited = F.one_hot(first_empty, m).bool()
                actions = torch.where(bootstrap[:, None], recruited, actions)
            eligible = torch.full_like(actions, write_override is None) & ~bootstrap[:, None]
            producer_state = producer['state']
            state_actions = torch.ones_like(actions)
            if c.state_commit == 'publication':
                state_actions = actions
                producer_state = commit_producer_state(state.producer, producer_state, actions,
                    probabilities if write_override is None and torch.is_grad_enabled() else None, eligible)
                state_masks.append(actions)
            bootstrap_masks.append(bootstrap)
            publications = {}
            for owner, name in enumerate(self.owner_names):
                old = state.publications[name]
                header = SemanticHeader(name, candidate_header[:, owner].reshape(b, 3, c.groups),
                                        producer['header'].group_ids.reshape(b, m, 3, c.active_groups)[:, owner],
                                        producer['header'].lace_ids.reshape(b, m, 3, c.active_laces)[:, owner])
                new = publish(old, candidate_payload[:, owner], header, actions[:, owner], state.epochs,
                              state.events_seen, torch.zeros_like(state.events_seen), ())
                if write_override is None and torch.is_grad_enabled():
                    tangent = ((probabilities[:, owner]-probabilities[:, owner].detach())*
                               eligible[:, owner])[:, None]
                    # Forward values exactly match publish, including true holds.
                    new = replace(new, payload=new.payload+tangent*(candidate_payload[:, owner]-old.payload),
                                  header=replace(new.header, memberships=new.header.memberships+
                                                 tangent[:, :, None]*(header.memberships-old.header.memberships)))
                publications[name] = new
            if capture:
                values = dict(read_weights=weights, read_versions=versions, read_valid=valid,
                              read_is_current=torch.full_like(active, control == 'fresh'),
                              write_probs=probabilities,
                              write_actions=actions, state_actions=state_actions, write_features=features, memory_before=before,
                              memory_candidates=candidates,
                              memory_versions=torch.stack([p.version for p in publications.values()], 1),
                              memory_valid=torch.stack([p.valid for p in publications.values()], 1),
                              write_eligible=eligible)
                if c.payload_rank:
                    values['payload_features'] = payload_features
                for name, value in values.items():
                    records[name].append(value)
            state = TaskMemoryState(ingress['state'], producer_state, composer['state'], publications,
                                    state.events_seen+1, state.epochs)
            aux = aux+(producer['aux']+composer['aux'])/2
            hat_aux = hat_aux+(ingress['aux']+producer['hat_aux']+composer['hat_aux'])/3
        result = dict(logits=torch.stack(logits, 1), state=state, aux=aux/t, hat_aux=hat_aux/t,
                      expert_executions=b*t*(m+1)*3*c.active_groups*c.experts_per_group,
                      external_events=b*t, source_occurrences=b*t, internal_rounds=b*t,
                      write_mode=c.write_mode, bootstrap_mask=torch.stack(bootstrap_masks, 1),
                      bootstrap_empty=c.bootstrap_empty, memory_aware_write=c.memory_aware_write,
                      state_commit=c.state_commit, producer_candidates=b*t*m,
                      producer_commits=int(torch.stack(state_masks, 1).sum()) if state_masks else b*t*m,
                      read_source='candidate' if control == 'fresh' else 'publication')
        if capture:
            result.update({key: torch.stack(parts, 1) for key, parts in records.items()})
            result['reset'] = (torch.zeros_like(inputs, dtype=torch.bool) if reset is None else
                               reset if reset.ndim == 2 else
                               torch.cat((reset[:, None], torch.zeros(b, t-1, dtype=torch.bool, device=inputs.device)), 1))
        return result
