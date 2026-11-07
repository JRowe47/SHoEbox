"""All carried values live here, separately from the trainable modules."""
from dataclasses import dataclass, field, replace
import torch
from ..hat_minkron.hat import HATState
from .ports import TensorPorts, Publication, map_tensors, tensor_list, plain_dict, choose_rows


@dataclass(frozen=True)
class NodeState:
    contextualizer: tuple[torch.Tensor, ...]
    blocks: tuple[torch.Tensor, ...]
    clock: torch.Tensor

    def reset(self, rows):
        return map_tensors(self, lambda x: choose_rows(rows, torch.zeros_like(x), x))


@dataclass(frozen=True)
class GraphState:
    ingress: dict[str, HATState]
    nodes: dict[str, NodeState]
    ports: TensorPorts
    events_seen: torch.Tensor
    rounds_seen: torch.Tensor
    epochs: torch.Tensor
    input_mode: str = ''
    recall_feedback: dict[str, torch.Tensor] = field(default_factory=dict)
    publication_demand: dict[str, torch.Tensor] = field(default_factory=dict)

    def detach(self):
        return map_tensors(self, lambda x: x.detach())

    def clone(self):
        return map_tensors(self, lambda x: x.clone())

    def to(self, device):
        return map_tensors(self, lambda x: x.to(device))

    def reset(self, rows):
        if rows.dtype != torch.bool or rows.shape != self.events_seen.shape or rows.device != self.events_seen.device:
            raise ValueError('Reset mask must be bool [B] on the state device.')
        return replace(self, ingress={k: h.reset(rows) for k, h in self.ingress.items()},
                       nodes={k: n.reset(rows) for k, n in self.nodes.items()},
                       ports=self.ports.reset(rows), events_seen=self.events_seen*(~rows),
                       rounds_seen=self.rounds_seen*(~rows), epochs=self.epochs+rows.long(),
                       recall_feedback={k: choose_rows(rows, torch.zeros_like(v), v)
                                        for k, v in self.recall_feedback.items()},
                       publication_demand={k: choose_rows(rows, torch.zeros_like(v), v)
                                           for k, v in self.publication_demand.items()})

    def memory_tensor(self, owners):
        return torch.stack([torch.cat((self.ports.latest[n].payload,
            self.ports.latest[n].header.memberships.flatten(1)), -1) for n in owners], 1)

    def with_memory_tensor(self, owners, memory):
        if memory.shape != self.memory_tensor(owners).shape:
            raise ValueError('Graph memory tensor shape mismatch.')
        pubs = dict(self.ports.latest)
        for i, name in enumerate(owners):
            pub = pubs[name]
            width = pub.payload.shape[-1]
            pubs[name] = replace(pub, payload=memory[:, i, :width], header=replace(pub.header,
                memberships=memory[:, i, width:].reshape_as(pub.header.memberships)))
        return replace(self, ports=TensorPorts(pubs))

    def tensors(self):
        return tensor_list(self)

    def storage_bytes(self):
        # Shared ID tensors can occur in multiple dependency records. Count their
        # allocated storage once, including noncontiguous tensor storage correctly.
        unique = {(str(x.device), x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
                  for x in self.tensors()}
        return sum(unique.values())

    def tensor_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.tensors())

    def to_dict(self):
        return plain_dict(self.detach())

    @classmethod
    def from_dict(cls, data):
        return cls(ingress={k: HATState(**h) for k, h in data['ingress'].items()},
                   nodes={k: NodeState(tuple(n['contextualizer']), tuple(n['blocks']), n['clock'])
                          for k, n in data['nodes'].items()},
                   ports=TensorPorts({k: Publication.from_dict(p) for k, p in data['ports']['latest'].items()}),
                   events_seen=data['events_seen'], rounds_seen=data['rounds_seen'], epochs=data['epochs'],
                   input_mode=data['input_mode'], recall_feedback=data.get('recall_feedback', {}),
                   publication_demand=data.get('publication_demand', {}))
