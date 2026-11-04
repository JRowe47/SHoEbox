"""Exact (link, slot) bindings with common byte-valued static/live inputs."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ByteBatch:
    values: torch.Tensor
    links: torch.Tensor
    slots: torch.Tensor

    def validate(self, max_links, max_slots):
        if self.values.ndim != 2 or self.values.shape[1] == 0:
            raise ValueError('Expected a nonempty [batch, events] byte sequence.')
        for name, x, limit in [('values', self.values, 256),
                               ('links', self.links, max_links),
                               ('slots', self.slots, max_slots)]:
            if x.shape != self.values.shape or x.dtype != torch.long:
                raise ValueError(f'{name} must be int64 with the same [B,T] shape.')
            if x.device != self.values.device:
                raise ValueError('All event fields must share a device.')
            if bool(((x < 0) | (x >= limit)).any()):
                raise ValueError(f'{name} outside [0,{limit}).')
        if bool(((self.links == 0) & (self.slots != self.values)).any()):
            raise ValueError('Static byte-bank slot identity must equal its immutable byte value.')

    def slice(self, start, end=None):
        return ByteBatch(*(x[:, start:end] for x in (self.values, self.links, self.slots)))

    def to(self, device):
        return ByteBatch(*(x.to(device) for x in (self.values, self.links, self.slots)))

    @classmethod
    def static(cls, values):
        values = torch.as_tensor(values)
        if values.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError('Byte values must be integers; serialize floating point explicitly.')
        values = values.long()
        if values.ndim == 1: values = values[None]
        return cls(values, torch.zeros_like(values), values.clone())

    @classmethod
    def stream(cls, values, link=1, slot=0):
        if isinstance(values, (bytes, bytearray)): values = list(values)
        values = torch.as_tensor(values)
        if values.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError('Byte values must be integers; serialize floating point explicitly.')
        values = values.long()
        if values.ndim == 1: values = values[None]
        return cls(values, torch.full_like(values, link), torch.full_like(values, slot))

    @classmethod
    def text(cls, text):
        return cls.static(list(text.encode('utf-8')))


@dataclass(frozen=True)
class SlotValue:
    data: bytes
    version: int


class SlotDictionary:
    """Link 0 is the immutable 256-byte bank; other links hold published bytes.

    No modality dispatch. Each read materializes a snapshot, so a subsequent
    publish cannot modify events which have already been handed to the model.
    The dictionary stores current values only, not a history/KV cache.
    """
    def __init__(self, max_links=8, max_slots=256):
        if max_links < 2 or max_slots < 256:
            raise ValueError('Need a byte bank and at least one live link.')
        self.max_links, self.max_slots = max_links, max_slots
        self._live = {}

    def _check(self, link, slot):
        if not isinstance(link, int) or not isinstance(slot, int):
            raise TypeError('Link and slot are exact integer identities.')
        if not 0 <= link < self.max_links or not 0 <= slot < self.max_slots:
            raise ValueError('Address outside configured capacity.')

    def publish(self, link, slot, data):
        self._check(link, slot)
        if link == 0: raise ValueError('The byte dictionary is immutable.')
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError('Publish bytes or bytearray; serialize numeric values explicitly.')
        if not data: raise ValueError('An event payload cannot be empty.')
        old = self._live.get((link, slot))
        item = SlotValue(bytes(data), 1 if old is None else old.version + 1)
        self._live[link, slot] = item
        return item.version

    def read(self, link, slot):
        self._check(link, slot)
        if link == 0:
            if slot >= 256: raise KeyError((link, slot))
            return SlotValue(bytes([slot]), 0)
        return self._live[link, slot]

    def events(self, addresses):
        values, links, slots, versions = [], [], [], []
        for link, slot in addresses:
            item = self.read(link, slot)
            values.extend(item.data)
            links.extend([link] * len(item.data))
            slots.extend([slot] * len(item.data))
            versions.extend([item.version] * len(item.data))
        if not values: raise ValueError('At least one nonempty address is required.')
        events = ByteBatch(*(torch.tensor(x, dtype=torch.long)[None]
                            for x in (values, links, slots)))
        return events, torch.tensor(versions, dtype=torch.long)[None]
