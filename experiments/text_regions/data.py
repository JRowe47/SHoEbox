"""Ordered, independently owned stream rows with explicit document boundaries."""
from dataclasses import dataclass
import hashlib
import torch


@dataclass(frozen=True)
class StreamChunk:
    values: torch.Tensor
    targets: torch.Tensor
    reset: torch.Tensor


class OrderedByteStreams:
    def __init__(self, documents, batch_size=4):
        self.documents = tuple(bytes(d) for d in documents)
        if batch_size < 1 or len(self.documents) < batch_size or any(len(d) < 2 for d in self.documents):
            raise ValueError('Supply at least one two-byte document per batch row.')
        self.batch_size = batch_size
        digest = hashlib.sha256()
        for document in self.documents:
            digest.update(len(document).to_bytes(8, 'big'))
            digest.update(document)
        self.fingerprint = digest.hexdigest()
        self.document_ids = list(range(batch_size))
        self.positions = [0]*batch_size
        self.pending_reset = [True]*batch_size
        self.next_document = batch_size

    @classmethod
    def from_bytes(cls, data, batch_size=4):
        if len(data) < 2*batch_size:
            raise ValueError('Too few bytes to assign independent streams.')
        # No overlap and no target crossing a shard boundary. Each row is a
        # deliberately assigned independent stream, not a random window.
        edges = [len(data)*i//batch_size for i in range(batch_size+1)]
        return cls([data[a:b] for a, b in zip(edges, edges[1:])], batch_size)

    def next_chunk(self, length):
        if length < 1:
            raise ValueError('Chunk length must be positive.')
        x = torch.empty(self.batch_size, length, dtype=torch.long)
        y = torch.empty_like(x)
        reset = torch.zeros_like(x, dtype=torch.bool)
        for t in range(length):
            for row in range(self.batch_size):
                document = self.documents[self.document_ids[row]]
                if self.positions[row] >= len(document)-1:
                    self.document_ids[row] = self.next_document % len(self.documents)
                    self.next_document += 1
                    self.positions[row] = 0
                    self.pending_reset[row] = True
                    document = self.documents[self.document_ids[row]]
                pos = self.positions[row]
                x[row, t], y[row, t] = document[pos], document[pos+1]
                reset[row, t] = self.pending_reset[row]
                self.pending_reset[row] = False
                self.positions[row] += 1
        return StreamChunk(x, y, reset)

    def state_dict(self):
        return {'schema_version': 1, 'fingerprint': self.fingerprint, 'batch_size': self.batch_size,
                'document_ids': list(self.document_ids), 'positions': list(self.positions),
                'pending_reset': list(self.pending_reset), 'next_document': self.next_document}

    def load_state_dict(self, data):
        if data['schema_version'] != 1 or data['fingerprint'] != self.fingerprint or data['batch_size'] != self.batch_size:
            raise ValueError('Stream checkpoint dataset/schema/batch mismatch.')
        for field in ('document_ids', 'positions', 'pending_reset'):
            if len(data[field]) != self.batch_size:
                raise ValueError('Invalid cursor length.')
        if any(not 0 <= i < len(self.documents) for i in data['document_ids']):
            raise ValueError('Invalid document ID.')
        if any(not 0 <= p < len(self.documents[i]) for p, i in zip(data['positions'], data['document_ids'])):
            raise ValueError('Invalid document position.')
        self.document_ids = list(data['document_ids'])
        self.positions = list(data['positions'])
        self.pending_reset = list(data['pending_reset'])
        self.next_document = int(data['next_document'])


def split_bytes(raw):
    a, b = int(.9*len(raw)), int(.95*len(raw))
    return raw[:a], raw[a:b], raw[b:]
