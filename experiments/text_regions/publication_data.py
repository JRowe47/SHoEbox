"""Procedural single-producer acquisition tasks; labels are loss-only.

Bare A-D distractors and invalid '=x' strings prevent recognizing values alone
from solving the write decision. Per-row random timing prevents a clock rule.
"""
from dataclasses import dataclass
import hashlib
import random
import torch


@dataclass(frozen=True)
class PublicationBatch:
    inputs: torch.Tensor
    reset: torch.Tensor
    available: torch.Tensor
    writes: torch.Tensor
    retained: torch.Tensor
    answers: torch.Tensor
    old_answers: torch.Tensor
    identities: tuple[str, ...]


def causal_labels(inputs, reset, available):
    current = torch.full_like(inputs[:, 0], -100)
    previous = torch.full_like(current, -100)
    writes, retained = [], []
    for t in range(inputs.shape[1]):
        current = current.masked_fill(reset[:, t], -100)
        previous = previous.masked_fill(reset[:, t], -100)
        value = inputs[:, t]
        write = available[:, t] & previous.eq(61) & value.ge(65) & value.le(68)
        current = torch.where(write, value, current)
        writes.append(write)
        retained.append(current)
        # Missing input cannot act as an observed marker on the next step.
        previous = torch.where(available[:, t], value, -100)
    return torch.stack(writes, 1), torch.stack(retained, 1)


def make_batch(split, start, count, seed, gap=16, kind='correction', control='carried'):
    if split not in ('train', 'validation', 'test') or kind not in ('recall', 'correction'):
        raise ValueError('Unknown split or task.')
    if control not in ('carried', 'withheld', 'reset', 'no_writes', 'absent'):
        raise ValueError('Unknown control.')
    if min(start, gap) < 0 or count < 1:
        raise ValueError('Invalid batch dimensions.')
    rows, identities, old_answers, answers = [], [], [], []
    for index in range(start, start + count):
        identity = {'train': 't', 'validation': 'v', 'test': 'e'}[split] + f'{index:06x}'
        digest = hashlib.sha256(f'{seed}/{split}/{index}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        old = rng.randrange(65, 69)
        new = rng.choice([v for v in range(65, 69) if v != old]) if kind == 'correction' else old
        def noise(length):
            pieces = bytearray()
            while len(pieces) < length:
                token = rng.choice((b'A', b'B', b'C', b'D', b'x', b'?', b':', b'.', b'=x'))
                pieces.extend(token if len(pieces)+len(token) <= length else b'.')
            return pieces
        data = noise(rng.randrange(8))
        spans = []
        first = len(data)
        data.extend(identity.encode()+b'='+bytes([old])+b';')
        spans.append((first, len(data)))
        data.extend(noise(gap//2+rng.randrange(4)))
        second = len(data)
        if kind == 'correction':
            data.extend(identity.encode()+b'='+bytes([new])+b';')
            spans.append((second, len(data)))
        else:
            data.extend(noise(len(identity)+3))
        data.extend(noise(gap))
        # Constant length within a batch, with independent write positions.
        length = 7+2*(len(identity)+3)+gap//2+3+gap
        data.extend(noise(length-len(data)))
        data.append(ord('!'))
        if control == 'withheld' and kind == 'correction':
            a, b = spans[-1]; data[a:b] = b'.'*(b-a)
        elif control == 'no_writes':
            for a, b in spans: data[a:b] = b'.'*(b-a)
        rows.append(list(data)); identities.append(identity)
        old_answers.append(old)
        answers.append(old if control == 'withheld' else new)
    inputs = torch.tensor(rows, dtype=torch.long)
    reset = torch.zeros_like(inputs, dtype=torch.bool); reset[:, 0] = True
    if control == 'reset': reset[:, -1] = True
    available = torch.full_like(reset, control != 'absent')
    writes, retained = causal_labels(inputs, reset, available)
    return PublicationBatch(inputs, reset, available, writes, retained,
                            torch.tensor(answers), torch.tensor(old_answers), tuple(identities))


def batch_digest(batch):
    digest = hashlib.sha256()
    for name in ('inputs', 'reset', 'available', 'writes', 'retained', 'answers', 'old_answers'):
        tensor = getattr(batch, name)
        digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
