"""Two learned relation lookups, split by whole pairs of permutation tables.

B observes f, C observes g. After independent encoding, A receives x and the
network predicts g(f(x)). Inputs are four-byte tables in A/B/C/D key order.
No intermediate answers or task labels are supplied to the network.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import random
import torch

SYMBOLS = tuple(b'ABCD')
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def pair_split(first, second):
    value = hashlib.sha256(f'composition-v1:{first}:{second}'.encode()).digest()[0] % 4
    return ('test', 'validation', 'train', 'train')[value]


@lru_cache(None)
def pairs(split):
    if split not in ('train', 'validation', 'test', 'all'):
        raise ValueError('Unknown split.')
    return tuple((a, b) for a in range(24) for b in range(24)
                 if split == 'all' or pair_split(a, b) == split)


@dataclass(frozen=True)
class CompositionBatch:
    keys: tuple[tuple[int, int, int], ...]
    first: torch.Tensor
    second: torch.Tensor
    query: torch.Tensor
    answers: torch.Tensor

    @property
    def size(self):
        return len(self.keys)


def batch_from_keys(keys):
    keys = tuple(tuple(int(v) for v in key) for key in keys)
    if not keys or any(len(k) != 3 or not 0 <= k[0] < 24 or not 0 <= k[1] < 24
                       or not 0 <= k[2] < 4 for k in keys):
        raise ValueError('Expected nonempty (first permutation, second permutation, query) keys.')
    first = torch.tensor([PERMUTATIONS[a] for a, _, _ in keys], dtype=torch.long)
    second = torch.tensor([PERMUTATIONS[b] for _, b, _ in keys], dtype=torch.long)
    query = torch.tensor([q for _, _, q in keys], dtype=torch.long)
    intermediate = first.gather(1, query[:, None])
    answers = second.gather(1, intermediate).flatten() + SYMBOLS[0]
    return CompositionBatch(keys, first + SYMBOLS[0], second + SYMBOLS[0],
                            query[:, None] + SYMBOLS[0], answers)


def training_batch(rng, batch_size, lookup_warmup=False):
    if batch_size < 4 or batch_size % 4:
        raise ValueError('Training batch must be a positive multiple of four.')
    candidates = tuple((a,b) for a,b in pairs('train') if a == 0 or b == 0) if lookup_warmup else pairs('train')
    keys = []
    for _ in range(batch_size // 4):
        a, b = rng.choice(candidates)
        order = list(range(4)); rng.shuffle(order)
        keys.extend((a, b, q) for q in order)
    rng.shuffle(keys)
    return batch_from_keys(keys)


def evaluation_keys(split='validation', count=None, seed=819):
    # Every case is unique. All four queries per pair balance the answer bytes.
    groups = list(pairs(split)); random.Random(seed).shuffle(groups)
    keys = [(a, b, q) for a, b in groups for q in range(4)]
    if count is not None:
        if count < 4 or count % 4 or count > len(keys):
            raise ValueError(f'Count must be a multiple of four in [4, {len(keys)}].')
        keys = keys[:count]
    return tuple(keys)


def shuffled_bridge(batch):
    # A deterministic permutation of whole B tables; independent of answers.
    source = list(batch.keys)
    return batch_from_keys([(source[(i + 4) % len(source)][0], b, q)
                            for i, (_, b, q) in enumerate(source)])


def coverage():
    return {split: {'table_pairs': len(pairs(split)), 'unique_queries': 4*len(pairs(split)),
                    'first_tables': len({a for a, _ in pairs(split)}),
                    'second_tables': len({b for _, b in pairs(split)})}
            for split in ('train', 'validation', 'test', 'all')}
