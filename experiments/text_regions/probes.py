"""Seeded episode fixtures; capability training/evaluation is a later milestone."""
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import random

SPLIT_SEEDS = {'train': 1103, 'validation': 4409, 'test': 7919}


@dataclass(frozen=True)
class Episode:
    kind: str
    split: str
    seed: int
    views: dict[str, str]
    prompt: str
    answer: str
    private_answers: dict[str, str]

    def to_dict(self):
        return asdict(self)


def make_episode(kind='recall', split='train', index=0, distractors=16):
    if split not in SPLIT_SEEDS or kind not in ('recall', 'shared_private', 'composition'):
        raise ValueError('Unknown probe or split.')
    if index < 0 or distractors < 0:
        raise ValueError('Nonnegative episode index and distractor count required.')
    seed = SPLIT_SEEDS[split] + index*104729
    rng = random.Random(seed)
    # Split-specific, fresh exact identities prevent cross-split identifier reuse.
    ids = [f'{split[0]}{index}x{n:04x}' for n in rng.sample(range(65536), 8+distractors)]
    if kind == 'recall':
        entity, old, updated = ids[:3]
        noise = [f'{name}={rng.randrange(1000):03d}.\n' for name in ids[8:]]
        rng.shuffle(noise)
        middle = distractors//2
        text = f'{entity}={old}.\n' + ''.join(noise[:middle])
        text += f'Update {entity}={updated}.\n' + ''.join(noise[middle:])
        return Episode(kind, split, seed, {'A': text, 'B': text, 'C': text},
                       f'Now {entity}=', updated, {})
    if kind == 'shared_private':
        who, shared, left, right = ids[:4]
        a = f'{who} belongs to {shared}. Its mark is {left}.\n'
        b = f'Group {shared} contains {who}. Its seal is {right}.\n'
        return Episode(kind, split, seed, {'A': a, 'B': b, 'C': ''},
                       f'Group of {who}: ', shared, {'A': left, 'B': right})
    a, b, c, d = ids[:4]
    views = {'A': f'{a} -> {b}.\n', 'B': f'{b} -> {c}.\n', 'C': f'{c} -> {d}.\n'}
    # Randomize which view carries each relation while preserving a declared answer.
    shuffled = list(views.values())
    rng.shuffle(shuffled)
    return Episode(kind, split, seed, dict(zip(views, shuffled)), f'Three hops from {a}: ', d, {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=512)
    p.add_argument('--split', choices=tuple(SPLIT_SEEDS), default='validation')
    p.add_argument('--kind', choices=('recall', 'shared_private', 'composition'), default='recall')
    p.add_argument('--distractors', type=int, default=16)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as out:
        for i in range(args.count):
            out.write(json.dumps(make_episode(args.kind, args.split, i, args.distractors).to_dict())+'\n')


if __name__ == '__main__':
    main()
