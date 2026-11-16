"""EXP00: balanced, grouped byte fixtures with an independent interpreter.

Calibration folds hold out concrete worlds. Query order, presentation order,
padding, and explicitly renamed counterfactuals of one world share a fold.
Atomic fixtures do not claim unseen-program-structure generalization.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import random

from experiments.mad.tasks import (ANSWER, BOS, COPY, END, KEY, MAP, PAD,
                                   QUERY, SEP, SET, VALUE, Episode, TaskSpec)

SWAP = 14
TASKS = ('binding', 'initialize', 'set', 'copy', 'swap', 'one_hop')


@dataclass(frozen=True)
class Spec:
    task: str
    vocab: int = 8
    items: int = 2
    key_pool: int = 48
    answer_offset: int = 0

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError('unknown EXP00 task')
        if any(type(getattr(self, k)) is not int for k in ('vocab', 'items', 'key_pool', 'answer_offset')):
            raise ValueError('integer fields required')
        if not 2 <= self.vocab <= 64 or not 2 <= self.items <= 4:
            raise ValueError('vocab 2..64 and items 2..4 required')
        if not self.items <= self.key_pool <= 64 or not 0 <= self.answer_offset < 64:
            raise ValueError('invalid key pool or offset')
        if self.task != 'one_hop' and self.items > self.vocab:
            raise ValueError('distinct initial values require items <= vocab')
        if self.task not in ('binding', 'one_hop') and self.key_pool != self.items:
            raise ValueError('atomic register tasks use explicit register names 0..items-1')
        if self.task == 'one_hop' and (self.items < 3 or self.key_pool != self.vocab):
            raise ValueError('one-hop graphs need >=3 nodes and key_pool == vocab')

    def native(self):
        if self.task == 'one_hop':
            return TaskSpec('pointer_chasing', vocab=self.vocab, length=self.items, depth=1)
        if self.task == 'binding':
            return TaskSpec('recall', vocab=self.vocab, length=self.items)
        return TaskSpec('variable_tracking', vocab=self.vocab, length=1)


@dataclass(frozen=True)
class World:
    task: str
    records: tuple
    operation: tuple = ()

    @property
    def group_key(self):
        # All operations on one initial state stay in one fold, including the
        # operation-counterfactual audit. These are concrete-world folds, not
        # a claim of holding out every isomorphic renaming of an abstract task.
        return json.dumps([self.task, sorted(self.records)], separators=(',', ':')).encode()

    @property
    def content_key(self):
        operation = self.operation
        if operation and operation[0] == SWAP:
            operation = (SWAP, *sorted(operation[1:]))
        return self.group_key+b'\0'+json.dumps(operation).encode()


def fold(group_key):
    bucket = int.from_bytes(hashlib.sha256(b'EXP00-v2\0'+group_key).digest()[:8], 'big') % 10
    return 'train' if bucket < 6 else 'dev' if bucket < 8 else 'test'


def sample_world(spec, rng):
    keys = (rng.sample(range(spec.key_pool), spec.items) if spec.task in ('binding', 'one_hop')
            else list(range(spec.items)))
    if spec.task == 'one_hop':
        return World(spec.task, tuple((k, rng.choice([v for v in keys if v != k])) for k in keys))
    values = rng.sample(range(spec.vocab), spec.items)
    records = tuple(zip(keys, values))
    operation = ()
    if spec.task in ('set', 'copy', 'swap'):
        a = rng.randrange(spec.items)
        b = (rng.choice([v for v in range(spec.vocab) if v != values[a]]) if spec.task == 'set'
             else rng.choice([k for k in keys if k != a]))
        operation = (dict(set=SET, copy=COPY, swap=SWAP)[spec.task], a, b)
    return World(spec.task, records, operation)


def generated_state(world):
    """Generator semantics; the independent byte interpreter below does not call this."""
    state = dict(world.records)
    if world.operation:
        op, a, b = world.operation
        if op == SET:
            state[a] = b
        elif op == COPY:
            state[a] = state[b]
        else:
            state[a], state[b] = state[b], state[a]
    return state


def encode(world, spec, record_order=None, query_order=None, *, group_key=None):
    records = list(world.records)
    if record_order is not None:
        records = [records[i] for i in record_order]
    queries = list(query_order) if query_order is not None else [k for k, _ in world.records]
    raw, mask, answers = [BOS, 16+TASKS.index(spec.task)], [0, 0], []
    state = generated_state(world)
    def put(xs, scored=False):
        raw.extend(xs)
        mask.extend([int(scored)]*len(xs))
    for key, value in records:
        if spec.task in ('binding', 'one_hop'):
            put([MAP, KEY+key, SEP, (KEY if spec.task == 'one_hop' else VALUE)+value, END])
        else:
            put([SET, KEY+key, VALUE+value, END])
    if world.operation:
        op, a, b = world.operation
        put([op, KEY+a, (VALUE if op == SET else KEY)+b, END])
    for key in queries:
        put([QUERY, KEY+key, SEP])
        put([PAD]*((spec.answer_offset-len(raw)-1) % 64))
        put([ANSWER])
        put([VALUE+state[key]], True)
        put([END])
        answers.append(state[key])
    return Episode(spec.task, bytes(raw), bytes(mask), tuple(answers), group_key or world.group_key)


def variants(world, spec, rng):
    """Cartesian cyclic shifts balance record/query positions independently."""
    order = list(range(spec.items))
    queries = [k for k, _ in world.records]
    rng.shuffle(order)
    rng.shuffle(queries)
    return [encode(world, spec, order[i:]+order[:i], queries[j:]+queries[:j])
            for i in range(spec.items) for j in range(spec.items)]


def parse(ep):
    """Decode observed record syntax only; never execute instructions or inspect answers."""
    data = ep.data
    if len(data) != len(ep.mask) or data[:1] != bytes([BOS]):
        raise ValueError('bad episode')
    records, queries, i, queried = [], [], 2, False
    while i < len(data):
        op = data[i]
        if op == PAD:
            i += 1
        elif op == MAP:
            if queried or data[i+2] != SEP or data[i+4] != END:
                raise ValueError('bad/future MAP record')
            records.append((op, data[i+1]-KEY, data[i+3]))
            i += 5
        elif op in (SET, COPY, SWAP):
            if queried or data[i+3] != END:
                raise ValueError('bad/future operation record')
            records.append((op, data[i+1]-KEY, data[i+2]))
            i += 4
        elif op == QUERY:
            if data[i+2] != SEP:
                raise ValueError('bad query')
            queries.append(data[i+1]-KEY)
            queried = True
            i += 3
            while data[i] == PAD:
                i += 1
            if data[i] != ANSWER or data[i+2] != END or not ep.mask[i+1]:
                raise ValueError('bad answer framing')
            # Answer payload is intentionally not read.
            i += 3
        else:
            raise ValueError(f'unknown token {op}')
    return records, queries


def solve(ep, spec):
    """Independent emitted-byte interpreter: simultaneous updates use copied arrays."""
    records, queries = parse(ep)
    state = [None]*spec.key_pool
    for op, key, arg in records:
        previous = state.copy()
        if op == MAP:
            state[key] = arg-(KEY if spec.task == 'one_hop' else VALUE)
        elif op == SET:
            state[key] = arg-VALUE
        elif op == COPY:
            state[key] = previous[arg-KEY]
        elif op == SWAP:
            state[key], state[arg-KEY] = previous[arg-KEY], previous[key]
    return tuple(state[k] for k in queries)


def dataset(spec, count, split, seed=20261115):
    """Return >=count unique cases in complete balanced families (rounded up)."""
    if count < 1 or split not in ('train', 'dev', 'test'):
        raise ValueError('positive count and valid split required')
    rng = random.Random(seed)
    result, seen = [], set()
    for _ in range(max(10000, count*1000)):
        world = sample_world(spec, rng)
        key = world.group_key
        if world.content_key in seen or fold(key) != split:
            continue
        seen.add(world.content_key)
        family = variants(world, spec, rng)
        if any(solve(ep, spec) != ep.answer for ep in family):
            raise AssertionError('interpreter disagrees with generator')
        result.extend(family)
        if len(result) >= count:
            return result
    raise ValueError('not enough unique worlds for the requested complete families')


def fingerprint(cases):
    h = hashlib.sha256()
    for ep in cases:
        h.update(ep.data+ep.mask+ep.split_key)
    return h.hexdigest()


def reencode(ep, spec, *, offset=None, reverse_queries=False, rename=None, change_operation=False):
    records, queries = parse(ep)
    initial, operation = [], ()
    for op, key, arg in records:
        if len(initial) < spec.items:
            initial.append((key, arg-(KEY if spec.task == 'one_hop' else VALUE)))
        else:
            operation = (op, key, arg-(VALUE if op == SET else KEY))
    if change_operation:
        if not operation:
            raise ValueError('operation counterfactual requires an atomic update')
        op, a, b = operation
        if op == SET:
            b = (b+1) % spec.vocab
        elif op == COPY:
            a, b = b, a
        else:
            # Reversing a swap's arguments is semantically identical. Change
            # its partner instead, staying within the trained operator family.
            b = next(k for k, _ in initial if k not in (a, b))
        operation = (op, a, b)
    if rename:
        initial = [(rename[k], rename[v] if spec.task == 'one_hop' else v) for k, v in initial]
        queries = [rename[k] for k in queries]
        if operation:
            op, a, b = operation
            operation = (op, rename[a], b if op == SET else rename[b])
    if reverse_queries:
        queries.reverse()
    world = World(spec.task, tuple(initial), operation)
    return encode(world, replace(spec, answer_offset=offset) if offset is not None else spec,
                  query_order=queries, group_key=ep.split_key)


def shortcuts(cases, spec):
    """Predictions use observed records only. Ground truth is used only to score."""
    totals = {}
    for ep in cases:
        records, queries = parse(ep)
        initial = records[:spec.items]
        supplied = [v-(KEY if spec.task == 'one_hop' else VALUE) for _, _, v in initial]
        majority = min(set(supplied), key=lambda v: (-supplied.count(v), v))
        preds = {'majority_supplied': [majority]*len(queries),
                 'input_order': supplied[:len(queries)],
                 'last_supplied': [supplied[-1]]*len(queries)}
        if spec.task in ('set', 'copy', 'swap'):
            original = {k: v-VALUE for _, k, v in initial}
            preds['ignore_updates'] = [original[k] for k in queries]
        for name, values in preds.items():
            correct = [a == b for a, b in zip(values, ep.answer)]
            row = totals.setdefault(name, [0., 0., 0])
            row[0] += sum(correct)/len(correct)
            row[1] += all(correct)
            row[2] += 1
    result = {name: {'token_accuracy': a/n, 'exact_accuracy': b/n, 'kind': 'deterministic_shortcut'}
              for name, (a, b, n) in totals.items()}
    result['uniform_chance'] = {'token_accuracy': 1/spec.vocab,
                               'exact_accuracy': (1/spec.vocab)**spec.items,
                               'kind': 'expected_independent_guesses'}
    if spec.task == 'one_hop':
        result['node_set_guess'] = {'token_accuracy': 1/(spec.items-1),
                                   'exact_accuracy': (1/(spec.items-1))**spec.items,
                                   'kind': 'expected_independent_nonstart_node_guesses'}
    return result


def exhaustive_oracles():
    count = 0
    for task in TASKS:
        spec = Spec(task, vocab=3 if task == 'one_hop' else 2,
                    items=3 if task == 'one_hop' else 2, key_pool=3 if task == 'one_hop' else 2)
        if task == 'one_hop':
            worlds = [World(task, tuple(enumerate(v))) for v in itertools.product(*[[j for j in range(3) if j != i] for i in range(3)])]
        else:
            worlds = []
            for vals in itertools.product(range(2), repeat=2):
                ops = [()]
                if task == 'set':
                    ops = [(SET, a, v) for a in range(2) for v in range(2)]
                elif task in ('copy', 'swap'):
                    ops = [(COPY if task == 'copy' else SWAP, a, 1-a) for a in range(2)]
                worlds += [World(task, tuple(enumerate(vals)), op) for op in ops]
        for world in worlds:
            for offset in range(64):
                for ep in variants(world, replace(spec, answer_offset=offset), random.Random(47)):
                    if solve(ep, spec) != ep.answer:
                        raise AssertionError('exhaustive oracle mismatch')
                    count += 1
    return count
