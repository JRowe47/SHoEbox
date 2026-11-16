"""Small, deterministic byte tasks with independently checked answers.

This is a SHoEbox adaptation, not the original MAD dataset/leaderboard.
The fixed byte encoding is shared by every model and contains no task solver.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

TASKS = ("recall", "noisy_recall", "fuzzy_recall", "selective_copy",
         "compression", "memorization", "variable_tracking", "composition",
         "pointer_chasing")
BOS, MAP, SEP, END, QUERY, ANSWER, KEEP, AGG, PAD = range(1, 10)
SET, COPY, ADD, XOR = range(10, 14)
VALUE, KEY, NOISE = 32, 96, 160


@dataclass(frozen=True)
class TaskSpec:
    task: str
    vocab: int = 8
    length: int = 4
    delay: int = 0
    noise: float = 0.0
    key_width: int = 2
    value_width: int = 2
    depth: int = 2

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"unknown task: {self.task}")
        for key in ("vocab", "length", "delay", "key_width", "value_width", "depth"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not 2 <= self.vocab <= 64 or not 1 <= self.length <= 64 or not 0 <= self.delay <= 4096:
            raise ValueError("require vocab 2..64, length 1..64, delay 0..4096")
        if not 0 <= self.noise <= .95 or not 1 <= self.key_width <= 4 or not 1 <= self.value_width <= 4:
            raise ValueError("invalid noise or fuzzy widths")
        if self.task in ("recall", "noisy_recall", "pointer_chasing") and self.length > self.vocab:
            raise ValueError("unique keys require length <= vocab")
        if self.task == "pointer_chasing" and not 1 <= self.depth < self.length:
            raise ValueError("pointer depth must be between 1 and length-1")
        if self.task == "fuzzy_recall" and self.length > sum(self.vocab**i for i in range(1, self.key_width+1)):
            raise ValueError("not enough unique fuzzy keys")
        if self.task == "composition" and self.vocab & (self.vocab-1):
            raise ValueError("xor composition requires a power-of-two vocabulary")


@dataclass(frozen=True)
class Episode:
    task: str
    data: bytes
    mask: bytes
    answer: tuple
    split_key: bytes

    @property
    def digest(self):
        return hashlib.sha256(self.data + self.mask + bytes(self.answer)).hexdigest()


def fixed_table(spec, seed):
    rng = random.Random(seed ^ 0x4D454D)
    # Balanced, fixed arbitrary facts. Every training set covers every key.
    values = list(range(spec.vocab))
    rng.shuffle(values)
    return values


def _candidate(spec, rng, table, ordinal):
    s = spec
    raw = [BOS, 16 + TASKS.index(s.task)]
    mask = [0, 0]
    answers = []

    def put(values, scored=False):
        values = list(values)
        for value in values:
            if value == ANSWER:
                # A query completes a real L3 constituent before its answer.
                # Alignment bytes are observed and charged to the byte budget.
                padding = (-len(raw)-1) % 64
                raw.extend([PAD]*padding)
                mask.extend([0]*padding)
            raw.append(value)
            mask.append(int(scored))
        if scored:
            answers.extend(v-VALUE for v in values)

    def noise(n):
        put(NOISE+rng.randrange(64) for _ in range(n))

    def distract():
        for _ in range(4):
            while rng.random() < s.noise:
                noise(1)

    structure = None
    if s.task in ("recall", "noisy_recall", "fuzzy_recall"):
        keys, values = [], []
        while len(keys) < s.length:
            width = rng.randint(1, s.key_width) if s.task == "fuzzy_recall" else 1
            k = tuple(KEY+rng.randrange(s.vocab) for _ in range(width))
            if k not in keys:
                keys.append(k)
                width = rng.randint(1, s.value_width) if s.task == "fuzzy_recall" else 1
                values.append(tuple(VALUE+rng.randrange(s.vocab) for _ in range(width)))
        for k, v in zip(keys, values):
            distract()
            put([MAP, *k, SEP, *v, END])
        noise(s.delay)
        order = list(range(len(keys)))
        rng.shuffle(order)
        for i in order:
            put([QUERY, *keys[i], SEP, ANSWER])
            put(values[i], True)
            put([END])
    elif s.task in ("selective_copy", "compression"):
        values = [VALUE+rng.randrange(s.vocab) for _ in range(s.length)]
        for v in values:
            if s.task == "selective_copy":
                distract()
                put([KEEP, v])
            else:
                put([v])
        noise(s.delay)
        if s.task == "compression":
            # The last byte completes an L3 constituent. The decoder sees only
            # its single 64-float state, never any input or retained-memory slot.
            put([PAD]*((-len(raw)-1) % 64))
            put([AGG])
            answers = [v-VALUE for v in values]
            structure = bytes(values)
        else:
            put([ANSWER])
            put(values, True)
            put([END])
    elif s.task == "memorization":
        noise(12 + s.delay)
        k = ordinal % s.vocab
        put([QUERY, KEY+k, SEP, ANSWER])
        put([VALUE+table[k]], True)
        put([END])
    elif s.task in ("variable_tracking", "composition"):
        state = [rng.randrange(s.vocab) for _ in range(4)]
        for i, v in enumerate(state):
            put([SET, KEY+i, VALUE+v, END])
        ops = []
        choices = (SET, COPY) if s.task == "variable_tracking" else (COPY, ADD, XOR)
        for _ in range(s.length):
            op, a = rng.choice(choices), rng.randrange(4)
            b = rng.randrange(s.vocab) if op == SET else rng.choice([i for i in range(4) if i != a])
            ops.append((op, a, -1 if op == SET else b))
            put([op, KEY+a, (VALUE if op == SET else KEY)+b, END])
            if op == SET:
                state[a] = b
            elif op == COPY:
                state[a] = state[b]
            elif op == ADD:
                state[a] = (state[a]+state[b]) % s.vocab
            else:
                state[a] ^= state[b]
            distract()
        noise(s.delay)
        q = rng.randrange(4)
        put([QUERY, KEY+q, SEP, ANSWER])
        put([VALUE+state[q]], True)
        put([END])
        # Disjoint instruction/argument structures, excluding literal values.
        structure = json.dumps([ops, q], separators=(",", ":")).encode()
    else:
        nodes = rng.sample(range(s.vocab), s.length)
        edges = list(zip(nodes, nodes[1:]+nodes[:1]))
        rng.shuffle(edges)
        for a, b in edges:
            distract()
            put([MAP, KEY+a, SEP, KEY+b, END])
        noise(s.delay)
        start = rng.randrange(s.length)
        put([QUERY, KEY+nodes[start], SEP, VALUE+s.depth, ANSWER])
        put([VALUE+nodes[(start+s.depth) % s.length]], True)
        put([END])
    # Never use an answer to assign an in-context example to its split.
    if structure is None:
        structure = bytes(v for v, m in zip(raw, mask) if not m)
    return Episode(s.task, bytes(raw), bytes(mask), tuple(answers), structure)


def solve(ep, spec, table):
    """Parse the emitted input, independently of generation and stored labels."""
    data = list(ep.data)
    i, mapping, selected, state, output = 2, {}, [], {}, []
    if ep.task == "compression":
        return tuple(x-VALUE for x in data[2:] if VALUE <= x < VALUE+spec.vocab)
    while i < len(data):
        token = data[i]
        if token >= NOISE or token in (END, PAD):
            i += 1
        elif token == MAP:
            j = data.index(SEP, i+1)
            k = data.index(END, j+1)
            mapping[tuple(data[i+1:j])] = tuple(data[j+1:k])
            i = k+1
        elif token == KEEP:
            selected.append(data[i+1]-VALUE)
            i += 2
        elif token in (SET, COPY, ADD, XOR):
            a, b = data[i+1]-KEY, data[i+2]
            if token == SET:
                state[a] = b-VALUE
            elif token == COPY:
                state[a] = state[b-KEY]
            elif token == ADD:
                state[a] = (state[a]+state[b-KEY]) % spec.vocab
            else:
                state[a] ^= state[b-KEY]
            if data[i+3] != END:
                raise ValueError("bad operation record")
            i += 4
        elif token == QUERY:
            j = data.index(SEP, i+1)
            key = tuple(data[i+1:j])
            if ep.task == "memorization":
                expected = (table[key[0]-KEY],)
            elif ep.task in ("variable_tracking", "composition"):
                expected = (state[key[0]-KEY],)
            elif ep.task == "pointer_chasing":
                depth = data[j+1]-VALUE
                for _ in range(depth):
                    key = mapping[key]
                expected = (key[0]-KEY,)
            else:
                expected = tuple(x-VALUE for x in mapping[key])
            a = data.index(ANSWER, j+1)
            end = data.index(END, a+1)
            if end-a-1 != len(expected) or any(not ep.mask[k] for k in range(a+1, end)):
                raise ValueError("bad answer mask")
            output.extend(expected)
            i = end+1
        elif token == ANSWER and ep.task == "selective_copy":
            output.extend(selected)
            i = data.index(END, i+1)+1
        else:
            raise ValueError(f"bad token {token} at {i}")
    return tuple(output)


def dataset(spec, count, split, seed=20261115, map_seed=47):
    if type(count) is not int or count < 1 or split not in ("train", "dev", "test"):
        raise ValueError("positive count and train/dev/test split required")
    table = fixed_table(spec, map_seed)
    rng = random.Random(seed)
    selected, seen = [], set()
    for _ in range(max(10000, count*1000)):
        ep = _candidate(spec, rng, table, len(selected))
        h = hashlib.sha256(spec.task.encode()+ep.split_key).digest()
        bucket = int.from_bytes(h[:8], "big") % 10
        fold = "train" if bucket < 6 else "dev" if bucket < 8 else "test"
        if split != fold or ep.digest in seen:
            continue
        expected = solve(ep, spec, table)
        encoded = tuple(x-VALUE for x, m in zip(ep.data, ep.mask) if m)
        if expected != ep.answer or (spec.task != "compression" and encoded != expected):
            raise AssertionError("generator disagrees with independent interpreter")
        selected.append(ep)
        seen.add(ep.digest)
        if len(selected) == count:
            return selected
    raise ValueError("requested split exceeds task space; increase vocab/length or reduce cases")


def fingerprint(cases):
    return hashlib.sha256("".join(ep.digest for ep in cases).encode()).hexdigest()
