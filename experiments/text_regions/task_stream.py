"""Causal byte streams for learning when retained experience is useful.

Only ``TaskChunk.values`` is a model input. Answer masks and provenance belong
solely to the loss/evaluator: there are no producer, memory-slot or route IDs.
Query answers are never fed back as observations. Related events share one
stream; reset marks only the beginning of a new independent document.
"""
from dataclasses import asdict, dataclass
import hashlib
import random
import torch


_SPLITS = {'train': 104729, 'validation': 130363, 'val': 130363,
           'test': 155921, 'combinations': 196613, 'long_delay': 262147}


@dataclass(frozen=True)
class TaskStreamConfig:
    entities: str = 'AB'
    attributes: tuple[str, ...] = ('x',)
    values: str = '0123'
    cycles: int = 2
    queries_per_cycle: int = 3
    delays: tuple[int, ...] = (2, 24)
    long_delay_multiplier: int = 3
    semantic_fraction: float = .2
    contextual_questions: bool = True
    relations: bool = False
    holdout_combinations: bool = False

    def validate(self):
        if (len(self.entities) < 2 or len(set(self.entities)) != len(self.entities)
                or not self.entities.isalpha() or not self.entities.isascii()):
            raise ValueError('At least two distinct ASCII entity letters are required.')
        if (not self.attributes or len(set(self.attributes)) != len(self.attributes)
                or any(not a.isalpha() or not a.isascii() for a in self.attributes)):
            raise ValueError('Attributes must be distinct ASCII words.')
        if (len(self.values) < 2 or len(set(self.values)) != len(self.values)
                or any(c not in '0123456789' for c in self.values)):
            raise ValueError('Values must be distinct single decimal digits.')
        if self.cycles < 1 or self.queries_per_cycle < 2 or not self.delays or min(self.delays) < 0:
            raise ValueError('Positive cycles, at least two queries and nonnegative delays are required.')
        if self.long_delay_multiplier < 1 or not 0 <= self.semantic_fraction <= 1:
            raise ValueError('Invalid delay multiplier or semantic fraction.')
        if self.holdout_combinations and len(self.attributes) < 2:
            raise ValueError('Combination holdout requires at least two attributes.')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, 'attributes': tuple(data['attributes']), 'delays': tuple(data['delays'])})


@dataclass(frozen=True)
class TaskQuery:
    position: int
    answer: int
    kind: str
    entity: str
    attribute: str
    source_position: int
    source_version: int
    delay: int
    correction: bool
    previous_answer: int | None
    intervening_writes: int
    cycle: int


@dataclass(frozen=True)
class TaskEvent:
    start: int
    stop: int
    kind: str
    text: bytes
    # Evaluation-only link. This is never an input to the model.
    query_index: int | None = None


@dataclass(frozen=True)
class TaskDocument:
    data: bytes
    answer_targets: tuple[int, ...]
    queries: tuple[TaskQuery, ...]
    events: tuple[TaskEvent, ...]
    split: str
    index: int

    def __post_init__(self):
        if len(self.data) != len(self.answer_targets) or len(self.data) < 2:
            raise ValueError('Targets must align with input byte positions.')


@dataclass(frozen=True)
class TaskChunk:
    values: torch.Tensor
    ordinary_targets: torch.Tensor
    answer_targets: torch.Tensor
    answer_mask: torch.Tensor
    reset: torch.Tensor
    document_ids: torch.Tensor
    positions: torch.Tensor

    @property
    def targets(self):
        """Answer-only targets; use ordinary_targets for optional byte loss."""
        return self.answer_targets


def _heldout(entity_index, attribute_index):
    return (entity_index + attribute_index) % 2 == 1


def make_document(config=None, *, split='train', seed=0, index=0):
    """Render one independent stream, with corrections inside its carried state.

    ``combinations`` tests entity/attribute query pairs withheld from training
    when holdout_combinations is enabled. All facts are still observed, so this
    measures query composition rather than access to unobserved facts.
    ``long_delay`` uses the same grammar with multiplied acquisition/use gaps.
    """
    cfg = config or TaskStreamConfig()
    cfg.validate()
    if split not in _SPLITS or index < 0:
        raise ValueError('Unknown split or negative stream index.')
    rng = random.Random(f'{_SPLITS[split]}:{seed}:{index}')
    data, targets, queries, events = bytearray(), [], [], []
    facts, prior, versions, source, write_serial = {}, {}, {}, {}, {}
    writes = 0
    pairs = [(e, a) for e in cfg.entities for a in cfg.attributes]
    query_pairs = [(e, a) for ei, e in enumerate(cfg.entities) for ai, a in enumerate(cfg.attributes)
                   if not cfg.holdout_combinations or _heldout(ei, ai) == (split == 'combinations')]
    relation = {e: cfg.entities[(i + 1) % len(cfg.entities)] for i, e in enumerate(cfg.entities)}

    def append(text, kind, query_index=None):
        start = len(data)
        data.extend(text)
        targets.extend([-100] * len(text))
        events.append(TaskEvent(start, len(data), kind, bytes(text), query_index))

    def write(pair, value, correction=False):
        nonlocal writes
        e, a = pair
        previous = facts.get(pair)
        text = f'{e} {a}={value};'.encode('ascii')
        append(text, 'correction' if correction else 'observation')
        prior[pair], facts[pair] = previous, value
        versions[pair] = versions.get(pair, 0) + 1
        source[pair] = len(data) - 2
        writes += 1
        write_serial[pair] = writes

    def wait_for(pair, delay):
        # Query tokens provide additional delay; ensure the declared minimum
        # using already observed bytes only. Filler has no value digits.
        missing = max(0, delay - (len(data) - 1 - source[pair]))
        if missing:
            append(bytes(rng.choice(b' .,:~') for _ in range(missing)), 'distractor')

    def ask(pair, cycle, requested_delay, semantic=False, relational=False):
        e, a = pair
        owner = (relation[e], a) if relational else pair
        wait_for(owner, requested_delay)
        if semantic:
            text = f'{e} {a} high?'.encode('ascii')
            threshold = len(cfg.values) // 2
            answer = ord('H' if cfg.values.index(facts[owner]) >= threshold else 'L')
            old = prior[owner]
            old_answer = None if old is None else ord('H' if cfg.values.index(old) >= threshold else 'L')
            kind = 'semantic'
        else:
            text = f'{e} follows {a}?'.encode('ascii') if relational else f'{e} {a}?'.encode('ascii')
            answer = ord(facts[owner])
            old_answer = None if prior[owner] is None else ord(prior[owner])
            kind = 'relation' if relational else 'exact'
        append(text, 'query', len(queries))
        position = len(data) - 1
        targets[position] = answer
        queries.append(TaskQuery(position, answer, kind, e, a, source[owner], versions[owner],
                                 position - source[owner], versions[owner] > 1,
                                 old_answer, writes - write_serial[owner], cycle))
        append(b';', 'separator')

    # Different entities have conflicting values whenever the alphabet permits.
    # Random rotation avoids a fixed entity -> answer shortcut across documents.
    for ai, a in enumerate(cfg.attributes):
        shuffled = list(cfg.values)
        rng.shuffle(shuffled)
        for ei, e in enumerate(cfg.entities):
            write((e, a), shuffled[(ei + ai) % len(shuffled)])
    if cfg.relations:
        for e, leader in relation.items():
            append(f'{e} follows {leader};'.encode('ascii'), 'relation_observation')

    for cycle in range(cfg.cycles):
        correction_pair = rng.choice(query_pairs)
        if cycle:
            write(correction_pair, rng.choice([v for v in cfg.values if v != facts[correction_pair]]), True)
        order = list(query_pairs)
        rng.shuffle(order)
        # A correction is immediately followed by a query of the affected fact,
        # with a configurable retention interval between them.
        if cycle:
            order = [correction_pair] + [p for p in order if p != correction_pair]
        for q in range(cfg.queries_per_cycle):
            pair = order[q % len(order)]
            delay = max(cfg.delays) if q == cfg.queries_per_cycle - 1 else cfg.delays[q % len(cfg.delays)]
            if split == 'long_delay':
                delay *= cfg.long_delay_multiplier
            # First query after correction remains exact to measure adaptation.
            semantic = q > 0 and rng.random() < cfg.semantic_fraction
            relational = cfg.relations and q > 0 and not semantic and rng.random() < .25
            ask(pair, cycle, delay, semantic, relational)
        if cfg.contextual_questions:
            answer = rng.choice('KL')
            append(f'echo {answer}?'.encode('ascii'), 'query', len(queries))
            pos = len(data) - 1
            targets[pos] = ord(answer)
            queries.append(TaskQuery(pos, ord(answer), 'context', '', '', pos - 1, 0, 1,
                                     False, None, 0, cycle))
            append(b';', 'separator')
    return TaskDocument(bytes(data), tuple(targets), tuple(queries), tuple(events), split, index)


def make_documents(config=None, *, split='train', seed=0, count=32, start=0):
    if count < 1 or start < 0:
        raise ValueError('Positive count and nonnegative start required.')
    return tuple(make_document(config, split=split, seed=seed, index=i) for i in range(start, start + count))


class TaskByteStreams:
    """Resumable ordered independent rows; no reset at event/cycle/chunk edges."""
    def __init__(self, documents, batch_size=4):
        self.documents = tuple(documents)
        if batch_size < 1 or len(self.documents) < batch_size:
            raise ValueError('Supply at least one independent document per row.')
        self.batch_size = batch_size
        digest = hashlib.sha256()
        for document in self.documents:
            digest.update(len(document.data).to_bytes(8, 'big'))
            digest.update(document.data)
            digest.update(repr(document.answer_targets).encode('ascii'))
        self.fingerprint = digest.hexdigest()
        self.document_ids = list(range(batch_size))
        self.positions = [0] * batch_size
        self.next_document = batch_size

    def next_chunk(self, length):
        if length < 1:
            raise ValueError('Chunk length must be positive.')
        shape = (self.batch_size, length)
        x = torch.empty(shape, dtype=torch.long)
        y = torch.full(shape, -100, dtype=torch.long)
        answer = torch.full_like(y, -100)
        reset = torch.zeros(shape, dtype=torch.bool)
        ids, positions = torch.empty_like(y), torch.empty_like(y)
        for t in range(length):
            for row in range(self.batch_size):
                doc_id, pos = self.document_ids[row], self.positions[row]
                document = self.documents[doc_id]
                if pos >= len(document.data):
                    doc_id, pos = self.next_document % len(self.documents), 0
                    self.next_document += 1
                    self.document_ids[row] = doc_id
                    document = self.documents[doc_id]
                x[row, t] = document.data[pos]
                if pos + 1 < len(document.data):
                    y[row, t] = document.data[pos + 1]
                answer[row, t] = document.answer_targets[pos]
                reset[row, t] = pos == 0
                ids[row, t], positions[row, t] = doc_id, pos
                self.positions[row] = pos + 1
        return TaskChunk(x, y, answer, answer.ne(-100), reset, ids, positions)

    def state_dict(self):
        return {'schema_version': 1, 'fingerprint': self.fingerprint, 'batch_size': self.batch_size,
                'document_ids': list(self.document_ids), 'positions': list(self.positions),
                'next_document': self.next_document}

    def load_state_dict(self, state):
        if (state['schema_version'] != 1 or state['fingerprint'] != self.fingerprint
                or state['batch_size'] != self.batch_size):
            raise ValueError('Task stream checkpoint dataset/schema/batch mismatch.')
        ids, positions = state['document_ids'], state['positions']
        if len(ids) != self.batch_size or len(positions) != self.batch_size:
            raise ValueError('Invalid cursor length.')
        if any(not 0 <= i < len(self.documents) for i in ids):
            raise ValueError('Invalid document ID.')
        if any(not 0 <= p <= len(self.documents[i].data) for i, p in zip(ids, positions)):
            raise ValueError('Invalid document position.')
        if state['next_document'] < self.batch_size:
            raise ValueError('Invalid next document cursor.')
        self.document_ids, self.positions = list(ids), list(positions)
        self.next_document = int(state['next_document'])
