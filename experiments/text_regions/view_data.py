"""Factorized source views with held-out tuples and a full-factorial control cohort.

Every episode has a 16-event evidence/write segment and ``gap+1`` later events.
Source-specific padding suppresses ingress only; graph nodes keep their clocks
and expert budget. Answers label query outputs and are never fed back.
"""
from dataclasses import dataclass
import hashlib
from itertools import product
import math
import random
import torch

VALUES = b'ABCD'
VIEWS = ('A', 'B', 'C')
QUERY_TYPES = ('shared', 'left', 'right')
SPLITS = ('train', 'validation', 'test', 'independent')
DESCRIPTION_END = 14
EVIDENCE_END = 16
IGNORE = -100
FACTOR_SOURCES = {'shared': VIEWS, 'left': ('A',), 'right': ('C',)}
PRIVATE_FACTORS = {'A': 'left', 'C': 'right'}
MARKERS = {'shared': ord('S'), 'left': ord('L'), 'right': ord('R')}
CONTROLS = ('all', 'a_only', 'b_only', 'c_only', 'no_private', 'reset_query')
SOURCE_SCHEDULES = ('dropout', 'balanced')


def _rng(seed, *parts):
    key = repr((seed, *parts)).encode()
    return random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), 'big'))


def split_combinations(split):
    """Byte-valued (shared,left,right) tuples; 32/16/16/64 combinations.

    The three restricted cohorts test factor recombination. Use ``independent``
    for information-withholding controls: its missing factors remain uniform
    conditional on all visible factors. A multiple of 64 episodes per query
    exhausts complete cycles in this cohort.
    """
    if split not in SPLITS: raise ValueError('Unknown factor split.')
    residues = {'train': (0, 1), 'validation': (2,), 'test': (3,), 'independent': (0, 1, 2, 3)}[split]
    return tuple(tuple(VALUES[v] for v in values) for values in product(range(4), repeat=3)
                 if sum(values) % 4 in residues)


def _coordinates(index, query_type):
    if not isinstance(index, int) or index < 0: raise ValueError('Episode index must be nonnegative.')
    if query_type is None:
        query_type, ordinal, unique = QUERY_TYPES[index % 3], index // 3, index
    else:
        if query_type not in QUERY_TYPES: raise ValueError('Unknown query type.')
        ordinal, unique = index, 3*index+QUERY_TYPES.index(query_type)
    if unique >= 2**22: raise ValueError('Episode identifier space exhausted.')
    return query_type, ordinal, unique


def _opaque_identifier(split, unique, seed):
    # A keyed permutation preserves uniqueness across all four cohorts while
    # obscuring allocation order. No visible split prefix or answer-index digit.
    value = (SPLITS.index(split) << 22) | unique
    left, right = value >> 12, value & 4095
    for round_index in range(6):
        raw = repr((seed, 'identifier', round_index, right)).encode()
        mixed = int.from_bytes(hashlib.blake2s(raw, digest_size=4).digest(), 'big') & 4095
        left, right = right, left ^ mixed
    return f'{(left << 12) | right:06x}'


def _factors(split, query_type, ordinal, seed):
    axis = QUERY_TYPES.index(query_type)
    # Four consecutive episodes of one query have exactly one answer per byte.
    # Independently permuting each stratum also exhausts every allowed tuple.
    answer = VALUES[ordinal % 4]
    candidates = [values for values in split_combinations(split) if values[axis] == answer]
    _rng(seed, split, query_type, answer, 'factor-order').shuffle(candidates)
    return dict(zip(QUERY_TYPES, candidates[(ordinal // 4) % len(candidates)]))


def _balanced_source_bits(query_type, ordinal, seed):
    """Rotate supported whole-source subsets independently of answer values.

    The four answers within an ordinal group share one subset. There are eight
    train tuples per answer; rotating again every 32 episodes prevents a tuple
    from being tied to one subset. In 1280 episodes per query, each shared tuple
    visits every supported subset 5–6 times; each private tuple does so 10 times.
    """
    choices = [bits for bits in range(1, 8) if any(bits & (1 << VIEWS.index(view))
               for view in FACTOR_SOURCES[query_type])]
    _rng(seed, 'balanced-source-order', query_type).shuffle(choices)
    return choices[((ordinal // 4) % 8 + ordinal // 32) % len(choices)]


_TEMPLATES = {
    'A': ((b's=', 'shared', b'l=', 'left'),
          (b'l:', 'left', b'/s:', 'shared'),
          (b'(', 'shared', b'){', 'left', b'}')),
    'B': ((b's=', 'shared'), (b'all:', 'shared'), (b'(', 'shared', b')')),
    'C': ((b'r=', 'right', b';s=', 'shared'),
          (b'[', 'shared', b']/', 'right'),
          (b's:', 'shared', b'r:', 'right')),
}


@dataclass(frozen=True)
class ViewEpisode:
    identity: str
    split: str
    query_type: str
    factors: dict[str, int]
    views: dict[str, bytes]
    availability: dict[str, tuple[bool, ...]]
    fact_positions: dict[str, dict[str, int]]
    targets: tuple[int, ...]
    private_targets: dict[str, tuple[int, ...]]
    shared_targets: dict[str, tuple[int, ...]]
    query_positions: tuple[int, ...]
    template_ids: dict[str, int]
    description_end: int = DESCRIPTION_END

    @property
    def answer(self): return self.factors[self.query_type]

    @property
    def final_query(self): return len(self.targets)-1

    @property
    def private_masks(self):
        return {view: tuple(value != IGNORE for value in targets) for view, targets in self.private_targets.items()}

    @property
    def observation_positions(self): return self.fact_positions


def make_view_episode(split='train', index=0, seed=0, gap=8, query_every=4, query_type=None):
    if split not in SPLITS: raise ValueError('Unknown factor split.')
    if not isinstance(gap, int) or gap < 0 or not isinstance(query_every, int) or query_every < 1:
        raise ValueError('Gap must be nonnegative and query interval positive.')
    query_type, ordinal, unique = _coordinates(index, query_type)
    factors = _factors(split, query_type, ordinal, seed)
    identity = _opaque_identifier(split, unique, seed)
    identifier = identity.encode()
    headers = {'A': identifier+b':', 'B': b'#'+identifier, 'C': identifier[::-1]+b'/'}
    marker = MARKERS[query_type]
    views, masks, positions, templates = {}, {}, {}, {}
    for view in VIEWS:
        rng = _rng(seed, split, unique, view, 'render')
        template_id = rng.randrange(len(_TEMPLATES[view]))
        data = bytearray(headers[view]); positions[view] = {factor: -1 for factor in QUERY_TYPES}
        for part in _TEMPLATES[view][template_id]:
            if isinstance(part, str):
                positions[view][part] = len(data); data.append(factors[part])
            else: data.extend(part)
        if len(data) > DESCRIPTION_END: raise AssertionError('Evidence template exceeds its segment.')
        available = [True]*len(data)+[False]*(DESCRIPTION_END-len(data))
        data.extend(bytes(DESCRIPTION_END-len(data)))
        data.extend((marker, marker)); available.extend((True, True))
        for offset in range(gap):
            data.append(marker if (offset+1) % query_every == 0 else rng.choice(b'fghjkmnp:,.'))
            available.append(True)
        data.append(marker); available.append(True)
        views[view], masks[view], templates[view] = bytes(data), tuple(available), template_id
    query_positions = (14, 15, *(16+i for i in range(gap) if (i+1) % query_every == 0), 16+gap)
    targets = tuple(factors[query_type] if t in query_positions else IGNORE for t in range(17+gap))
    private_targets = {view: tuple(factors[factor] if t >= positions[view][factor] and masks[view][t] else IGNORE
                                   for t in range(17+gap)) for view, factor in PRIVATE_FACTORS.items()}
    shared_targets = {view: tuple(factors['shared'] if t >= positions[view]['shared'] and masks[view][t] else IGNORE
                                  for t in range(17+gap)) for view in VIEWS}
    return ViewEpisode(identity, split, query_type, factors, views, masks, positions, targets,
                       private_targets, shared_targets, query_positions, templates)


@dataclass(frozen=True)
class ViewBatch:
    inputs: dict[str, torch.Tensor]
    availability: dict[str, torch.Tensor]
    targets: torch.Tensor
    reset: torch.Tensor
    final: torch.Tensor
    answers: torch.Tensor
    factors: dict[str, torch.Tensor]
    query_types: tuple[str, ...]
    identities: tuple[str, ...]
    fact_positions: dict[str, dict[str, torch.Tensor]]
    observation_positions: dict[str, dict[str, torch.Tensor]]
    private_targets: dict[str, torch.Tensor]
    private_masks: dict[str, torch.Tensor]
    shared_targets: dict[str, torch.Tensor]
    answer_available: torch.Tensor
    template_ids: dict[str, torch.Tensor]
    retained_views: torch.Tensor
    split: str
    control: str
    description_end: int = DESCRIPTION_END
    time_offset: int = 0
    last_reset_positions: torch.Tensor | None = None

    @property
    def length(self): return self.targets.shape[1]

    @property
    def distances(self):
        result = []
        final = self.time_offset+self.length-1
        for row, factor in enumerate(self.query_types):
            observations = [int(self.observation_positions[view][factor][row]) for view in FACTOR_SOURCES[factor]]
            found = [position for position in observations if 0 <= position <= final]
            result.append(final-max(found) if found and bool(self.answer_available[row, -1]) else -1)
        return tuple(result)

    def pooled_private_targets(self):
        """Target at each source's last available occurrence in this chunk."""
        targets = {}
        for view in PRIVATE_FACTORS:
            available = self.availability[view]
            positions = torch.arange(self.length, device=available.device)[None]
            indices = torch.where(available, positions, -1).max(1).values
            values = self.private_targets[view].gather(1, indices.clamp_min(0)[:, None]).squeeze(1)
            targets[view] = values.masked_fill(indices < 0, IGNORE)
        return targets

    def baseline_observed_mask(self, view):
        """Whether this available source has observed the requested factor.

        Causal target-presence metadata retains knowledge across slices and
        clears it at resets. Equal-valued unrelated factors do not establish
        knowledge. This loss mask never becomes a model input.
        """
        if view not in VIEWS: raise ValueError('Unknown source view.')
        available = self.availability[view]
        shared_rows = torch.tensor([query == 'shared' for query in self.query_types],
                                   dtype=torch.bool, device=available.device)[:, None]
        observed = shared_rows & self.shared_targets[view].ne(IGNORE)
        if view in PRIVATE_FACTORS:
            private_rows = torch.tensor([query == PRIVATE_FACTORS[view] for query in self.query_types],
                                        dtype=torch.bool, device=available.device)[:, None]
            observed |= private_rows & self.private_targets[view].ne(IGNORE)
        return observed & available

    def slice(self, start, stop):
        if not 0 <= start < stop <= self.length: raise ValueError('Invalid nonempty time slice.')
        temporal = lambda values: {key: value[:, start:stop] for key, value in values.items()}
        return ViewBatch(temporal(self.inputs), temporal(self.availability), self.targets[:, start:stop],
                         self.reset[:, start:stop], self.final[:, start:stop], self.answers, self.factors,
                         self.query_types, self.identities, self.fact_positions, self.observation_positions,
                         temporal(self.private_targets), temporal(self.private_masks), temporal(self.shared_targets),
                         self.answer_available[:, start:stop], self.template_ids, self.retained_views,
                         self.split, self.control, self.description_end, self.time_offset+start,
                         None if self.last_reset_positions is None else self.last_reset_positions[:, start:stop])


def _causal_targets(episodes, availability, reset):
    """Information bookkeeping is metadata; it never enters model inputs/state."""
    batch, length = reset.shape
    private = {view: torch.full((batch, length), IGNORE, dtype=torch.long) for view in PRIVATE_FACTORS}
    shared = {view: torch.full((batch, length), IGNORE, dtype=torch.long) for view in VIEWS}
    answer_available = torch.zeros_like(reset)
    for row, episode in enumerate(episodes):
        known = {view: {factor: False for factor in QUERY_TYPES} for view in VIEWS}
        for t in range(length):
            if bool(reset[row, t]):
                known = {view: {factor: False for factor in QUERY_TYPES} for view in VIEWS}
            for view in VIEWS:
                if not bool(availability[view][row, t]): continue
                for factor in QUERY_TYPES:
                    if episode.fact_positions[view][factor] == t: known[view][factor] = True
                if known[view]['shared']: shared[view][row, t] = episode.factors['shared']
                if view in PRIVATE_FACTORS and known[view][PRIVATE_FACTORS[view]]:
                    private[view][row, t] = episode.factors[PRIVATE_FACTORS[view]]
            answer_available[row, t] = any(known[view][episode.query_type] for view in VIEWS)
    return private, shared, answer_available


def make_view_batch(split, start, batch_size, seed=0, gap=8, query_every=4,
                    control='all', query_type=None, dropout=0.0, source_schedule='dropout'):
    if not isinstance(batch_size, int) or batch_size < 1: raise ValueError('Batch size must be positive.')
    if control not in CONTROLS: raise ValueError('Unknown view control.')
    if not math.isfinite(dropout) or not 0 <= dropout <= 1: raise ValueError('Dropout must be in [0,1].')
    if source_schedule not in SOURCE_SCHEDULES: raise ValueError('Unknown source schedule.')
    if source_schedule == 'balanced' and (split != 'train' or control != 'all' or dropout != 0):
        raise ValueError('Balanced source scheduling requires the train split, all control, and dropout=0.')
    if dropout and (split != 'train' or control != 'all'):
        raise ValueError('Random source dropout is training-only and cannot be combined with an evaluation control.')
    episodes = [make_view_episode(split, start+i, seed, gap, query_every, query_type) for i in range(batch_size)]
    length = len(episodes[0].targets)
    inputs = {view: torch.tensor([list(episode.views[view]) for episode in episodes], dtype=torch.long) for view in VIEWS}
    availability = {view: torch.tensor([episode.availability[view] for episode in episodes], dtype=torch.bool) for view in VIEWS}
    reset = torch.zeros(batch_size, length, dtype=torch.bool); reset[:, 0] = True
    final = torch.zeros_like(reset); final[:, -1] = True
    retained = torch.ones(batch_size, 3, dtype=torch.bool)
    if control in ('a_only', 'b_only', 'c_only'):
        retained[:] = False; retained[:, VIEWS.index(control[0].upper())] = True
    if source_schedule == 'balanced':
        for row, episode in enumerate(episodes):
            _, ordinal, _ = _coordinates(start+row, query_type)
            bits = _balanced_source_bits(episode.query_type, ordinal, seed)
            retained[row] = torch.tensor([bool(bits & (1 << index)) for index in range(3)])
    elif dropout:
        for row, episode in enumerate(episodes):
            rng = _rng(seed, split, episode.identity, 'dropout')
            if rng.random() < dropout:
                choices = [bits for bits in range(1, 7) if any(bits & (1 << VIEWS.index(view))
                           for view in FACTOR_SOURCES[episode.query_type])]
                bits = rng.choice(choices)
                retained[row] = torch.tensor([bool(bits & (1 << index)) for index in range(3)])
    for view_index, view in enumerate(VIEWS): availability[view] &= retained[:, view_index, None]
    if control == 'no_private':
        for row, episode in enumerate(episodes):
            for view, factor in PRIVATE_FACTORS.items(): availability[view][row, episode.fact_positions[view][factor]] = False
    if control == 'reset_query': reset[:, -1] = True
    positions = {view: {factor: torch.tensor([episode.fact_positions[view][factor] for episode in episodes], dtype=torch.long)
                        for factor in QUERY_TYPES} for view in VIEWS}
    observed = {view: {factor: value.clone() for factor, value in fields.items()} for view, fields in positions.items()}
    for view in VIEWS:
        for factor in QUERY_TYPES:
            for row in range(batch_size):
                position = int(observed[view][factor][row])
                if position >= 0 and not bool(availability[view][row, position]): observed[view][factor][row] = -1
    private, shared, answer_available = _causal_targets(episodes, availability, reset)
    return ViewBatch(inputs, availability, torch.tensor([episode.targets for episode in episodes]), reset, final,
                     torch.tensor([episode.answer for episode in episodes]),
                     {factor: torch.tensor([episode.factors[factor] for episode in episodes]) for factor in QUERY_TYPES},
                     tuple(episode.query_type for episode in episodes), tuple(episode.identity for episode in episodes),
                     positions, observed, private, {view: values != IGNORE for view, values in private.items()}, shared,
                     answer_available, {view: torch.tensor([episode.template_ids[view] for episode in episodes]) for view in VIEWS},
                     retained, split, control, last_reset_positions=torch.where(
                         reset, torch.arange(length)[None], -1).cummax(1).values)


class ViewStream:
    """Ordered episode batches with a reconstructible deterministic cursor."""
    def __init__(self, batch_size=32, gap=8, query_every=4, seed=0, dropout=0.0, query_type=None,
                 source_schedule='dropout'):
        self.spec = dict(batch_size=batch_size, gap=gap, query_every=query_every, seed=seed,
                         dropout=dropout, query_type=query_type, source_schedule=source_schedule)
        self.batch_start = self.position = 0
        self.batch = self._make()

    def _make(self): return make_view_batch(split='train', start=self.batch_start, **self.spec)

    def next_chunk(self, length):
        if not isinstance(length, int) or length < 1: raise ValueError('Positive segment length required.')
        if self.position == self.batch.length:
            self.batch_start += self.spec['batch_size']; self.position = 0; self.batch = self._make()
        stop = min(self.position+length, self.batch.length)
        result = self.batch.slice(self.position, stop); self.position = stop
        return result

    def state_dict(self):
        return {'schema_version': 1, 'spec': dict(self.spec), 'batch_start': self.batch_start, 'position': self.position}

    def load_state_dict(self, state):
        saved_spec = {'source_schedule': 'dropout', **state['spec']}
        if state['schema_version'] != 1 or saved_spec != self.spec: raise ValueError('View stream protocol mismatch.')
        start, position = state['batch_start'], state['position']
        if not isinstance(start, int) or start < 0 or start % self.spec['batch_size']: raise ValueError('Invalid batch cursor.')
        if not isinstance(position, int) or not 0 <= position <= 17+self.spec['gap']: raise ValueError('Invalid time cursor.')
        batch = make_view_batch(split='train', start=start, **self.spec)
        self.batch_start, self.position, self.batch = start, position, batch
