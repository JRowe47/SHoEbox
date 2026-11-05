"""Compact byte-fact episodes with explicit query supervision and no answer feedback."""
from dataclasses import dataclass
import random
import torch

SPLIT_SEEDS = {'train': 1103, 'validation': 4409, 'test': 7919}
VALUES = b'ABCD'

@dataclass(frozen=True)
class ProbeEpisode:
    identity: str
    kind: str
    split: str
    views: dict[str, bytes]
    availability: dict[str, tuple[bool, ...]]
    targets: tuple[int, ...]
    answer: int
    old_answer: int
    query_type: str
    update_span: tuple[int, int]
    last_value_position: int
    final_query: int

@dataclass(frozen=True)
class ProbeBatch:
    inputs: object
    availability: dict[str, torch.Tensor]
    targets: torch.Tensor
    reset: torch.Tensor
    final: torch.Tensor
    answers: torch.Tensor
    old_answers: torch.Tensor
    query_types: tuple[str, ...]
    identities: tuple[str, ...]
    distances: tuple[int, ...]
    retained_targets: torch.Tensor | None = None

    @property
    def length(self):
        return self.targets.shape[1]

    def slice(self, start, stop):
        inputs = {k: v[:, start:stop] for k, v in self.inputs.items()} if isinstance(self.inputs, dict) else self.inputs[:, start:stop]
        return ProbeBatch(inputs, {k: v[:, start:stop] for k, v in self.availability.items()},
                          self.targets[:, start:stop], self.reset[:, start:stop], self.final[:, start:stop],
                          self.answers, self.old_answers, self.query_types, self.identities, self.distances,
                          None if self.retained_targets is None else self.retained_targets[:, start:stop])


def observed_memory_targets(inputs, reset):
    """Loss-only current-value labels from actual writes, including controls.

    Compute before slicing so a later TBPTT segment retains causal eligibility.
    A reset discards the old fact; a withheld write cannot change its label.
    """
    labels = torch.full_like(inputs, -100)
    current = torch.full_like(inputs[:, 0], -100)
    previous = torch.full_like(current, -100)
    for t in range(inputs.shape[1]):
        current = current.masked_fill(reset[:, t], -100)
        previous = previous.masked_fill(reset[:, t], -100)
        value = inputs[:, t]
        written = previous.eq(ord('=')) & value.ge(65) & value.le(68)
        current = torch.where(written, value, current)
        labels[:, t] = current
        previous = value
    return labels


def make_probe(kind='correction', split='train', index=0, seed=0, gap=16, query_every=4):
    """One fact, four values; query answers are never fed back as observations.

    For recall/correction the last observed value is gap+2 positions before
    the final query; shared/private distances also include field delimiters.
    Intermediate query labels train retention within TBPTT segments; only the
    final query is scored. All controls retain identical query positions.
    """
    if kind not in ('recall', 'correction', 'shared_private') or split not in SPLIT_SEEDS:
        raise ValueError('Unknown probe kind or split.')
    if index < 0 or gap < 0 or query_every < 1 or index >= 16**6:
        raise ValueError('Invalid episode index, gap or query interval.')
    rng = random.Random(SPLIT_SEEDS[split] + 104729*index + 10000019*seed)
    identity = {'train':'t','validation':'v','test':'e'}[split] + f'{index:06x}'
    old = rng.choice(VALUES)
    new = rng.choice([x for x in VALUES if x != old]) if kind == 'correction' else old
    noise = lambda: rng.choice(b'fghjkmn:,.')
    if kind != 'shared_private':
        data = bytearray(identity.encode()+b'='+bytes([old])+b'!')
        labels = [-100]*(len(data)-1)+[old]
        update_span = (0, 0)
        if kind == 'correction':
            for i in range(gap//2):
                data.append(ord('!') if (i+1)%query_every == 0 else noise())
                labels.append(old if data[-1] == ord('!') else -100)
            start = len(data)
            update = identity.encode()+b'='+bytes([new])+b'!'
            data.extend(update); labels.extend([-100]*(len(update)-1)+[new])
            update_span = (start, len(data)-1)  # keep the following query itself
        last_value = len(data)-2
        for i in range(gap):
            data.append(ord('!') if (i+1)%query_every == 0 else noise())
            labels.append(new if data[-1] == ord('!') else -100)
        data.append(ord('!')); labels.append(new)
        return ProbeEpisode(identity, kind, split, {'text': bytes(data)}, {}, tuple(labels), new, old,
                            'current', update_span, last_value, len(data)-1)
    shared, left, right = old, rng.choice(VALUES), rng.choice(VALUES)
    query = rng.choice(('shared', 'left', 'right'))
    marker = {'shared':ord('S'), 'left':ord('L'), 'right':ord('R')}[query]
    answer = {'shared':shared, 'left':left, 'right':right}[query]
    prefix = identity.encode()+b':s='+bytes([shared])+b';'
    texts = {'A': bytearray(prefix+b'l='+bytes([left])+b';'),
             'B': bytearray(prefix+b'....'), 'C': bytearray(prefix+b'r='+bytes([right])+b';')}
    # The readout question selects a known fact after all three have arrived.
    last_value = len(texts['A'])-2 if query != 'shared' else len(prefix)-2
    labels = [-100]*len(texts['A'])
    # Put a query in the write segment itself (index 15 with seven-byte IDs),
    # rather than relying on a target just beyond a TBPTT boundary.
    for value in texts.values(): value[-1] = marker
    labels[-1] = answer
    if query == 'shared':
        for value in texts.values(): value[len(prefix)-1] = marker
        labels[len(prefix)-1] = shared
    for value in texts.values(): value.append(marker)
    labels.append(answer)
    for i in range(gap):
        byte = marker if (i+1)%query_every == 0 else noise()
        for value in texts.values(): value.append(byte)
        labels.append(answer if byte == marker else -100)
    for value in texts.values(): value.append(marker)
    labels.append(answer)
    availability = {k: tuple(True for _ in value) for k, value in texts.items()}
    # B has shared information only: the private-field interval is absent.
    mask = list(availability['B']); mask[len(prefix):len(prefix)+4] = [False]*4
    availability['B'] = tuple(mask)
    return ProbeEpisode(identity, kind, split, {k: bytes(v) for k,v in texts.items()}, availability,
                        tuple(labels), answer, answer, query, (0,0), last_value, len(labels)-1)


def make_batch(kind, split, start, batch_size, seed=0, gap=16, query_every=4, control='carried'):
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    if control not in ('carried', 'reset_query', 'withhold_update', 'a_only', 'b_only', 'c_only', 'no_private'):
        raise ValueError('Unknown information control.')
    if control == 'withhold_update' and kind != 'correction':
        raise ValueError('Update withholding requires correction episodes.')
    if control in ('a_only','b_only','c_only','no_private') and kind != 'shared_private':
        raise ValueError('View controls require shared/private episodes.')
    episodes = [make_probe(kind, split, start+i, seed, gap, query_every) for i in range(batch_size)]
    length = len(episodes[0].targets)
    if any(len(e.targets)!=length for e in episodes): raise ValueError('Aligned episode lengths required.')
    views = {k: torch.tensor([list(e.views[k]) for e in episodes], dtype=torch.long) for k in episodes[0].views}
    availability = {k: torch.tensor([e.availability[k] for e in episodes], dtype=torch.bool) for k in episodes[0].availability}
    reset = torch.zeros(batch_size,length,dtype=torch.bool); reset[:,0] = True
    final = torch.zeros_like(reset);final[:,-1] = True
    targets = torch.tensor([e.targets for e in episodes],dtype=torch.long)
    if control == 'reset_query': reset[:,-1] = True
    if control == 'withhold_update':
        for row,e in enumerate(episodes):
            a,b=e.update_span
            views['text'][row,a:b] = ord('.')
        # Control labels always describe the observed stream; final scoring also
        # records accuracy against the intended corrected answer separately.
        for row,e in enumerate(episodes):
            targets[row, e.update_span[0]:] = torch.where(targets[row,e.update_span[0]:]!=-100,
                                                        e.old_answer, -100)
    if control in ('a_only','b_only','c_only'):
        keep = control[0].upper()
        availability = {k: mask if k == keep else torch.zeros_like(mask) for k,mask in availability.items()}
    if control == 'no_private':
        # Exact private-value observations only; positions and all execution stay.
        for row,e in enumerate(episodes):
            private_position = len(e.identity)+1+4+2
            for view in ('A','C'):availability[view][row,private_position] = False
    return ProbeBatch(views if kind=='shared_private' else views['text'], availability, targets, reset, final,
                      torch.tensor([e.answer for e in episodes]), torch.tensor([e.old_answer for e in episodes]),
                      tuple(e.query_type for e in episodes), tuple(e.identity for e in episodes),
                      tuple(e.final_query-e.last_value_position for e in episodes),
                      None if kind == 'shared_private' else observed_memory_targets(views['text'], reset))


class EpisodeStream:
    """Ordered fixed-length episodes; cursor fully reconstructs the active batch."""
    def __init__(self, kind='correction', batch_size=4, gap=16, query_every=4, seed=0):
        self.spec = dict(kind=kind, batch_size=batch_size, gap=gap, query_every=query_every, seed=seed)
        self.batch_start = 0
        self.position = 0
        self.batch = self._make()

    def _make(self):
        return make_batch(split='train', start=self.batch_start, **self.spec)

    def next_chunk(self, length):
        if length<1: raise ValueError('Positive segment length required.')
        if self.position == self.batch.length:
            self.batch_start += self.spec['batch_size']; self.position=0; self.batch=self._make()
        end=min(self.position+length,self.batch.length)
        result=self.batch.slice(self.position,end);self.position=end
        return result

    def state_dict(self):
        return {'schema_version':1,'spec':dict(self.spec),'batch_start':self.batch_start,'position':self.position}

    def load_state_dict(self,data):
        if data['schema_version']!=1 or data['spec']!=self.spec:raise ValueError('Episode cursor/protocol mismatch.')
        if data['batch_start']<0 or data['batch_start']%self.spec['batch_size']:raise ValueError('Invalid episode cursor.')
        self.batch_start=data['batch_start'];self.batch=self._make()
        if not 0<=data['position']<=self.batch.length:raise ValueError('Invalid segment cursor.')
        self.position=data['position']
