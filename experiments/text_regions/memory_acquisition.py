"""Fresh, balanced source-value acquisition prefixes from the ordinary grammar.

All four versions of a case differ only at the observed value byte. Targets and
source positions stay outside the model. Prefixes retain all earlier observations
and questions; answers have never been inserted into that byte stream. Training
uses a new document index for every base case and a full connected prefix.
"""
from copy import deepcopy
import torch
from .task_stream import make_document
from .value_recall import value_cases


def acquisition_cases(config, *, split, seed, start, count, distractor_bytes=None):
    cases = []
    for index in range(start, start + count):
        document = make_document(config, split=split, seed=seed, index=index,
                                 distractor_bytes=distractor_bytes)
        candidates = value_cases([document], config.values, acquisition=True)
        case = candidates[index % len(candidates)]
        case['query_positions'] = [q.position for q in document.queries
                                   if q.position <= case['query']['position']]
        cases.append(case)
    return cases


def pack_cases(cases):
    sequences = [seq for case in cases for seq in case['variants']]
    length = max(map(len, sequences))
    x = torch.zeros(len(sequences), length, dtype=torch.long)
    ordinary = torch.full_like(x, -100)
    valid = torch.zeros_like(x, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    answers = []
    row = 0
    for case in cases:
        for sequence, answer in zip(case['variants'], case['answers']):
            offset = length-len(sequence)
            values = torch.tensor(list(sequence))
            x[row, offset:] = values
            ordinary[row, offset:-1] = values[1:]
            for position in case['query_positions']:
                ordinary[row, offset+position] = -100
            valid[row, offset:] = True
            reset[row, offset] = True
            answers.append(answer)
            row += 1
    return x, ordinary, torch.tensor(answers), valid, reset


def prefix_forward(model, packed):
    x, ordinary, answers, valid, reset = packed
    result = model(x, reset=reset, availability={v: valid for v in model.cfg.views},
                   node_masks={n: valid for n in model.nodes})
    return result


def delayed_cases(cases, gap, distractor_bytes):
    """Insert held-out natural text just before the final native question."""
    if gap < 0 or not distractor_bytes:
        raise ValueError('Nonnegative gap and nonempty natural text required.')
    result = deepcopy(cases)
    for case in result:
        query = case['query']
        query_start = query['position'] - len(f"{query['entity']} {query['attribute']}?") + 1
        offset = (case['document']*137) % len(distractor_bytes)
        repeated = distractor_bytes*((offset+gap)//len(distractor_bytes)+1)
        middle = repeated[offset:offset+gap]
        case['variants'] = [seq[:query_start]+middle+seq[query_start:] for seq in case['variants']]
        query['position'] += gap; query['delay'] += gap
        case['query_positions'][-1] += gap
    return result
