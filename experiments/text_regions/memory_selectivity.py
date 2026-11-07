"""Balanced relevance and revision tasks in the existing observation grammar.

Two ordinary writes precede a question. Each base template crosses every
queried value with every nuisance value, including equal assignments. The
same template can query its older fact, its newer fact, or a corrected fact.
Only bytes enter the model; all provenance below is evaluator metadata.
"""
from dataclasses import asdict
import random

from .task_stream import TaskQuery
from .memory_acquisition import acquisition_cases, delayed_cases


MODES = ('single', 'older', 'newer', 'correction')
SPLITS = ('train', 'validation', 'test')


def selective_cases(config, *, split, seed, start, count, mode, gap,
                    distractor_bytes):
    config.validate()
    if (split not in SPLITS or mode not in MODES or start < 0 or count < 1
            or gap < 0 or not isinstance(distractor_bytes, bytes) or not distractor_bytes):
        raise ValueError('Valid split/mode, nonnegative start/gap, positive count and text required.')
    if mode == 'single':
        cases = acquisition_cases(config, split=split, seed=seed, start=start,
                                  count=count, distractor_bytes=distractor_bytes)
        result = delayed_cases(cases, gap, distractor_bytes)
        for case in result:
            case['mode'] = mode
        return result
    cases = []
    for index in range(start, start+count):
        # Do not include mode: older/newer query interventions share exactly
        # the same event templates and natural-text positions.
        rng = random.Random(f'selective:{split}:{seed}:{index}')
        first, second = rng.sample(list(config.entities), 2)
        attribute = rng.choice(config.attributes)
        if mode == 'correction':
            second = first
        offset = rng.randrange(len(distractor_bytes))
        repeated = distractor_bytes*((offset+gap)//len(distractor_bytes)+1)
        middle = repeated[offset:offset+gap]
        owner = first if mode == 'older' else second
        question = f'{owner} {attribute}?'.encode('ascii')
        for nuisance in config.values:
            variants = []
            for target in config.values:
                old, new = (target, nuisance) if mode == 'older' else (nuisance, target)
                write1 = f'{first} {attribute}={old};'.encode('ascii')
                write2 = f'{second} {attribute}={new};'.encode('ascii')
                variants.append(write1+write2+middle+question)
            source = len(write1)-2 if mode == 'older' else len(write1)+len(write2)-2
            position = len(variants[0])-1
            query = TaskQuery(position=position, answer=ord(config.values[0]), kind='exact',
                entity=owner, attribute=attribute, source_position=source,
                source_version=2 if mode == 'correction' else 1,
                delay=position-source, correction=mode == 'correction',
                previous_answer=ord(nuisance) if mode == 'correction' else None,
                intervening_writes=int(mode == 'older'), cycle=0)
            cases.append(dict(document=index, mode=mode, nuisance_value=ord(nuisance),
                query=asdict(query), variants=variants, answers=list(map(ord, config.values)),
                query_positions=[position]))
    return cases


def training_cases(config, *, step, start, seed, gaps, distractor_bytes):
    """Sixteen answer targets per four-value update; never reuse a base index."""
    mode = MODES[step % len(MODES)]
    gap = gaps[step % len(gaps)]
    count = len(config.values) if mode == 'single' else 1
    cases = selective_cases(config, split='train', seed=seed, start=start, count=count,
                           mode=mode, gap=gap, distractor_bytes=distractor_bytes)
    return cases, mode, gap, start+count


def summarize_values(result):
    summary = {k:v for k,v in result.items() if k != 'rows'}
    values = {}
    for row in result['rows']:
        for answer, prediction in zip(row['answers'], row['predictions']):
            item = values.setdefault(chr(answer), dict(targets=0, correct=0, predictions={}))
            item['targets'] += 1
            item['correct'] += int(answer == prediction)
            key = chr(prediction)
            item['predictions'][key] = item['predictions'].get(key, 0)+1
    summary['per_value'] = values
    return summary
