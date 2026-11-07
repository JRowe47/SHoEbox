"""Acquisition curriculum and matched gradient-window training.

The model sees ordinary bytes only. Stages control training-document generation;
query/source annotations are used exclusively for coverage accounting. Windows
contain one optimizer update regardless of the number of detach boundaries.
"""
from dataclasses import replace

import torch


def stage_for_step(step, curriculum='none', acquire_steps=200, bridge_steps=200):
    if step < 0 or min(acquire_steps, bridge_steps) < 1:
        raise ValueError('Nonnegative step and positive stage lengths required.')
    if curriculum == 'none':
        return 'target'
    if curriculum != 'acquisition':
        raise ValueError('Unknown curriculum.')
    if step <= acquire_steps:
        return 'acquire'
    return 'bridge' if step <= acquire_steps + bridge_steps else 'target'


def stage_config(target, stage):
    if stage == 'acquire':
        return replace(target, delays=(0, 2), interference=False, acquisition_queries=True)
    if stage == 'bridge':
        return replace(target, delays=(2, 16), acquisition_queries=False)
    if stage == 'target':
        return target
    raise ValueError('Unknown training stage.')


def window_forward(model, chunk, state, credit, control='intact', truncate_every=0):
    """One byte window, optionally detached into matched smaller graphs.

    Each boundary gets its own delayed-credit receipt. Aggregate the answer and
    text losses over the whole window before backward, then consume receipts
    once. The truncated arm retains the existing conditional write-head credit;
    it loses historical payload/feature and recurrent gradients only.
    """
    length = chunk.values.shape[1]
    if truncate_every < 0 or (truncate_every and length % truncate_every):
        raise ValueError('Truncation must be zero or a positive divisor of the window.')
    span = truncate_every or length
    results, boundaries = [], []
    for start in range(0, length, span):
        boundary = credit.capture(state.memory_tensor())
        boundaries.append(boundary)
        state = state.with_memory_tensor(boundary.memory)
        result = model(chunk.values[:, start:start+span], state=state,
                       reset=chunk.reset[:, start:start+span], control=control)
        credit = credit.advance_chunk(result)
        results.append(result)
        state = result['state'].detach()
    combined = dict(logits=torch.cat([r['logits'] for r in results], 1),
                    write_probs=torch.cat([r['write_probs'] for r in results], 1),
                    aux=torch.stack([r['aux'] for r in results]).mean(),
                    state=state,
                    external_events=sum(r['external_events'] for r in results),
                    producer_candidates=sum(r['producer_candidates'] for r in results),
                    producer_commits=sum(r['producer_commits'] for r in results),
                    expert_executions=sum(r['expert_executions'] for r in results))
    return combined, credit, boundaries


class SourceCoverage:
    """Count observations and questions sharing each live gradient segment."""
    def __init__(self, documents):
        self.queries = {}
        for document_id, document in enumerate(documents):
            sources = {event.stop-2: event.start for event in document.events
                       if event.kind in ('observation', 'correction')}
            for query in document.queries:
                if query.kind == 'exact':
                    self.queries[document_id, query.position] = (
                        query.source_position, sources[query.source_position])

    def count(self, chunk, truncate_every=0):
        length = chunk.values.shape[1]
        if truncate_every < 0 or (truncate_every and length % truncate_every):
            raise ValueError('Invalid gradient segment length.')
        span = truncate_every or length
        counts = dict(exact_queries=0, source_value_connected=0, observation_connected=0)
        for ids, positions in zip(chunk.document_ids.tolist(), chunk.positions.tolist()):
            seen = set()
            for index, (document_id, position) in enumerate(zip(ids, positions)):
                if index % span == 0:
                    seen.clear()
                seen.add((document_id, position))
                source = self.queries.get((document_id, position))
                if source is not None:
                    counts['exact_queries'] += 1
                    counts['source_value_connected'] += int((document_id, source[0]) in seen)
                    counts['observation_connected'] += int((document_id, source[1]) in seen)
        return counts
