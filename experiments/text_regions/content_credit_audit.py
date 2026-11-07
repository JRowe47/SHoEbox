"""Training-only provenance audit; no query metadata enters the graph."""


def window_coverage(chunk, documents, segment):
    """Count source/answer pairs connected by exact autograd, not just delay.

    A delay shorter than the horizon can still cross its boundary. Check the
    actual source occurrence and intervening resets in the sampled stream.
    Counts describe available paths, not useful gradients or successful recall.
    """
    groups = {name: dict(answers=0, within_window=0, within_segment=0)
              for name in ('all', 'exact', 'acquisition', 'exact_beyond_96')}
    queries = {i: {q.position: q for q in doc.queries} for i, doc in enumerate(documents)}
    for row, offset in chunk.answer_mask.nonzero().tolist():
        document = int(chunk.document_ids[row, offset])
        position = int(chunk.positions[row, offset])
        query = queries[document][position]
        source = offset-query.delay
        connected = (source >= 0 and int(chunk.document_ids[row, source]) == document
                     and int(chunk.positions[row, source]) == query.source_position
                     and not bool(chunk.reset[row, source+1:offset+1].any()))
        local = connected and source//segment == offset//segment
        labels = ['all']
        if query.kind == 'exact':
            labels.append('exact')
            if query.cycle == -1:
                labels.append('acquisition')
            if query.delay > 96:
                labels.append('exact_beyond_96')
        for label in labels:
            counts = groups[label]
            counts['answers'] += 1
            counts['within_window'] += int(connected)
            counts['within_segment'] += int(local)
    return groups
