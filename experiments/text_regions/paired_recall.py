"""Two native byte questions fork one live, differentiable event history.

Answers, branch identities and source positions are supervision metadata only.
No answer or externally supplied owner index enters the model. Every template
crosses the full Cartesian product of the two observed values.
"""
from dataclasses import replace
import torch

from .memory_selectivity import selective_cases
from .ports import map_tensors


def paired_batch(config, *, split, seed, start, count, gap, distractor_bytes):
    older = selective_cases(config, split=split, seed=seed, start=start, count=count,
        mode='older', gap=gap, distractor_bytes=distractor_bytes)
    newer = selective_cases(config, split=split, seed=seed, start=start, count=count,
        mode='newer', gap=gap, distractor_bytes=distractor_bytes)
    size = len(config.values)
    histories, questions, answers, sources, documents = [], [], [], [], []
    for template in range(count):
        for a in range(size):
            for b in range(size):
                left, right = older[template*size+b], newer[template*size+a]
                q = left['query']; r = right['query']
                n = len(f"{q['entity']} {q['attribute']}?")
                m = len(f"{r['entity']} {r['attribute']}?")
                old, new = left['variants'][a], right['variants'][b]
                if n != m or old[:-n] != new[:-m]:
                    raise ValueError('Paired questions must share one byte history.')
                histories.append(list(old[:-n]))
                questions.extend((list(old[-n:]), list(new[-m:])))
                answers.extend((ord(config.values[a]), ord(config.values[b])))
                sources.append((q['source_position'], r['source_position']))
                documents.append(left['document'])
    history = torch.tensor(histories, dtype=torch.long)
    query = torch.tensor(questions, dtype=torch.long)
    rows = torch.arange(len(histories)).repeat_interleave(2)
    full = torch.cat((history.index_select(0, rows), query), 1)
    syntax = torch.full_like(full, -100)
    syntax[:, :-1] = full[:, 1:]
    return dict(history=history, query=query, rows=rows, answers=torch.tensor(answers),
                syntax=syntax, full=full, sources=sources, documents=documents)


def fork_state(state, rows):
    """Select state rows without detaching, including HAT's batch metadata."""
    if rows.dtype != torch.long or rows.ndim != 1 or rows.numel() == 0:
        raise ValueError('State row selection requires a nonempty int64 vector.')
    if rows.device != state.events_seen.device:
        raise ValueError('State row selection must stay on the state device.')
    result = map_tensors(replace(state, hippocampus=None), lambda x: x.index_select(0, rows))
    if state.hippocampus is not None:
        result = replace(result, hippocampus=state.hippocampus.fork(rows))
    return replace(result, ingress={k:replace(v, batch_size=len(rows))
                                    for k,v in result.ingress.items()})


def paired_forward(model, batch):
    history, query, rows = (batch[k] for k in ('history','query','rows'))
    prefix = model(history)
    branch = model(query, state=fork_state(prefix['state'], rows))
    n, m = history.shape[1], query.shape[1]
    # Every row/event is active. Duplicating prefix rows leaves each router's
    # batch balance statistic unchanged; weight segment means by event count.
    return dict(logits=torch.cat((prefix['logits'].index_select(0, rows), branch['logits']), 1),
        state=branch['state'], history_state=prefix['state'],
        aux=(n*prefix['aux']+m*branch['aux'])/(n+m),
        hat_aux=(n*prefix['hat_aux']+m*branch['hat_aux'])/(n+m),
        executed_positions=history.numel()+query.numel(),
        duplicated_positions=batch['full'].numel())
