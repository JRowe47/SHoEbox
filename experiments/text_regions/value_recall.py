"""Balanced causal recall checks: change only the earlier observed value."""
from dataclasses import asdict
import torch
from torch.nn import functional as F


def value_cases(documents, values='0123', *, acquisition=False, minimum_delay=0,
                maximum_delay=None, one_per_document=False):
    cases = []
    for document in documents:
        for query in document.queries:
            if (query.kind != 'exact' or query.delay < minimum_delay
                    or (maximum_delay is not None and query.delay > maximum_delay)
                    or (acquisition and query.cycle != -1)):
                continue
            if not 0 <= query.source_position < query.position or document.data[query.source_position] != query.answer:
                raise ValueError('Exact query does not identify an earlier observed value.')
            variants = []
            for value in values:
                sequence = bytearray(document.data[:query.position+1])
                sequence[query.source_position] = ord(value)
                variants.append(bytes(sequence))
            cases.append(dict(document=document.index, query=asdict(query),
                              variants=variants, answers=list(map(ord, values))))
            if one_per_document:
                break
    if not cases:
        raise ValueError('No eligible value-conditioned recall cases.')
    return cases


@torch.no_grad()
def evaluate_values(model, cases, *, context=32, cases_per_batch=8):
    if context < 1 or cases_per_batch < 1:
        raise ValueError('Positive context and batch required.')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = []
    try:
        for begin in range(0, len(cases), cases_per_batch):
            selected = cases[begin:begin+cases_per_batch]
            sequences = [data for case in selected for data in case['variants']]
            length = max(map(len, sequences))
            x = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
            valid = torch.zeros_like(x, dtype=torch.bool)
            reset = torch.zeros_like(valid)
            for row, data in enumerate(sequences):
                offset = length-len(data)
                x[row, offset:] = torch.tensor(list(data), device=device)
                valid[row, offset:] = True; reset[row, offset] = True
            state = None
            for start in range(0, length, context):
                mask = valid[:, start:start+context]
                result = model(x[:, start:start+context], state=state, reset=reset[:, start:start+context],
                    availability={v: mask for v in model.cfg.views},
                    node_masks={n: mask for n in model.nodes})
                state = result['state']
            logits = result['logits'][:, -1]
            offset = 0
            for case in selected:
                size = len(case['answers']); scores = logits[offset:offset+size]; offset += size
                answers = torch.tensor(case['answers'], device=device)
                prediction = scores.argmax(-1)
                candidate = scores[:, answers]
                diagonal = candidate.diagonal()
                # A constant answer preference cancels exactly over the full
                # value set; only input-conditioned discrimination contributes.
                margin = (diagonal-(candidate.sum(-1)-diagonal)/(size-1)).mean()
                rows.append(dict(document=case['document'], query=case['query'],
                    answers=case['answers'], predictions=prediction.tolist(),
                    nll=F.cross_entropy(scores, answers, reduction='none').tolist(),
                    correct=int((prediction == answers).sum()), complete=bool((prediction == answers).all()),
                    changes_answer=len(set(prediction.tolist())) > 1, value_margin=float(margin)))
    finally:
        model.train(was_training)
    count = sum(len(row['answers']) for row in rows)
    return dict(cases=len(rows), targets=count, accuracy=sum(row['correct'] for row in rows)/count,
        nll=sum(sum(row['nll']) for row in rows)/count,
        complete_set_accuracy=sum(row['complete'] for row in rows)/len(rows),
        answer_switch_fraction=sum(row['changes_answer'] for row in rows)/len(rows),
        value_margin=sum(row['value_margin'] for row in rows)/len(rows), rows=rows)
