"""One real paired-source update at the repaired 384-byte training horizon."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.content_credit_train import ContentCreditRun
from experiments.text_regions.data import split_bytes
from experiments.text_regions.graph_mixed_train import evaluate_language
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents


@torch.no_grad()
def probes(model, raw, data_config):
    _, _, test = split_bytes(raw)
    docs = make_documents(data_config, split='test', seed=483, count=8, distractor_bytes=test)
    stop = max(q.position for q in docs[0].queries if q.cycle == -1)+1
    assert all(max(q.position for q in d.queries if q.cycle == -1)+1 == stop for d in docs)
    result = model(torch.tensor([list(d.data[:stop]) for d in docs]))
    losses, hits, predictions = [], [], []
    for row, doc in enumerate(docs):
        for q in doc.queries:
            if q.cycle != -1: continue
            logits = result['logits'][row, q.position]
            prediction = int(logits.argmax())
            predictions.append(chr(prediction)); hits.append(prediction == q.answer)
            losses.append(float(F.cross_entropy(logits[None], torch.tensor([q.answer]))))
    return dict(acquisition=dict(count=len(hits), accuracy=sum(hits)/len(hits), nll=sum(losses)/len(losses),
                                 predictions=dict(Counter(predictions))),
                language=evaluate_language(model, test, batch=1, context=64, chunks=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=ROOT/'artifacts/state_credit/state/best.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/content_credit/smoke')
    p.add_argument('--horizon', type=int, default=384)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use a fresh smoke output directory.')
    torch.set_num_threads(1)
    saved = torch.load(a.parent, map_location='cpu', weights_only=True)
    raw = a.data.read_bytes()
    data_config = TaskStreamConfig.from_dict(saved['data_config'])
    protocol = dict(steps=1, context=a.horizon, batch=1, lr=.001, seed=97, data_seed=583,
        documents=1024, eval_every=1, eval_count=4, eval_chunks=4, truncate_every=0,
        execution_segment=16, cut_at_segments=False, syntax_weight=.1, gradient_mode='sum',
        gradient_cap=1., natural_context=True, delayed_credit=True, state_credit=True,
        parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest())
    run = ContentCreditRun(raw, GraphConfig.from_dict(saved['config']), data_config, protocol, a.parent)
    a.out.mkdir(parents=True, exist_ok=True)
    before = probes(run.model, raw, data_config)
    old = {n:p.detach().clone() for n,p in run.model.named_parameters()}
    run.save(a.out/'initial.pt')
    start = time.perf_counter()
    print(json.dumps(dict(stage='paired_update', horizon=a.horizon)), flush=True)
    row = run.advance()
    seconds = time.perf_counter()-start
    run.save(a.out/'last.pt')
    after = probes(run.model, raw, data_config)
    changed = [n for n,p in run.model.named_parameters() if not torch.equal(old[n], p)]
    assert all(bool(p.isfinite().all()) for p in run.model.parameters())
    assert row['language']['answers'] == 0 and row['memory']['answers'] > 0
    output = dict(protocol=protocol, parent_sha256=protocol['parent_sha256'],
        raw_sha256=hashlib.sha256(raw).hexdigest(), before=before, after=after, update=row,
        seconds=seconds, peak_process_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        changed_parameter_tensors=len(changed), parameters=sum(p.numel() for p in run.parameters),
        checkpoint='last.pt is a smoke-test continuation, not a validation-selected research result')
    (a.out/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(completed=run.step, seconds=seconds, peak_process_mib=output['peak_process_mib'],
        changed_tensors=len(changed), gradient_norm=row['gradient_norm'], answers=row['memory']['answers'])), flush=True)


if __name__ == '__main__':
    main()
