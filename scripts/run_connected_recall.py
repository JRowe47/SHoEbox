"""Matched ordinary-stream continuation: connected vs 16-byte content credit."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.content_credit_train import ContentCreditRun
from experiments.text_regions.task_stream import TaskStreamConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=ROOT/'artifacts/state_credit/state/best.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=('connected', 'cut_16'), required=True)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=8)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-segments', action='store_true')
    a = p.parse_args()
    stop = a.steps if a.stop_after is None else a.stop_after
    if min(a.steps, a.horizon, a.batch, a.eval_every) < 1 or not 1 <= stop <= a.steps:
        p.error('Positive budgets and a stopping step within the protocol are required.')
    if a.horizon % 16:
        p.error('Matched truncation control requires a multiple of 16 bytes.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()):
        p.error('A new continuation requires an empty directory.')
    torch.set_num_threads(1)
    saved = torch.load(a.parent, map_location='cpu', weights_only=True)
    protocol = dict(steps=a.steps, context=a.horizon, batch=a.batch, lr=.001, seed=97, data_seed=583,
        documents=1024, eval_every=a.eval_every, eval_count=4, eval_chunks=1, truncate_every=0,
        execution_segment=16, cut_at_segments=a.arm == 'cut_16', checkpoint_segments=a.checkpoint_segments,
        audit_coverage=True, syntax_weight=.1, gradient_mode='sum', gradient_cap=1.,
        natural_context=True, delayed_credit=True, state_credit=True,
        parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest())
    run = ContentCreditRun(a.data.read_bytes(), GraphConfig.from_dict(saved['config']),
        TaskStreamConfig.from_dict(saved['data_config']), protocol, a.parent)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        run.restore(a.out/'last.pt')
        if not (a.out/'best.pt').exists():
            p.error('Resume requires the selected checkpoint.')
    else:
        print(json.dumps(dict(arm=a.arm, stage='initial_validation')), flush=True)
        initial = run.validation(); run.best = initial['score']
        run.save(a.out/'best.pt'); run.save(a.out/'last.pt')
        (a.out/'initial.json').write_text(json.dumps(initial, indent=2)+'\n')
        (a.out/'protocol.json').write_text(json.dumps(dict(protocol=protocol,
            config=saved['config'], data_config=saved['data_config'], initialization=run.initialization,
            raw_sha256=run.raw_digest, parameters=sum(p.numel() for p in run.parameters),
            torch_version=str(torch.__version__),
            selection='Minimum mean validation language/answer NLL at 0, eval_every and final; earlier ties win',
            training='Unchanged natural text and four-entity task grammar, no new curriculum'), indent=2)+'\n')
    if stop < run.step:
        p.error('Stop precedes the saved update.')
    started = time.perf_counter()
    while run.step < stop:
        row = run.advance()
        if run.step % a.eval_every == 0 or run.step == a.steps:
            val = run.validation()
            if val['score'] < run.best:
                run.best = val['score']; run.save(a.out/'best.pt')
            row['validation'] = val
        row.update(invocation_seconds=time.perf_counter()-started,
                   peak_process_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(arm=a.arm, step=run.step, seconds=row['invocation_seconds'],
            language_nll=row['language']['primary_nll'], answer_nll=row['memory']['primary_nll'],
            validation=row.get('validation', {}).get('score'), peak_mib=row['peak_process_mib'])), flush=True)


if __name__ == '__main__':
    main()
