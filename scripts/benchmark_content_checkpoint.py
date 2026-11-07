"""Measure one paired update in an isolated process, with resumable output."""
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
    p.add_argument('--horizon', type=int, default=32)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--checkpoint-segments', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        p.error('Use a new benchmark output directory.')
    torch.set_num_threads(1)
    saved = torch.load(a.parent, map_location='cpu', weights_only=True)
    protocol = dict(steps=1, context=a.horizon, batch=a.batch, lr=.001, seed=97, data_seed=583,
        documents=1024, eval_every=1, eval_count=4, eval_chunks=1, truncate_every=0,
        execution_segment=16, cut_at_segments=False, checkpoint_segments=a.checkpoint_segments,
        audit_coverage=True, syntax_weight=.1, gradient_mode='sum', gradient_cap=1.,
        natural_context=True, delayed_credit=True, state_credit=True)
    run = ContentCreditRun(a.data.read_bytes(), GraphConfig.from_dict(saved['config']),
        TaskStreamConfig.from_dict(saved['data_config']), protocol, a.parent)
    started = time.perf_counter()
    record = run.advance()
    result = dict(seconds=time.perf_counter()-started, protocol=protocol, update=record,
        peak_process_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        torch_version=str(torch.__version__), parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest())
    a.out.mkdir(parents=True)
    run.save(a.out/'last.pt')
    (a.out/'benchmark.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('seconds','peak_process_mib')}), flush=True)


if __name__ == '__main__':
    main()
