"""Paired language/task updates with exact cross-chunk content gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from .config import GraphConfig
from .state_credit_train import StateCreditRun
from .state_credit import state_window_forward
from .content_credit import content_window_forward
from .content_credit_audit import window_coverage
from .task_stream import TaskStreamConfig


class ContentCreditRun(StateCreditRun):
    kind = 'connected_content_credit'

    def __init__(self, raw, config, data_config, protocol, parent=None):
        if protocol['truncate_every'] != 0 or protocol['execution_segment'] < 1:
            raise ValueError('Connected content credit detaches only at the optimizer window boundary.')
        if protocol['context'] < 1:
            raise ValueError('Gradient horizon must be positive.')
        super().__init__(raw, config, data_config, protocol, parent=None)
        if parent is not None:
            saved = torch.load(parent, map_location='cpu', weights_only=True)
            if saved['config'] != config.to_dict() or saved['raw_sha256'] != self.raw_digest:
                raise ValueError('Connected continuation graph/corpus mismatch.')
            self.initialization = self.initialize(parent)
            self.optimizer.load_state_dict(saved['optimizer'])
            for group in self.optimizer.param_groups:
                group['lr'] = protocol['lr']
            self.initialization.update(optimizer_transferred=True, source_states='fresh', source_cursors='fresh')

    def forward_window(self, model, chunk, state, credit, truncate_every=0):
        if truncate_every:
            raise ValueError('Execution segmentation must not set the gradient horizon.')
        segment = self.protocol['execution_segment']
        checkpoint_segments = self.protocol.get('checkpoint_segments', False)
        if self.protocol.get('cut_at_segments', False):
            return state_window_forward(model, chunk, state, credit, truncate_every=segment,
                                        checkpoint_segments=checkpoint_segments)
        return content_window_forward(model, chunk, state, credit, segment=segment,
                                      checkpoint_segments=checkpoint_segments)

    def chunk_metrics(self, source, chunk):
        if source != 'memory' or not self.protocol.get('audit_coverage', False):
            return {}
        counts = window_coverage(chunk, self.streams[source].documents,
                                 self.protocol['execution_segment'])
        totals = self.totals[source].setdefault('coverage', {})
        for group, row in counts.items():
            summed = totals.setdefault(group, {key: 0 for key in row})
            for key, value in row.items():
                summed[key] += value
        return dict(coverage=counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--horizon', type=int, default=384)
    p.add_argument('--segment', type=int, default=16)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=97)
    p.add_argument('--data-seed', type=int, default=583)
    p.add_argument('--cut-at-segments', action='store_true', help='Matched old-path ablation only')
    p.add_argument('--checkpoint-segments', action='store_true',
                   help='Recompute segment activations during backward to reduce memory')
    p.add_argument('--audit-coverage', action='store_true')
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    stop = a.stop_after or a.steps
    if min(a.steps, a.batch, a.horizon, a.segment, a.eval_every) < 1 or not 1 <= stop <= a.steps:
        p.error('Positive counts and a valid stopping step are required.')
    if a.cut_at_segments and a.horizon % a.segment:
        p.error('The old-path ablation requires segment to divide horizon.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()):
        p.error('Fresh runs require an empty output directory.')
    torch.set_num_threads(1)
    saved = torch.load(a.parent, map_location='cpu', weights_only=True)
    config, data_config = GraphConfig.from_dict(saved['config']), TaskStreamConfig.from_dict(saved['data_config'])
    protocol = dict(steps=a.steps, context=a.horizon, batch=a.batch, lr=a.lr, seed=a.seed, data_seed=a.data_seed,
        documents=1024, eval_every=a.eval_every, eval_count=4, eval_chunks=4, truncate_every=0,
        execution_segment=a.segment, cut_at_segments=a.cut_at_segments,
        syntax_weight=.1, gradient_mode='sum', gradient_cap=1., natural_context=True,
        delayed_credit=True, state_credit=True, parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest())
    if a.checkpoint_segments:
        protocol['checkpoint_segments'] = True
    if a.audit_coverage:
        protocol['audit_coverage'] = True
    run = ContentCreditRun(a.data.read_bytes(), config, data_config, protocol, a.parent)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        run.restore(a.out/'last.pt')
        if not (a.out/'best.pt').exists():
            p.error('Resume requires best.pt.')
    else:
        initial = run.validation()
        run.best = initial['score']
        run.save(a.out/'best.pt'); run.save(a.out/'last.pt')
        (a.out/'initial.json').write_text(json.dumps(initial, indent=2)+'\n')
        (a.out/'protocol.json').write_text(json.dumps(dict(protocol=protocol, config=config.to_dict(),
            data_config=data_config.to_dict(), parameters=sum(p.numel() for p in run.parameters),
            initialization=run.initialization, raw_sha256=run.raw_digest,
            gradient_horizon='Full optimizer window; execution chunks do not detach',
            historical_credit='One entry receipt for older conditional gate/payload credit; no internal double counting',
            clipping='Unchanged combined global norm cap 1.0',
            selection='Mean validation language and answer NLL; ties retain earlier checkpoint'), indent=2)+'\n')
    if stop < run.step:
        p.error('Stop precedes checkpoint.')
    started = time.perf_counter()
    while run.step < stop:
        row = run.advance()
        if run.step % a.eval_every == 0 or run.step == a.steps:
            validation = run.validation()
            if validation['score'] < run.best:
                run.best = validation['score']; run.save(a.out/'best.pt')
            row['validation'] = validation
        row['seconds'] = time.perf_counter()-started
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(dict(step=run.step, seconds=row['seconds'],
            gradient_norm=row['gradient_norm'], language_nll=row['language']['primary_nll'],
            memory_nll=row['memory']['primary_nll'], validation=row.get('validation', {}).get('score'))), flush=True)


if __name__ == '__main__':
    main()
