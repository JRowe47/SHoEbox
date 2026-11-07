"""Matched paired-source continuation with optional producer-state credit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from .config import GraphConfig
from .context_balance import ContextBalancedRun
from .state_credit import make_state_graph_eligibility, state_window_forward
from .task_stream import TaskStreamConfig


class StateCreditRun(ContextBalancedRun):
    kind = 'producer_state_credit'
    credit_factory = staticmethod(make_state_graph_eligibility)
    forward_window = staticmethod(state_window_forward)

    def apply_credit(self, boundaries):
        delayed, payload = super().apply_credit(boundaries)
        enabled = self.protocol['delayed_credit'] and self.protocol['state_credit']
        self.state_gradient_norm = sum(b.apply_state(self.model.publisher.policy, enabled) for b in boundaries)
        return delayed, payload

    def source_gradient(self, source):
        gradients, record = super().source_gradient(source)
        record['state_gradient_norm'] = self.state_gradient_norm
        totals = self.totals[source]
        totals['state_gradient_norm_sum'] = totals.get('state_gradient_norm_sum', 0.) + self.state_gradient_norm
        return gradients, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--state-credit', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=16)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    stop = args.stop_after or args.steps
    if min(args.steps, args.eval_every) < 1 or not 1 <= stop <= args.steps:
        parser.error('Positive budgets required.')
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        parser.error('Fresh runs require an empty output directory.')
    torch.set_num_threads(1)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    config = GraphConfig.from_dict(parent['config'])
    data = TaskStreamConfig.from_dict(parent['data_config'])
    protocol = dict(steps=args.steps, context=32, batch=4, lr=.001, seed=89, data_seed=483,
        documents=1024, eval_every=args.eval_every, eval_count=4, eval_chunks=4, truncate_every=16,
        syntax_weight=.1, delayed_credit=True, gradient_mode='sum', gradient_cap=1., natural_context=True,
        state_credit=args.state_credit, parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest())
    run = StateCreditRun(args.data.read_bytes(), config, data, protocol, args.parent)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run.restore(args.out/'last.pt')
        if not (args.out/'best.pt').exists():
            parser.error('Resume requires best.pt.')
    else:
        initial = run.validation()
        run.best = initial['score']
        run.save(args.out/'best.pt'); run.save(args.out/'last.pt')
        (args.out/'initial.json').write_text(json.dumps(initial, indent=2)+'\n')
        (args.out/'protocol.json').write_text(json.dumps(dict(protocol=protocol, config=config.to_dict(),
            data_config=data.to_dict(), parameters=sum(p.numel() for p in run.parameters),
            raw_sha256=run.raw_digest, initialization=run.initialization,
            producer_trace_bytes_per_source=run.credits['language'].producer.storage_bytes(),
            selection='Mean validation language and answer NLL; ties retain earlier checkpoint',
            comparison='Both arms carry the same traces; only the state-credit contraction is ablated.'), indent=2)+'\n')
    if stop < run.step:
        parser.error('Stop precedes saved step.')
    start = time.perf_counter()
    while run.step < stop:
        record = run.advance()
        if run.step % args.eval_every == 0 or run.step == args.steps:
            validation = run.validation()
            if validation['score'] < run.best:
                run.best = validation['score']; run.save(args.out/'best.pt')
            record['validation'] = validation
        record['seconds'] = time.perf_counter()-start
        run.save(args.out/'last.pt')
        with (args.out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        if run.step % 8 == 0:
            print(json.dumps(dict(step=run.step, seconds=record['seconds'],
                state_gradient_norm=run.state_gradient_norm,
                validation=record.get('validation', {}).get('score'))), flush=True)
    print(json.dumps(dict(completed=run.step, state_credit=args.state_credit)), flush=True)


if __name__ == '__main__':
    main()
