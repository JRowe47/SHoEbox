"""Shared-weight language and memory training with independent resumable states.

Odd updates consume ordered natural text, even updates consume fresh procedural
memory documents. Both use the ordinary byte interface, the same owned memory,
and the same sparse experts. Task identities never enter the model. Each stream
owns its runtime state, cursor and version-bound delayed-credit trace.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .data import OrderedByteStreams, split_bytes
from .task_curriculum import window_forward
from .task_memory import TaskMemoryConfig, TaskMemoryModel, TaskMemoryState
from .task_memory_train import byte_loss, evaluate, initialize_weights, objective
from .task_payload import make_eligibility
from .task_stream import TaskByteStreams, TaskChunk, TaskStreamConfig, make_documents
from .mixed_gradients import clip_mixed_gradients


def source_for_step(step):
    if step < 1:
        raise ValueError('Update indices begin at one.')
    return 'language' if step % 2 else 'memory'


def language_chunk(chunk):
    absent = torch.full_like(chunk.values, -100)
    return TaskChunk(chunk.values, chunk.targets, absent, absent.ne(-100), chunk.reset,
                     torch.full_like(chunk.values, -1), torch.full_like(chunk.values, -1))


@torch.no_grad()
def evaluate_language(model, raw, batch=4, context=64, chunks=4, control='intact'):
    streams = OrderedByteStreams.from_bytes(raw, batch)
    state, total, count = None, 0., 0
    occupied = unique = events = 0.
    was_training = model.training
    model.eval()
    try:
        for _ in range(chunks):
            chunk = streams.next_chunk(context)
            result = model(chunk.values, state=state, reset=chunk.reset, control=control)
            state = result['state']
            total += float(F.cross_entropy(result['logits'].flatten(0, 1), chunk.targets.flatten(), reduction='sum'))
            count += chunk.targets.numel()
            if model.cfg.recall_slots:
                weights = result['recall_weights'][..., :-1]
                occupied += float(weights.sum())
                unique += float(weights.sum(2).clamp(max=1).sum())
                events += chunk.values.numel()
        return dict(nll=total/count, targets=count, mean_active_slots=occupied/max(events, 1),
                    mean_unique_owners=unique/max(events, 1), state_bytes=state.storage_bytes())
    finally:
        model.train(was_training)


class MixedRun:
    """Own all resumable training state; model inputs contain bytes only."""
    kind = 'mixed_recall'
    model_type = TaskMemoryModel
    state_type = TaskMemoryState
    credit_factory = staticmethod(make_eligibility)
    forward_window = staticmethod(window_forward)

    def initialize(self, parent):
        return initialize_weights(self.model, parent) if parent else None

    def apply_credit(self, boundaries):
        delayed = sum(boundary.apply(self.model.write_policy) for boundary in boundaries)
        payload = sum(boundary.apply_payload(self.model.payload_adapter, True)
                      for boundary in boundaries) if self.model.cfg.payload_rank else 0.
        return delayed, payload

    def language_metrics(self, raw, chunks=None):
        p = self.protocol
        return evaluate_language(self.model, raw, p['batch'], p['context'], chunks or p['eval_chunks'])

    def memory_metrics(self, split='validation', count=None):
        p = self.protocol
        return evaluate(self.model, self.data_config, p['data_seed'], count or p['eval_count'], p['context'], split)

    def __init__(self, raw, config, data_config, protocol, parent=None):
        self.protocol = dict(protocol)
        self.data_config = data_config
        self.raw_digest = hashlib.sha256(raw).hexdigest()
        self.train_raw, self.validation_raw, self.test_raw = split_bytes(raw)
        torch.manual_seed(protocol['seed'])
        self.model = self.model_type(config)
        self.initialization = self.initialize(parent)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=protocol['lr'],
                                          betas=(.9, .95), weight_decay=.01)
        documents = make_documents(data_config, seed=protocol['data_seed'], count=protocol['documents'])
        batch = protocol['batch']
        self.streams = dict(language=OrderedByteStreams.from_bytes(self.train_raw, batch),
                            memory=TaskByteStreams(documents, batch))
        self.states = {key: self.model.initial_state(batch) for key in self.streams}
        self.credits = {key: self.credit_factory(self.model, batch) for key in self.streams}
        self.totals = {key: dict(updates=0, bytes=0, experts=0, answers=0,
            delayed_gradient_norm_sum=0., payload_gradient_norm_sum=0.) for key in self.streams}
        self.step, self.best = 0, float('inf')

    def advance(self):
        self.model.train()
        step = self.step + 1
        source = source_for_step(step)
        chunk = self.streams[source].next_chunk(self.protocol['context'])
        if source == 'language':
            chunk = language_chunk(chunk)
        self.optimizer.zero_grad(set_to_none=True)
        result, credit, boundaries = self.forward_window(self.model, chunk, self.states[source],
            self.credits[source], truncate_every=self.protocol['truncate_every'])
        if source == 'language':
            primary = byte_loss(result['logits'], chunk.ordinary_targets)
            loss = primary + .001 * result['aux']
        else:
            loss, primary, _ = objective(result, chunk, self.protocol['syntax_weight'])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite mixed training objective.')
        loss.backward()
        delayed, payload = self.apply_credit(boundaries)
        norm, norm_recovered = clip_mixed_gradients(self.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite mixed training gradient.')
        self.optimizer.step()
        self.states[source], self.credits[source] = result['state'].detach(), credit
        totals = self.totals[source]
        if norm_recovered:
            totals['fp64_norm_recoveries'] = totals.get('fp64_norm_recoveries', 0) + 1
        for key, amount in dict(updates=1, bytes=result['external_events'],
                experts=result['expert_executions'], answers=int(chunk.answer_mask.sum()),
                delayed_gradient_norm_sum=delayed, payload_gradient_norm_sum=payload).items():
            totals[key] += amount
        self.step = step
        return dict(step=step, source=source, primary_nll=float(primary.detach()),
                    gradient_norm=float(norm), fp64_norm_recovered=norm_recovered, totals=self.totals)

    def validation(self):
        p = self.protocol
        language = self.language_metrics(self.validation_raw)
        memory, _ = self.memory_metrics()
        # Same scale: nats per natural byte and nats per answer byte. The fixed
        # equal-weight criterion rewards both tasks; each metric is also exposed.
        return dict(language=language, memory=memory,
                    score=(language['nll'] + memory['answer_nll']) / 2)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(dict(schema=1, kind=self.kind, config=self.model.cfg.to_dict(),
            data_config=self.data_config.to_dict(), protocol=self.protocol, raw_sha256=self.raw_digest,
            weights=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            states={k: s.to_dict() for k, s in self.states.items()},
            credits={k: c.to_dict() for k, c in self.credits.items()},
            streams={k: s.state_dict() for k, s in self.streams.items()},
            rng=torch.get_rng_state(), step=self.step, best=self.best, totals=self.totals,
            initialization=self.initialization), temporary)
        temporary.replace(path)

    def restore(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('schema') != 1 or saved.get('kind') != self.kind:
            raise ValueError('Not a mixed recall checkpoint.')
        for key, expected in dict(config=self.model.cfg.to_dict(), data_config=self.data_config.to_dict(),
                protocol=self.protocol, raw_sha256=self.raw_digest).items():
            if saved[key] != expected:
                raise ValueError(f'Mixed resume {key} mismatch.')
        for key in self.streams:
            self.streams[key].load_state_dict(saved['streams'][key])
        self.model.load_state_dict(saved['weights'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.states = {k: self.state_type.from_dict(s) for k, s in saved['states'].items()}
        self.credits = {k: self.credit_factory(self.model, self.protocol['batch'], c)
                        for k, c in saved['credits'].items()}
        self.step, self.best, self.totals = saved['step'], saved['best'], saved['totals']
        self.initialization = saved['initialization']
        torch.set_rng_state(saved['rng'])


def final_evaluation(run, out):
    """Separate selected/final metrics and whole-rollout recall interventions."""
    output = {}
    for tag in ('best', 'last'):
        saved = torch.load(out/(tag+'.pt'), map_location='cpu', weights_only=True)
        run.model.load_state_dict(saved['weights'])
        p = run.protocol
        language = evaluate_language(run.model, run.test_raw, p['batch'], p['context'], p['eval_chunks']*2)
        memory = {}
        for split in ('test', 'long_delay'):
            metrics, traces = evaluate(run.model, run.data_config, p['data_seed'], p['eval_count']*2,
                                       p['context'], split)
            memory[split] = metrics
            (out/f'{tag}_{split}_traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traces))
        interventions = {}
        original = run.model.cfg
        try:
            for control in ('pooled', 'no_feedback', 'blocked'):
                run.model.cfg = replace(original,
                    recall_layout='pooled' if control == 'pooled' else original.recall_layout,
                    recall_feedback=False if control == 'no_feedback' else original.recall_feedback)
                mode = 'blocked' if control == 'blocked' else 'intact'
                language_control = evaluate_language(run.model, run.test_raw, p['batch'], p['context'],
                                                      p['eval_chunks']*2, control=mode)
                memory_control, _ = evaluate(run.model, run.data_config, p['data_seed'],
                    p['eval_count']*2, p['context'], 'test', control=mode)
                interventions[control] = dict(language_nll=language_control['nll'],
                    answer_nll=memory_control['answer_nll'],
                    language_delta=language_control['nll']-language['nll'],
                    answer_delta=memory_control['answer_nll']-memory['test']['answer_nll'])
        finally:
            run.model.cfg = original
        output[tag] = dict(step=saved['step'], language=language, memory=memory,
                          interventions=interventions, totals=saved['totals'])
    (out/'evaluation.json').write_text(json.dumps(output, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--init-from', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, default=320)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--owners', type=int, default=4)
    parser.add_argument('--recall-slots', type=int, default=2)
    parser.add_argument('--recall-layout', choices=('distinct', 'pooled'), default='distinct')
    parser.add_argument('--state-commit', choices=('always', 'publication'), default='publication')
    parser.add_argument('--payload-rank', type=int, default=4)
    parser.add_argument('--truncate-every', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--syntax-weight', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--data-seed', type=int, default=271)
    parser.add_argument('--documents', type=int, default=4096)
    parser.add_argument('--eval-every', type=int, default=80)
    parser.add_argument('--eval-count', type=int, default=8)
    parser.add_argument('--eval-chunks', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--final-eval', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.context, args.batch, args.eval_every, args.eval_count,
           args.eval_chunks, args.threads, args.documents) < 1 or args.lr <= 0 or args.syntax_weight < 0:
        parser.error('Positive budgets, learning rate and nonnegative syntax weight required.')
    stop = args.steps if args.stop_after is None else args.stop_after
    if not 1 <= stop <= args.steps or args.documents < args.batch:
        parser.error('Invalid stopping step or insufficient documents.')
    if args.truncate_every < 0 or (args.truncate_every and args.context % args.truncate_every):
        parser.error('Truncation must be zero or a positive divisor of context.')
    if args.resume and args.init_from:
        parser.error('Resume and weight initialization are mutually exclusive.')
    if args.evaluate_only and not args.resume:
        parser.error('Evaluation-only requires a resumable checkpoint.')
    if args.resume and args.resume.resolve() != (args.out/'last.pt').resolve():
        parser.error('Resume the last checkpoint in its existing output directory.')
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        parser.error('A new run requires an empty output directory.')
    torch.set_num_threads(args.threads)
    raw = args.data.read_bytes()
    config = TaskMemoryConfig(width=args.width, owners=args.owners, recall_slots=args.recall_slots,
        recall_layout=args.recall_layout, payload_rank=args.payload_rank,
        state_commit=args.state_commit, write_mode='competitive', memory_aware_write=True,
        bootstrap_empty=True)
    data_config = TaskStreamConfig(entities='ABCD', queries_per_cycle=4, delays=(2,48), interference=True)
    protocol = {name: getattr(args, name) for name in ('steps', 'context', 'batch', 'lr',
        'syntax_weight', 'seed', 'data_seed', 'documents', 'eval_every', 'eval_count',
        'eval_chunks', 'threads', 'truncate_every')}
    run = MixedRun(raw, config, data_config, protocol, args.init_from)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run.restore(args.resume)
        if not (args.out/'best.pt').exists():
            parser.error('Resume requires the accompanying best checkpoint.')
        if args.evaluate_only:
            final_evaluation(run, args.out)
            return
        if stop <= run.step:
            parser.error('Stopping step must follow the saved step.')
    else:
        initial = run.validation()
        run.best = initial['score']
        run.save(args.out/'best.pt')
        run.save(args.out/'last.pt')
        (args.out/'initial.json').write_text(json.dumps(initial, indent=2)+'\n')
        (args.out/'protocol.json').write_text(json.dumps(dict(protocol=protocol, config=config.to_dict(),
            data_config=data_config.to_dict(), raw_sha256=run.raw_digest,
            data_bytes=len(raw), parameters=sum(p.numel() for p in run.model.parameters()),
            initialization=run.initialization, schedule='odd language, even memory; constant learning rate',
            selection='mean of validation natural-byte NLL and memory-answer NLL',
            source_states='independent runtime and credit per stream; shared trainable weights'), indent=2)+'\n')
    started = time.perf_counter()
    while run.step < stop:
        record = run.advance()
        if run.step % args.eval_every == 0 or run.step == args.steps:
            validation = run.validation()
            if validation['score'] < run.best:
                run.best = validation['score']
                run.save(args.out/'best.pt')
            run.save(args.out/'last.pt')
            record.update(validation=validation, invocation_seconds=time.perf_counter()-started)
            with (args.out/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print(json.dumps(dict(step=run.step, language_nll=validation['language']['nll'],
                answer_nll=validation['memory']['answer_nll'], seconds=record['invocation_seconds'])), flush=True)
        elif run.step == stop:
            # A pause between scheduled evaluations must not create an extra
            # checkpoint-selection opportunity relative to an uninterrupted run.
            run.save(args.out/'last.pt')
            print(json.dumps(dict(step=run.step, paused=True)), flush=True)
    if args.final_eval:
        final_evaluation(run, args.out)


if __name__ == '__main__':
    main()
