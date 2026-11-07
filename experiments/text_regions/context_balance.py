"""Paired language/task updates through one graph, with natural delay context.

Each source has its own state and credit. Both gradients see identical weights,
then the existing task/group clipping rule combines them for one AdamW update.
No activations, gradients or examples are replayed on a later update.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch

from . import gradient_balance
from .config import GraphConfig
from .graph_mixed_train import GraphMixedRun, evaluate_memory
from .mixed_recall_train import language_chunk
from .mixed_gradients import clip_mixed_gradients
from .task_memory_train import byte_loss, objective
from .task_stream import TaskByteStreams, TaskStreamConfig, make_documents


def parameter_groups(model):
    """Disjoint controls for experts/routing, byte emission, recall and writes."""
    groups = {key: [] for key in ('core', 'readout', 'recall', 'publication')}
    for i, (name, _) in enumerate(model.named_parameters()):
        group = ('publication' if name.startswith('publisher.') else
                 'recall' if name.startswith('connections.recall.') else
                 'readout' if name.startswith(('readout.', 'readout_mix.', 'readout_norm.')) else 'core')
        groups[group].append(i)
    return groups


def retention_metrics(traces, owners, segment):
    """Publication age, separated from ingress and readout ports and fact age."""
    counts = {name: dict(selected=0., age_sum=0., old=0.) for name in ('all', 'producers')}
    for query in traces:
        for reads in query['reads'].values():
            for weights, valid, occurrences in zip(reads['weights'], reads['valid'], reads['occurrences']):
                for slot in weights:
                    for edge, weight, present, occurrence in zip(reads['edges'], slot[:-1], valid, occurrences):
                        if not present or not weight:
                            continue
                        age = query['position'] - occurrence
                        if age < 0:
                            raise ValueError('Future publication in recall trace.')
                        source = edge.split('__')[0]
                        for group in ('all', 'producers') if source in owners else ('all',):
                            row = counts[group]
                            row['selected'] += weight
                            row['age_sum'] += weight * age
                            row['old'] += weight * (age > segment)
    return {name: dict(selected_slot_reads=row['selected'],
        mean_age=row['age_sum']/max(row['selected'], 1),
        older_than_segment_fraction=row['old']/max(row['selected'], 1)) for name, row in counts.items()}


class ContextBalancedRun(GraphMixedRun):
    kind = 'context_balanced_graph'

    def __init__(self, raw, config, data_config, protocol, parent=None):
        if protocol['gradient_mode'] not in ('sum', 'group_clip'):
            raise ValueError('Unknown paired gradient mode.')
        if not math.isfinite(protocol['gradient_cap']) or protocol['gradient_cap'] <= 0:
            raise ValueError('Finite positive gradient cap required.')
        super().__init__(raw, config, data_config, protocol, parent)
        if parent is not None:
            saved = torch.load(parent, map_location='cpu', weights_only=True)
            if saved.get('raw_sha256') != self.raw_digest:
                raise ValueError('Continuation corpus differs from parent.')
            self.streams['language'].load_state_dict(saved['streams']['language'])
            self.initialization['language_cursor_transferred'] = True
        self.streams['memory'] = TaskByteStreams(self.documents('train', protocol['documents']), protocol['batch'])
        self.parameters = list(self.model.parameters())
        self.groups = parameter_groups(self.model)

    def documents(self, split, count):
        corpus = (self.train_raw if split == 'train' else self.validation_raw
                  if split in ('validation', 'val') else self.test_raw)
        return make_documents(self.data_config, split=split, seed=self.protocol['data_seed'], count=count,
            distractor_bytes=corpus if self.protocol['natural_context'] else None)

    def memory_metrics(self, split='validation', count=None, control='intact'):
        count = count or self.protocol['eval_count']
        metrics, traces = evaluate_memory(self.model, self.data_config, self.protocol['data_seed'],
            count, self.protocol['context'], split, control, documents=self.documents(split, count))
        metrics['retention'] = retention_metrics(traces, self.model.cfg.task_publication,
            self.protocol['truncate_every'] or self.protocol['context'])
        return metrics, traces

    def source_gradient(self, source):
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
            raise RuntimeError('Nonfinite paired objective.')
        loss.backward()
        delayed, payload = self.apply_credit(boundaries)
        gradients = gradient_balance.gradients(self.parameters)
        # These source states remain independent; neither source sees an update
        # of the other's weights until both backward passes are complete.
        self.states[source], self.credits[source] = result['state'].detach(), credit
        for key, value in dict(updates=1, bytes=result['external_events'], experts=result['expert_executions'],
            answers=int(chunk.answer_mask.sum()), delayed_gradient_norm_sum=delayed,
            payload_gradient_norm_sum=payload).items():
            self.totals[source][key] += value
        return gradients, dict(primary_nll=float(primary.detach()), objective=float(loss.detach()),
            answers=int(chunk.answer_mask.sum()), **self.chunk_metrics(source, chunk))

    def chunk_metrics(self, source, chunk):
        return {}

    def advance(self):
        self.model.train()
        language, language_row = self.source_gradient('language')
        memory, memory_row = self.source_gradient('memory')
        moments = gradient_balance.gradient_moments(language, memory)
        report = gradient_balance.combine(self.parameters, language, memory, self.groups,
            self.protocol['gradient_mode'], self.protocol['gradient_cap'], moments=moments)
        norm, recovered = clip_mixed_gradients(self.parameters, 1.)
        self.optimizer.step()
        self.step += 1
        return dict(step=self.step, language=language_row, memory=memory_row, gradient_groups=report,
            gradient_norm=float(norm), fp64_norm_recovered=recovered, totals=self.totals)


def final_evaluation(run, out):
    """Cache completed evaluations by complete checkpoint and corpus identity."""
    path = out/'evaluation_cache.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    def persist():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(cache, indent=2)+'\n'); temp.replace(path)
    results = {}
    for tag in ('best', 'last'):
        saved = torch.load(out/(tag+'.pt'), map_location='cpu', weights_only=True)
        run.model.load_state_dict(saved['weights'])
        digest = hashlib.sha256(json.dumps(dict(schema=1, config=run.model.cfg.to_dict(),
            protocol=run.protocol, data=run.data_config.to_dict(), corpus=run.raw_digest), sort_keys=True).encode())
        for name, value in sorted(saved['weights'].items()):
            digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        signature = digest.hexdigest()
        row = cache.setdefault(signature, {})
        measurements = ['language', 'test', 'long_delay', 'legacy_test']
        if tag == 'best':
            measurements += ['blocked_test', 'always_write_test', 'no_demand_test', 'always_state_test']
        for key in measurements:
            if key not in row:
                if key == 'language':
                    row[key] = run.language_metrics(run.test_raw, run.protocol['eval_chunks']*2)
                elif key == 'legacy_test':
                    legacy = TaskStreamConfig(entities='ABCD', queries_per_cycle=4, delays=(2,48), interference=True)
                    row[key], _ = evaluate_memory(run.model, legacy, 283, 8, run.protocol['context'], 'test')
                else:
                    control = key.removesuffix('_test') if key.endswith('_test') else 'intact'
                    split = 'long_delay' if key == 'long_delay' else 'test'
                    metrics, traces = run.memory_metrics(split, run.protocol['eval_count']*2, control)
                    row[key] = metrics
                    (out/f'{signature}_{key}_traces.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
                persist()
                print(json.dumps(dict(evaluation=tag, measurement=key)), flush=True)
        results[tag] = dict(step=saved['step'], totals=saved['totals'], signature=signature,
            **{key: row[key] for key in measurements})
    (out/'evaluation.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('sum','group_clip'), required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=32)
    args = parser.parse_args()
    stop = args.stop_after or args.steps
    if min(args.steps, args.context, args.eval_every) < 1 or args.context % 16 or not 1 <= stop <= args.steps:
        parser.error('Positive budgets; context must divide into 16-byte segments.')
    if args.evaluate_only and not args.resume:
        parser.error('Evaluation-only requires resume.')
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        parser.error('Fresh runs require an empty output directory.')
    torch.set_num_threads(1)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    config = GraphConfig.from_dict(parent['config'])
    data_config = TaskStreamConfig(entities='ABCD', queries_per_cycle=6, delays=(4,48,96),
        interference=True, acquisition_queries=True)
    protocol = dict(steps=args.steps, context=args.context, batch=4, lr=.001, seed=83, data_seed=383,
        documents=1024, eval_every=args.eval_every, eval_count=8, eval_chunks=8, truncate_every=16,
        syntax_weight=.1, delayed_credit=True, gradient_mode=args.mode, gradient_cap=1.,
        natural_context=True, parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest())
    run = ContextBalancedRun(args.data.read_bytes(), config, data_config, protocol, args.parent)
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
            data_config=asdict(data_config), parameters=sum(p.numel() for p in run.parameters),
            raw_sha256=run.raw_digest, initialization=run.initialization,
            group_parameters={k:sum(run.parameters[i].numel() for i in ids) for k,ids in run.groups.items()},
            selection='Mean validation language-byte and answer NLL; ties retain earlier checkpoint',
            schedule='One language and one memory window at identical weights; one optimizer step'), indent=2)+'\n')
    if args.evaluate_only:
        final_evaluation(run, args.out); return
    if stop <= run.step:
        parser.error('Stop must follow saved update; use --evaluate-only for a complete run.')
    start = time.perf_counter()
    while run.step < stop:
        record = run.advance()
        if run.step % args.eval_every == 0 or run.step == args.steps:
            validation = run.validation()
            if validation['score'] < run.best:
                run.best = validation['score']; run.save(args.out/'best.pt')
            record['validation'] = validation
        record['seconds'] = time.perf_counter()-start
        # Every completed pair is recoverable. Mid-pair interruptions restart
        # from the preceding completed optimizer step, with no selection bonus.
        run.save(args.out/'last.pt')
        with (args.out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        if 'validation' in record or run.step % 8 == 0:
            print(json.dumps(dict(step=run.step, seconds=record['seconds'],
                **({'validation': {k:validation[k]['nll' if k == 'language' else 'answer_nll']
                                  for k in ('language','memory')}} if 'validation' in record else {}))), flush=True)
    if run.step == args.steps:
        final_evaluation(run, args.out)


if __name__ == '__main__':
    main()
