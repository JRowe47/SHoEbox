"""Unified regional graph learning from natural bytes and memory-task bytes."""
import argparse
import hashlib
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time
import torch
from torch.nn import functional as F
from .config import GraphConfig
from .model import RegionLanguageModel
from .state import GraphState
from .graph_credit import make_graph_eligibility, graph_window_forward
from .mixed_recall_train import MixedRun
from .data import OrderedByteStreams
from .task_memory_train import packed_documents
from .task_stream import TaskStreamConfig, make_documents


@contextmanager
def intervention(model, control):
    original = model.cfg
    changes, kwargs = {}, {}
    if control == 'pooled':
        kwargs['recall_control'] = 'pooled'
    elif control == 'no_feedback':
        changes.update(recall_feedback=False, publication_feedback=False)
    elif control == 'no_demand':
        changes['publication_feedback'] = False
    elif control == 'always_state':
        changes['publication_state_commit'] = 'always'
    elif control == 'always_write':
        kwargs['publication_overrides'] = {n: 'always' for n in original.task_publication}
    elif control == 'blocked':
        kwargs['recall_control'] = 'blocked'
    elif control != 'intact':
        raise ValueError('Unknown graph intervention.')
    try:
        model.cfg = model.connections.cfg = replace(original, **changes)
        yield kwargs
    finally:
        model.cfg = model.connections.cfg = original


@torch.no_grad()
def evaluate_language(model, raw, batch=4, context=32, chunks=4, control='intact'):
    streams = OrderedByteStreams.from_bytes(raw, batch)
    state, nats, count = None, 0., 0
    with intervention(model, control) as kwargs:
        for _ in range(chunks):
            chunk = streams.next_chunk(context)
            result = model(chunk.values, state=state, reset=chunk.reset, **kwargs)
            state = result['state']
            nats += float(F.cross_entropy(result['logits'].flatten(0, 1), chunk.targets.flatten(), reduction='sum'))
            count += chunk.targets.numel()
    return dict(nll=nats/count, targets=count, state_bytes=state.storage_bytes())


@torch.no_grad()
def evaluate_memory(model, config, seed, count=4, context=32, split='validation', control='intact',
                    *, documents=None):
    if documents is None:
        documents = make_documents(config, split=split, seed=seed, count=count)
    x, y, answer, valid, queries = packed_documents(documents)
    owners = model.cfg.task_publication
    writes = torch.zeros(len(owners), dtype=torch.float64)
    demands = torch.zeros(len(owners), 2, dtype=torch.float64)
    opportunities = 0
    nats = text_nats = 0.
    correct = targets = ordinary_targets = executions = 0
    state, groups, traces = None, {}, []
    with intervention(model, control) as kwargs:
        for start in range(0, x.shape[1], context):
            mask = valid[:, start:start+context]
            result = model(x[:, start:start+context], state=state,
                availability={v: mask for v in model.cfg.views},
                node_masks={n: mask for n in model.nodes}, return_write_activity=True,
                return_recall_activity=True, **kwargs)
            state = result['state']
            activity = result['write_activity']
            writes += activity['write_actions'].sum((0, 1)).double()
            active_rounds = mask.repeat_interleave(model.cfg.rounds, 1)
            demands += (activity['demand'] * active_rounds[..., None, None]).sum((0, 1)).double()
            opportunities += int(active_rounds.sum())
            executions += result['expert_executions']
            ordinary = y[:, start:start+context]
            known = ordinary.ne(-100)
            text_nats += float(F.cross_entropy(result['logits'][known], ordinary[known], reduction='sum'))
            ordinary_targets += int(known.sum())
            for row, offset in answer[:, start:start+context].ne(-100).nonzero().tolist():
                position = start + offset
                q = queries[row, position]
                logits = result['logits'][row, offset]
                loss = float(F.cross_entropy(logits[None], answer[row, position:position+1]))
                prediction = int(logits.argmax())
                hit = int(prediction == q.answer)
                nats += loss; correct += hit; targets += 1
                for label in (q.kind, 'corrected' if q.correction else 'initial',
                              'beyond_chunk' if q.delay > context else 'within_chunk'):
                    g = groups.setdefault(label, dict(count=0, correct=0, nats=0.))
                    g['count'] += 1; g['correct'] += hit; g['nats'] += loss
                reads = {}
                for name, records in result['recall_activity'].items():
                    reads[name] = dict(edges=result['recall_edges'][name],
                        weights=records['weights'][row, offset].tolist(),
                        valid=records['valid'][row, offset].tolist(),
                        versions=records['versions'][row, offset].tolist(),
                        occurrences=records['occurrences'][row, offset].tolist(),
                        rounds=records['round_indices'][row, offset].tolist())
                traces.append(dict(**asdict(q), document=row, prediction=prediction, nll=loss, reads=reads))
    return dict(split=split, control=control, answers=targets, accuracy=correct/max(targets, 1),
        answer_nll=nats/max(targets, 1), ordinary_nll=text_nats/max(ordinary_targets, 1),
        groups={k: dict(count=g['count'], accuracy=g['correct']/g['count'], nll=g['nats']/g['count'])
                for k, g in groups.items()},
        owners={n: dict(write_fraction=float(writes[i])/max(opportunities, 1),
            mean_demand=(demands[i]/max(opportunities, 1)).tolist()) for i, n in enumerate(owners)},
        external_bytes=int(valid.sum()), executed_bytes=x.numel(), expert_executions=executions,
        state_bytes=state.storage_bytes()), traces


class GraphMixedRun(MixedRun):
    kind = 'graph_mixed'
    model_type = RegionLanguageModel
    state_type = GraphState
    credit_factory = staticmethod(make_graph_eligibility)
    forward_window = staticmethod(graph_window_forward)

    def initialize(self, parent):
        if parent is None:
            return None
        saved = torch.load(parent, map_location='cpu', weights_only=True)
        weights = saved.get('weights', saved.get('model'))
        if weights is None:
            raise ValueError('Parent checkpoint has no model weights.')
        own = self.model.state_dict()
        if set(weights) - set(own) or any(own[n].shape != value.shape for n, value in weights.items()):
            raise ValueError('Parent graph weights are incompatible.')
        missing = set(own) - set(weights)
        if any(not name.startswith('publisher.') for name in missing):
            raise ValueError('Only the new publication interface may be missing from the parent.')
        self.model.load_state_dict(weights, strict=False)
        return dict(parent=str(parent), sha256=hashlib.sha256(Path(parent).read_bytes()).hexdigest(),
                    transferred=len(weights), new_parameters=sorted(missing))

    def apply_credit(self, boundaries):
        enabled = self.protocol['delayed_credit']
        publisher = self.model.publisher
        delayed = sum(b.apply(publisher.policy, enabled) for b in boundaries)
        payload = sum(b.apply_payload(publisher.payload_adapter, enabled) for b in boundaries
                      ) if publisher.payload_adapter is not None else 0.
        return delayed, payload

    def language_metrics(self, raw, chunks=None, control='intact'):
        p = self.protocol
        return evaluate_language(self.model, raw, p['batch'], p['context'], chunks or p['eval_chunks'], control)

    def memory_metrics(self, split='validation', count=None, control='intact'):
        p = self.protocol
        return evaluate_memory(self.model, self.data_config, p['data_seed'], count or p['eval_count'],
                               p['context'], split, control)

    def restore(self, path):
        super().restore(path)
        for source in self.states:
            self.credits[source].capture(self.states[source])


def final_evaluation(run, out):
    """Cache each finished measurement; identical checkpoint weights reuse it."""
    out = Path(out)
    cache_path = out/'evaluation_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def persist():
        temporary = cache_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(cache,indent=2)+'\n')
        temporary.replace(cache_path)

    results = {}
    for tag in ('best', 'last'):
        saved = torch.load(out/(tag+'.pt'), map_location='cpu', weights_only=True)
        run.model.load_state_dict(saved['weights'])
        digest = hashlib.sha256(json.dumps(dict(schema=1,config=run.model.cfg.to_dict(),
            data=run.data_config.to_dict(),protocol=run.protocol,corpus=run.raw_digest),sort_keys=True).encode())
        for name,value in sorted(saved['weights'].items()):
            digest.update(repr((name,str(value.dtype),tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        signature = digest.hexdigest()
        row = cache.setdefault(signature,dict(memory={},interventions={}))
        if 'language' not in row:
            row['language'] = run.language_metrics(run.test_raw,run.protocol['eval_chunks']*2)
            persist()
        language = row['language']
        for split in ('test', 'long_delay'):
            trace_path = out/f'cache_{signature}_{split}_traces.jsonl'
            if split not in row['memory'] or not trace_path.exists():
                row['memory'][split],traces = run.memory_metrics(split,run.protocol['eval_count']*2)
                trace_path.write_text(''.join(json.dumps(t)+'\n' for t in traces))
                persist()
            shutil.copyfile(trace_path,out/f'{tag}_{split}_traces.jsonl')
        memory = row['memory']
        controls = row['interventions']
        for control in ('blocked', 'no_demand', 'always_state', 'always_write', 'pooled', 'no_feedback'):
            if control in controls:
                continue
            natural = run.language_metrics(run.test_raw, run.protocol['eval_chunks']*2, control)
            task, _ = run.memory_metrics('test', run.protocol['eval_count']*2, control)
            controls[control] = dict(language_nll=natural['nll'], answer_nll=task['answer_nll'],
                language_delta=natural['nll']-language['nll'],
                answer_delta=task['answer_nll']-memory['test']['answer_nll'])
            persist()
        publisher = run.model.publisher
        results[tag] = dict(step=saved['step'], signature=signature, language=language, memory=memory,
            interventions=controls, totals=saved['totals'],
            learned=dict(demand_feature_norm=float(publisher.demand_features.weight.detach().norm()),
                         memory_feature_norm=float(publisher.memory_features.weight.detach().norm()),
                         payload_coefficient_norm=float(publisher.payload_adapter.coefficients.detach().norm())
                             if publisher.payload_adapter is not None else 0.),
            credit_bytes_per_source={k: v.storage_bytes() for k, v in run.credits.items()})
    (out/'evaluation.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--init-from', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--recall-slots', type=int, default=7)
    parser.add_argument('--truncate-every', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=83)
    parser.add_argument('--data-seed', type=int, default=283)
    parser.add_argument('--documents', type=int, default=1024)
    parser.add_argument('--eval-every', type=int, default=40)
    parser.add_argument('--eval-count', type=int, default=4)
    parser.add_argument('--eval-chunks', type=int, default=4)
    parser.add_argument('--no-delayed-credit', action='store_true')
    parser.add_argument('--final-eval', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.context, args.batch, args.rounds, args.recall_slots, args.documents,
           args.eval_every, args.eval_count, args.eval_chunks) < 1 or args.lr <= 0:
        parser.error('Positive budgets and learning rate required.')
    stop = args.stop_after or args.steps
    if not 1 <= stop <= args.steps or args.documents < args.batch:
        parser.error('Invalid stopping step or document count.')
    if args.truncate_every < 0 or (args.truncate_every and args.context % args.truncate_every):
        parser.error('Truncation must divide context.')
    if args.resume and args.init_from or args.evaluate_only and not args.resume:
        parser.error('Resume and initialization are exclusive; evaluation-only requires resume.')
    if args.resume and args.resume.resolve() != (args.out/'last.pt').resolve():
        parser.error('Resume last.pt in its existing output directory.')
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        parser.error('A fresh run requires an empty output directory.')
    torch.set_num_threads(1)
    config = GraphConfig.tiny(task_publication=('A','B','C','U','V'),
        publication_payload_rank=4, publication_state_commit='publication',
        recall_slots=args.recall_slots, rounds=args.rounds, recurrent_feedback=True, seed=args.seed)
    data_config = TaskStreamConfig(entities='ABCD', queries_per_cycle=4, delays=(2,48), interference=True)
    protocol = {name: getattr(args, name) for name in ('steps','context','batch','lr','seed','data_seed',
        'documents','eval_every','eval_count','eval_chunks','truncate_every')}
    protocol.update(syntax_weight=.1, threads=1, delayed_credit=not args.no_delayed_credit)
    run = GraphMixedRun(args.data.read_bytes(), config, data_config, protocol, args.init_from)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run.restore(args.resume)
        if not (args.out/'best.pt').exists():
            parser.error('Resume requires best.pt.')
        if args.evaluate_only:
            final_evaluation(run, args.out)
            return
        if stop <= run.step:
            parser.error('Stopping step must follow the saved step.')
    else:
        initial = run.validation()
        run.best = initial['score']
        run.save(args.out/'best.pt'); run.save(args.out/'last.pt')
        (args.out/'initial.json').write_text(json.dumps(initial, indent=2)+'\n')
        (args.out/'protocol.json').write_text(json.dumps(dict(config=config.to_dict(),
            data_config=data_config.to_dict(), protocol=protocol, raw_sha256=run.raw_digest,
            parameters=sum(p.numel() for p in run.model.parameters()), initialization=run.initialization,
            selection='mean validation natural-byte and memory-answer NLL',
            schedule='odd language, even memory; separate states and credit; all weights trainable'), indent=2)+'\n')
    started = time.perf_counter()
    while run.step < stop:
        record = run.advance()
        if run.step % args.eval_every == 0 or run.step == args.steps:
            validation = run.validation()
            if validation['score'] < run.best:
                run.best = validation['score']; run.save(args.out/'best.pt')
            run.save(args.out/'last.pt')
            record.update(validation=validation, seconds=time.perf_counter()-started)
            with (args.out/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print(json.dumps(dict(step=run.step, language_nll=validation['language']['nll'],
                answer_nll=validation['memory']['answer_nll'], seconds=record['seconds'])), flush=True)
        elif run.step == stop:
            run.save(args.out/'last.pt')
    if args.final_eval:
        final_evaluation(run, args.out)


if __name__ == '__main__':
    main()
