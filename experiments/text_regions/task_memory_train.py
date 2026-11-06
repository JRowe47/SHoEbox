"""Paired, resumable pilot for task-driven publications and detached credit."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .task_stream import TaskStreamConfig, TaskByteStreams, make_documents
from .task_memory import TaskMemoryConfig, TaskMemoryModel, TaskMemoryState
from .task_credit import WriteEligibility


def byte_loss(logits, targets):
    mask = targets.ne(-100)
    return F.cross_entropy(logits[mask], targets[mask]) if mask.any() else logits.sum()*0


def objective(result, chunk, text_weight=.1, write_cost=0.):
    ordinary = chunk.ordinary_targets.masked_fill(chunk.answer_mask, -100)
    answer = byte_loss(result['logits'], chunk.answer_targets)
    text = byte_loss(result['logits'], ordinary)
    return answer + text_weight*text + .001*result['aux'] + write_cost*result['write_probs'].mean(), answer, text


def packed_documents(documents):
    length = max(len(d.data) for d in documents)
    x = torch.full((len(documents), length), ord(' '), dtype=torch.long)
    y = torch.full_like(x, -100)
    answer = torch.full_like(x, -100)
    valid = torch.zeros_like(x, dtype=torch.bool)
    queries = {}
    for row, d in enumerate(documents):
        n = len(d.data)
        x[row, :n] = torch.tensor(list(d.data))
        y[row, :n-1] = x[row, 1:n]
        answer[row, :n] = torch.tensor(d.answer_targets)
        valid[row, :n] = True
        queries.update({(row, q.position): q for q in d.queries})
    y[answer.ne(-100)] = -100
    return x, y, answer, valid, queries


@torch.no_grad()
def evaluate(model, data_config, seed, count=16, context=16, split='validation', interventions=False, control='intact'):
    """Fresh streams; counterfactuals fork the identical live pre-event state."""
    was_training = model.training
    model.eval()
    documents = make_documents(data_config, split=split, seed=seed, count=count)
    x, y, answer, valid, queries = packed_documents(documents)
    state = None
    nats = text_nats = correct = n = nt = executions = 0
    write_sum = read_sum = 0.
    owner_writes = torch.zeros(model.cfg.owners, dtype=torch.float64)
    owner_valid = torch.zeros_like(owner_writes)
    owner_reads = torch.zeros_like(owner_writes)
    weighted_age_sum = query_age_sum = 0.
    weighted_age_count = query_age_count = 0
    groups, records, controls = {}, [], {}
    started = time.perf_counter()
    stale_publications = None
    # One-byte evaluation makes the fork exactly the same pre-answer state.
    for position in range(x.shape[1]):
        prestate = state
        if prestate is None:
            publication_valid = torch.zeros(len(documents), model.cfg.owners, dtype=torch.bool)
            publication_occurrences = torch.zeros_like(publication_valid, dtype=torch.long)
            publication_ages = torch.zeros_like(publication_occurrences)
        else:
            publication_valid = torch.stack([p.valid for p in prestate.publications.values()], 1)
            publication_occurrences = torch.stack([p.occurrence for p in prestate.publications.values()], 1)
            publication_ages = prestate.events_seen[:, None]-publication_occurrences
        # The shuffle control reads the preceding stream row's publications.
        if control == 'shuffle':
            publication_valid = publication_valid.roll(1, 0)
            publication_occurrences = publication_occurrences.roll(1, 0)
            publication_ages = publication_ages.roll(1, 0)
        result = model(x[:, position:position+1], state=prestate, control=control)
        state = result['state']
        read_weights = result['read_weights'][:, 0, :-1]*publication_valid
        nonnull_mass = read_weights.sum(1)
        weighted_ages = (read_weights*publication_ages).sum(1)/nonnull_mass.clamp_min(1e-12)
        logits = result['logits'][:, 0]
        mask = answer[:, position].ne(-100)
        ordinary = y[:, position].ne(-100)
        if ordinary.any():
            text_nats += float(F.cross_entropy(logits[ordinary], y[ordinary, position], reduction='sum'))
            nt += int(ordinary.sum())
        rows = mask.nonzero().flatten().tolist()
        counterfactuals = {}
        if rows and interventions and control == 'intact':
            for intervention in ('blocked', 'shuffle', 'stale'):
                alternate = prestate
                if intervention == 'stale':
                    if stale_publications is None:
                        stale_publications = prestate.publications
                    alternate = replace(prestate, publications=stale_publications)
                cf = model(x[:, position:position+1], state=alternate,
                           control='intact' if intervention == 'stale' else intervention)
                assert cf['expert_executions'] == result['expert_executions']
                counterfactuals[intervention] = cf['logits'][:, 0]
        for row in rows:
            target = answer[row, position]
            loss = float(F.cross_entropy(logits[row:row+1], target[None]))
            hit = int(logits[row].argmax() == target)
            nats += loss; correct += hit; n += 1
            q = queries[row, position]
            labels = [q.kind, 'corrected' if q.correction else 'initial',
                      'beyond_chunk' if q.delay > context else 'within_chunk']
            for label in labels:
                g = groups.setdefault(label, dict(count=0, correct=0, nats=0.))
                g['count'] += 1; g['correct'] += hit; g['nats'] += loss
            record = dict(**asdict(q), document=row, prediction=int(logits[row].argmax()),
                          nll=loss, read_weights=result['read_weights'][row, 0].tolist(),
                          read_versions=result['read_versions'][(row-1) % len(documents) if control == 'shuffle' else row, 0].tolist(),
                          read_source_owners=list(model.owner_names),
                          read_source_document=(row-1) % len(documents) if control == 'shuffle' else row,
                          read_valid=publication_valid[row].tolist(),
                          read_occurrences=[int(v) if bool(ok) else None for v, ok in
                                            zip(publication_occurrences[row], publication_valid[row])],
                          read_ages=[int(v) if bool(ok) else None for v, ok in
                                     zip(publication_ages[row], publication_valid[row])],
                          weighted_publication_age=float(weighted_ages[row]) if nonnull_mass[row] > 0 else None,
                          effective_weighted_publication_age=float(weighted_ages[row]) if
                              control != 'blocked' and nonnull_mass[row] > 0 else None)
            if nonnull_mass[row] > 0:
                query_age_sum += float(weighted_ages[row]); query_age_count += 1
            for intervention, values in counterfactuals.items():
                g = controls.setdefault(intervention, dict(count=0, correct=0, nats=0.))
                closs = float(F.cross_entropy(values[row:row+1], target[None]))
                g['count'] += 1; g['correct'] += int(values[row].argmax() == target); g['nats'] += closs
                record[intervention+'_nll'] = closs
                record[intervention+'_prediction'] = int(values[row].argmax())
            records.append(record)
        executions += result['expert_executions']
        active = valid[:, position]
        write_sum += float(result['write_actions'][:, 0][active].float().sum())
        read_sum += float((1-result['read_weights'][:, 0, -1])[active].sum())
        owner_writes += result['write_actions'][:, 0][active].sum(0).double()
        owner_valid += publication_valid[active].sum(0).double()
        owner_reads += read_weights[active].sum(0).double()
        has_age = active & nonnull_mass.gt(0)
        weighted_age_sum += float(weighted_ages[has_age].sum())
        weighted_age_count += int(has_age.sum())
    def rates(g):
        return dict(count=g['count'], accuracy=g['correct']/max(g['count'], 1),
                    nll=g['nats']/max(g['count'], 1))
    summary = dict(split=split, control=control, answers=n, accuracy=correct/max(n, 1), answer_nll=nats/max(n, 1),
                   ordinary_nll=text_nats/max(nt, 1), groups={k: rates(v) for k, v in groups.items()},
                   interventions={k: rates(v) for k, v in controls.items()},
                   write_rate=write_sum/(int(valid.sum())*model.cfg.owners),
                   read_mass=read_sum/int(valid.sum()), effective_read_mass=0. if control == 'blocked' else read_sum/int(valid.sum()),
                   owners={name: dict(write_fraction=float(owner_writes[i])/int(valid.sum()),
                                       valid_fraction=float(owner_valid[i])/int(valid.sum()),
                                       read_mass=float(owner_reads[i])/int(valid.sum()))
                           for i, name in enumerate(model.owner_names)},
                   mean_weighted_publication_age=weighted_age_sum/weighted_age_count if weighted_age_count else None,
                   query_mean_weighted_publication_age=query_age_sum/query_age_count if query_age_count else None,
                   publication_age_definition='Pre-event byte clock minus committed occurrence; valid nonnull read weights normalized per event. Valid fraction is pre-read occupancy.',
                   state_bytes=state.storage_bytes(),
                   expert_executions=executions, external_bytes=int(valid.sum()),
                   executed_bytes=x.numel(), seconds=time.perf_counter()-started)
    model.train(was_training)
    return summary, records


def checkpoint(path, model, optimizer, state, credit, streams, args, data_cfg, step, best, totals=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    serialized_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(dict(schema=1, config=model.cfg.to_dict(), data_config=data_cfg.to_dict(),
        weights=model.state_dict(), optimizer=optimizer.state_dict(),
        state=None if state is None else state.to_dict(), credit=credit.to_dict(),
        stream=streams.state_dict(), rng=torch.get_rng_state(), args=serialized_args,
        step=step, best=best, totals=totals or {}), temporary)
    temporary.replace(path)


def validate_resume(saved, args, model, data_cfg):
    if saved['config'] != model.cfg.to_dict() or saved['data_config'] != data_cfg.to_dict():
        raise ValueError('Resume configuration mismatch.')
    for name in ('seed', 'data_seed', 'arm', 'batch', 'context', 'width', 'owners',
                 'lr', 'text_weight', 'write_cost', 'threads', 'eval_count'):
        if saved['args'].get(name, 0. if name == 'write_cost' else None) != getattr(args, name, 0. if name == 'write_cost' else None):
            raise ValueError(f'Resume {name} mismatch.')
    if args.steps < saved['step']:
        raise ValueError('Requested budget precedes the saved training step.')


def final_evaluation(out, model, data_cfg, data_seed, count, context, control):
    """Report selected checkpoint and final-budget results on paired fresh data."""
    summaries = {}
    for checkpoint_name, filename_prefix in (('best', ''), ('last', 'final_')):
        saved = torch.load(out/(checkpoint_name+'.pt'), map_location='cpu', weights_only=False)
        model.load_state_dict(saved['weights'])
        evaluated = {}
        for split in ('test', 'long_delay'):
            summary, records = evaluate(model, data_cfg, data_seed, count, context, split,
                                         interventions=True, control=control)
            if control == 'intact':
                blocked, _ = evaluate(model, data_cfg, data_seed, count, context, split,
                                      interventions=False, control='blocked')
                summary['full_blocked_rollout'] = blocked
            evaluated[split] = summary
            (out/(filename_prefix+split+'_traces.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        if checkpoint_name == 'best':
            summaries.update(selected_step=saved['step'], **evaluated)
        else:
            summaries['final_budget'] = dict(step=saved['step'], **evaluated)
    (out/'evaluation.json').write_text(json.dumps(summaries, indent=2)+'\n')
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--data-seed', type=int, default=123)
    p.add_argument('--arm', choices=('integrated', 'no_credit', 'no_memory'), default='integrated')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--context', type=int, default=16)
    p.add_argument('--width', type=int, default=80)
    p.add_argument('--owners', type=int, default=4)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--text-weight', type=float, default=.1)
    p.add_argument('--write-cost', type=float, default=0., help='Optional mean write-probability penalty; report separately.')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-count', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', type=Path)
    p.add_argument('--final-eval', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.batch, args.context, args.eval_every, args.eval_count, args.threads) < 1:
        p.error('Positive step, batch, context, evaluation and thread counts required.')
    if not 0 <= args.write_cost <= 1:
        p.error('Write cost must be between zero and one.')
    if not args.resume and any((args.out/name).exists() for name in ('best.pt', 'last.pt', 'history.jsonl')):
        p.error('Output already contains a run; use --resume or a fresh --out directory.')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    model = TaskMemoryModel(TaskMemoryConfig(width=args.width, owners=args.owners))
    data_cfg = TaskStreamConfig()
    documents = make_documents(data_cfg, seed=args.data_seed, count=4096)
    streams = TaskByteStreams(documents, args.batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    state = model.initial_state(args.batch)
    credit = WriteEligibility.zeros(args.batch, model.owner_names,
                                    model.memory_dim, model.cfg.policy_features)
    start = 0; best = float('inf')
    external = experts = 0; delayed_norm = prior_seconds = 0.
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume(saved, args, model, data_cfg)
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        state = TaskMemoryState.from_dict(saved['state']); credit = WriteEligibility.from_dict(saved['credit'])
        streams.load_state_dict(saved['stream']); torch.set_rng_state(saved['rng'])
        start, best = saved['step'], saved['best']
        totals = saved.get('totals', {})
        external = totals.get('external_bytes', start*args.batch*args.context)
        experts = totals.get('expert_executions', external*(args.owners+1)*3*
                             model.cfg.active_groups*model.cfg.experts_per_group)
        delayed_norm = totals.get('delayed_gradient_norm_sum', 0.)
        prior_seconds = totals.get('elapsed_seconds', 0.)
        if not (args.out/'best.pt').exists():
            previous_best = args.resume.parent/'best.pt'
            if previous_best.exists():
                shutil.copy2(previous_best, args.out/'best.pt')
            else:
                raise FileNotFoundError('Resume requires the accompanying best.pt checkpoint.')
        if not (args.out/'last.pt').exists():
            shutil.copy2(args.resume, args.out/'last.pt')
    invocation = {**vars(args), 'out': str(args.out), 'resume': str(args.resume) if args.resume else None,
                  'data_digest': streams.fingerprint, 'model_parameters': sum(p.numel() for p in model.parameters()),
                  'config': model.cfg.to_dict(), 'data_config': data_cfg.to_dict()}
    (args.out/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    started = time.perf_counter()
    model.train()
    for step in range(start+1, args.steps+1):
        chunk = streams.next_chunk(args.context)
        optimizer.zero_grad(set_to_none=True)
        boundary = credit.capture(state.memory_tensor())
        state = state.with_memory_tensor(boundary.memory)
        result = model(chunk.values, state=state, reset=chunk.reset,
                       control='blocked' if args.arm == 'no_memory' else 'intact')
        loss, answer, text = objective(result, chunk, args.text_weight, args.write_cost)
        loss.backward()
        if args.arm == 'integrated':
            delayed_norm += boundary.apply(model.write_policy)
        credit = credit.advance_chunk(result)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        state = result['state'].detach()
        external += result['external_events']; experts += result['expert_executions']
        if step % args.eval_every == 0 or step == args.steps:
            evaluation, _ = evaluate(model, data_cfg, args.data_seed, args.eval_count, args.context,
                                     control='blocked' if args.arm == 'no_memory' else 'intact')
            elapsed = prior_seconds+time.perf_counter()-started
            totals = dict(external_bytes=external, expert_executions=experts,
                          elapsed_seconds=elapsed, delayed_gradient_norm_sum=delayed_norm)
            record = dict(step=step, loss=float(loss.detach()), train_answer_nll=float(answer.detach()),
                          train_text_nll=float(text.detach()), validation=evaluation,
                          training_external_bytes=external, training_expert_executions=experts,
                          elapsed_seconds=elapsed, run_seconds=time.perf_counter()-started,
                          delayed_gradient_norm_sum=delayed_norm,
                          trace_bytes=credit.storage_bytes())
            with (args.out/'history.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if evaluation['answer_nll'] < best:
                best = evaluation['answer_nll']
                checkpoint(args.out/'best.pt', model, optimizer, state, credit, streams, args, data_cfg, step, best, totals)
            checkpoint(args.out/'last.pt', model, optimizer, state, credit, streams, args, data_cfg, step, best, totals)
            print(json.dumps({k: record[k] for k in ('step', 'train_answer_nll', 'elapsed_seconds')} |
                             dict(val_nll=evaluation['answer_nll'], val_accuracy=evaluation['accuracy'])), flush=True)
    if args.final_eval:
        final_evaluation(args.out, model, data_cfg, args.data_seed, args.eval_count*2, args.context,
                         'blocked' if args.arm == 'no_memory' else 'intact')


if __name__ == '__main__':
    main()
