"""Paired, resumable pilot for task-driven publications and detached credit."""
import argparse
import hashlib
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .task_stream import TaskStreamConfig, TaskByteStreams, make_documents
from .task_memory import TaskMemoryConfig, TaskMemoryModel, TaskMemoryState
from .task_payload import make_eligibility
from .task_curriculum import stage_for_step, stage_config, window_forward, SourceCoverage


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
    write_sum = read_sum = old_read_sum = query_old_read_sum = 0.
    bootstrap_events = learned_writes = 0
    owner_writes = torch.zeros(model.cfg.owners, dtype=torch.float64)
    owner_valid = torch.zeros_like(owner_writes)
    owner_reads = torch.zeros_like(owner_writes)
    weighted_age_sum = query_age_sum = 0.
    weighted_age_count = query_age_count = 0
    recall_selected = recall_unique = recall_events = 0
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
        if control == 'fresh':
            publication_valid = result['read_valid'][:, 0]
            publication_occurrences = torch.full_like(publication_occurrences, position)
            publication_ages = torch.zeros_like(publication_ages)
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
            if q.cycle == -1:
                labels.append('acquisition')
            for label in labels:
                g = groups.setdefault(label, dict(count=0, correct=0, nats=0.))
                g['count'] += 1; g['correct'] += hit; g['nats'] += loss
            record = dict(**asdict(q), document=row, prediction=int(logits[row].argmax()),
                          nll=loss, read_source=result.get('read_source', 'publication'), read_weights=result['read_weights'][row, 0].tolist(),
                          read_versions=result['read_versions'][row, 0].tolist(),
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
            record['old_read_mass'] = float((read_weights[row]*publication_ages[row].gt(context)).sum())
            if model.cfg.recall_slots:
                record['recall_weights'] = result['recall_weights'][row, 0].tolist()
                record['recall_probabilities'] = result['recall_probabilities'][row, 0].tolist()
            query_old_read_sum += record['old_read_mass']
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
        if model.cfg.recall_slots:
            slot_weights = result['recall_weights'][:, 0, :, :-1][active]
            recall_selected += float(slot_weights.sum())
            recall_unique += float(slot_weights.sum(1).clamp(max=1).sum())
            recall_events += int(active.sum())
        bootstrap = result.get('bootstrap_mask', torch.zeros_like(x[:, position:position+1], dtype=torch.bool))[:, 0]
        bootstrap_events += int((bootstrap & active).sum())
        learned_writes += int((result['write_actions'][:, 0] & ~bootstrap[:, None])[active].sum())
        old_read_sum += float((read_weights*publication_ages.gt(context))[active].sum())
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
    summary = dict(split=split, control=control, read_source='candidate' if control == 'fresh' else 'publication', answers=n, accuracy=correct/max(n, 1), answer_nll=nats/max(n, 1),
                   ordinary_nll=text_nats/max(nt, 1), groups={k: rates(v) for k, v in groups.items()},
                   interventions={k: rates(v) for k, v in controls.items()},
                   bootstrap_events=bootstrap_events, learned_writes=learned_writes,
                   learned_write_rate=learned_writes/(max(int(valid.sum())-bootstrap_events, 1)*model.cfg.owners),
                   old_read_mass=old_read_sum/int(valid.sum()), query_old_read_mass=query_old_read_sum/max(n, 1),
                   effective_old_read_mass=0. if control == 'blocked' else old_read_sum/int(valid.sum()),
                   effective_query_old_read_mass=0. if control == 'blocked' else query_old_read_sum/max(n, 1),
                   write_rate=write_sum/(int(valid.sum())*model.cfg.owners),
                   read_mass=read_sum/int(valid.sum()), effective_read_mass=0. if control == 'blocked' else read_sum/int(valid.sum()),
                   owners={name: dict(write_fraction=float(owner_writes[i])/int(valid.sum()),
                                       valid_fraction=float(owner_valid[i])/int(valid.sum()),
                                       read_mass=float(owner_reads[i])/int(valid.sum()))
                           for i, name in enumerate(model.owner_names)},
                   mean_weighted_publication_age=weighted_age_sum/weighted_age_count if weighted_age_count else None,
                   query_mean_weighted_publication_age=query_age_sum/query_age_count if query_age_count else None,
                   publication_age_definition='Fresh candidates have age 0 and version -1. Pre-event byte clock minus committed occurrence; valid nonnull read weights normalized per event. Valid fraction is pre-read occupancy.',
                   state_bytes=state.storage_bytes(),
                   expert_executions=executions, external_bytes=int(valid.sum()),
                   executed_bytes=x.numel(), seconds=time.perf_counter()-started)
    if model.cfg.recall_slots:
        summary['recall'] = dict(slots=model.cfg.recall_slots, layout=model.cfg.recall_layout,
            hard=model.cfg.recall_hard, feedback=model.cfg.recall_feedback,
            mean_active_slots=recall_selected/max(recall_events, 1),
            mean_unique_owners=recall_unique/max(recall_events, 1),
            effective_active_slots=0. if control == 'blocked' else recall_selected/max(recall_events, 1),
            coverage_strength=F.softplus(model.recall.coverage_logit).tolist(),
            feedback_weight_norms={name: float(getattr(model.recall, name).weight.norm())
                for name in ('query_feedback', 'null_feedback', 'usage_feedback')},
            aggregate_read_weights='Mean selection mass per slot; full per-slot choices are in query records.')
    model.train(was_training)
    return summary, records


def checkpoint(path, model, optimizer, state, credit, streams, args, data_cfg, step, best, totals=None, initialization=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    serialized_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(dict(schema=1, config=model.cfg.to_dict(), data_config=data_cfg.to_dict(),
        weights=model.state_dict(), optimizer=optimizer.state_dict(),
        state=None if state is None else state.to_dict(), credit=credit.to_dict(),
        stream=streams.state_dict(), rng=torch.get_rng_state(), args=serialized_args,
        step=step, best=best, totals=totals or {}, initialization=initialization), temporary)
    temporary.replace(path)


def validate_resume(saved, args, model, data_cfg):
    if TaskMemoryConfig.from_dict(saved['config']).to_dict() != model.cfg.to_dict() or TaskStreamConfig.from_dict(saved['data_config']).to_dict() != data_cfg.to_dict():
        raise ValueError('Resume configuration mismatch.')
    for name in ('seed', 'data_seed', 'arm', 'batch', 'context', 'width', 'owners',
                 'lr', 'text_weight', 'write_cost', 'threads', 'eval_count'):
        if saved['args'].get(name, 0. if name == 'write_cost' else None) != getattr(args, name, 0. if name == 'write_cost' else None):
            raise ValueError(f'Resume {name} mismatch.')
    for name, default in (('write_mode', 'independent'), ('memory_aware_write', False), ('bootstrap_empty', False), ('stream_preset', 'basic')):
        if saved['args'].get(name, default) != getattr(args, name, default):
            raise ValueError(f'Resume {name} mismatch.')
    for name, default in (('curriculum', 'none'), ('acquire_steps', 200),
                          ('bridge_steps', 200), ('truncate_every', 0),
                          ('payload_rank', 0), ('payload_credit', False), ('state_commit', 'always'),
                          ('recall_slots', 0), ('recall_layout', 'distinct'),
                          ('recall_feedback', True), ('recall_hard', True)):
        if saved['args'].get(name, default) != getattr(args, name, default):
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


def initialize_weights(model, path):
    """Start a new independent study from compatible learned weights only."""
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    # Optional identity/zero initialized additions may be new.
    missing, unexpected = model.load_state_dict(saved['weights'], strict=False)
    allowed = {'memory_policy_features.weight'} if model.cfg.memory_aware_write else set()
    if model.cfg.payload_rank:
        allowed.update(('payload_adapter.coefficients', 'payload_adapter.basis'))
    if model.cfg.recall_slots:
        allowed.update(name for name in model.state_dict() if name.startswith('recall.'))
    if set(missing)-allowed or unexpected:
        raise ValueError(f'Incompatible parent weights: missing={missing}, unexpected={unexpected}')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parent_step=saved.get('step'), new_parameters=missing,
                optimizer='fresh AdamW', runtime_state='fresh independent streams')


def training_control(arm):
    return {'no_memory': 'blocked', 'fresh': 'fresh'}.get(arm, 'intact')


def validate_resume_destination(out, resume):
    """Keep another run's checkpoints/history out of a resumed study."""
    if resume is None:
        return
    out, resume = Path(out), Path(resume)
    if out.resolve() != resume.parent.resolve() and out.exists() and any(out.iterdir()):
        raise ValueError('Resuming into another directory requires an empty output directory.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--data-seed', type=int, default=123)
    p.add_argument('--arm', choices=('integrated', 'no_credit', 'no_memory', 'fresh'), default='integrated')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--context', type=int, default=16)
    p.add_argument('--width', type=int, default=80)
    p.add_argument('--write-mode', choices=('independent', 'competitive'), default='independent')
    p.add_argument('--memory-aware-write', action='store_true')
    p.add_argument('--bootstrap-empty', action='store_true')
    p.add_argument('--payload-rank', type=int, default=0)
    p.add_argument('--state-commit', choices=('always', 'publication'), default='always',
                   help='Advance all producer states or only owners committing a publication; candidate compute is unchanged.')
    p.add_argument('--recall-slots', type=int, default=0,
                   help='Bounded recall ports; zero uses the legacy pooled reader. Any positive count, including seven, is supported.')
    p.add_argument('--recall-layout', choices=('distinct', 'pooled'), default='distinct')
    p.add_argument('--recall-feedback', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--recall-hard', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--payload-credit', action='store_true',
                   help='Apply version-owned delayed payload-adapter gradients after backward.')
    p.add_argument('--stream-preset', choices=('basic', 'retention'), default='basic')
    p.add_argument('--curriculum', choices=('none', 'acquisition'), default='none')
    p.add_argument('--acquire-steps', type=int, default=200)
    p.add_argument('--bridge-steps', type=int, default=200)
    p.add_argument('--truncate-every', type=int, default=0,
                   help='Detach inside each context window; keep one optimizer update and whole-window loss.')
    p.add_argument('--owners', type=int, default=4)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--text-weight', type=float, default=.1)
    p.add_argument('--write-cost', type=float, default=0., help='Optional mean write-probability penalty; report separately.')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-count', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', type=Path)
    p.add_argument('--init-from', type=Path, help='Compatible learned weights for a new paired study; fresh optimizer/state.')
    p.add_argument('--final-eval', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.batch, args.context, args.eval_every, args.eval_count, args.threads) < 1:
        p.error('Positive step, batch, context, evaluation and thread counts required.')
    if min(args.acquire_steps, args.bridge_steps) < 1 or args.truncate_every < 0 or (
            args.truncate_every and args.context % args.truncate_every):
        p.error('Positive curriculum lengths and truncation zero or a divisor of context required.')
    if args.curriculum != 'none' and args.stream_preset != 'retention':
        p.error('The acquisition curriculum requires the retention target stream.')
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive.')
    if not 0 <= args.payload_rank <= args.width or (args.payload_credit and not args.payload_rank):
        p.error('Payload rank must fit width; delayed payload credit requires a positive rank.')
    try:
        validate_resume_destination(args.out, args.resume)
    except ValueError as error:
        p.error(str(error))
    if not 0 <= args.write_cost <= 1:
        p.error('Write cost must be between zero and one.')
    if not args.resume and any((args.out/name).exists() for name in ('best.pt', 'last.pt', 'history.jsonl')):
        p.error('Output already contains a run; use --resume or a fresh --out directory.')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    model = TaskMemoryModel(TaskMemoryConfig(width=args.width, owners=args.owners, write_mode=args.write_mode,
                                            memory_aware_write=args.memory_aware_write, bootstrap_empty=args.bootstrap_empty,
                                            payload_rank=args.payload_rank, state_commit=args.state_commit,
                                            recall_slots=args.recall_slots, recall_layout=args.recall_layout,
                                            recall_feedback=args.recall_feedback, recall_hard=args.recall_hard))
    initialization = initialize_weights(model, args.init_from) if args.init_from else None
    data_cfg = TaskStreamConfig() if args.stream_preset == 'basic' else TaskStreamConfig(
        entities='ABCD', queries_per_cycle=4, delays=(2,48), interference=True)
    saved = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    training_stage = stage_for_step(saved['step'] if saved else 0, args.curriculum,
                                    args.acquire_steps, args.bridge_steps)
    current_cfg = stage_config(data_cfg, training_stage)
    documents = make_documents(current_cfg, seed=args.data_seed, count=4096)
    streams = TaskByteStreams(documents, args.batch)
    coverage = SourceCoverage(documents)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    state = model.initial_state(args.batch)
    credit = make_eligibility(model, args.batch)
    start = 0; best = float('inf')
    external = experts = 0; delayed_norm = prior_seconds = 0.
    payload_norm = payload_update = 0.
    payload_credit_steps = 0
    stage_totals = {}
    if args.resume:
        validate_resume(saved, args, model, data_cfg)
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        state = TaskMemoryState.from_dict(saved['state']); credit = make_eligibility(model, args.batch, saved['credit'])
        streams.load_state_dict(saved['stream']); torch.set_rng_state(saved['rng'])
        start, best = saved['step'], saved['best']
        initialization = saved.get('initialization', initialization)
        totals = saved.get('totals', {})
        external = totals.get('external_bytes', start*args.batch*args.context)
        experts = totals.get('expert_executions', external*(args.owners+1)*3*
                             model.cfg.active_groups*model.cfg.experts_per_group)
        delayed_norm = totals.get('delayed_gradient_norm_sum', 0.)
        payload_norm = totals.get('payload_gradient_norm_sum', 0.)
        payload_update = totals.get('payload_update_norm_sum', 0.)
        payload_credit_steps = totals.get('payload_credit_steps', 0)
        prior_seconds = totals.get('elapsed_seconds', 0.)
        stage_totals = totals.get('stages', {})
        if not (args.out/'best.pt').exists():
            previous_best = args.resume.parent/'best.pt'
            if previous_best.exists():
                shutil.copy2(previous_best, args.out/'best.pt')
            else:
                raise FileNotFoundError('Resume requires the accompanying best.pt checkpoint.')
        if not (args.out/'last.pt').exists():
            shutil.copy2(args.resume, args.out/'last.pt')
        for stage_name in ('acquire', 'bridge'):
            previous_stage = args.resume.parent/(stage_name+'.pt')
            if previous_stage.exists() and not (args.out/previous_stage.name).exists():
                shutil.copy2(previous_stage, args.out/previous_stage.name)
    invocation = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'data_digest': streams.fingerprint, 'model_parameters': sum(p.numel() for p in model.parameters()),
                  'config': model.cfg.to_dict(), 'data_config': data_cfg.to_dict(),
                  'training_stages': {stage: stage_config(data_cfg, stage).to_dict()
                                      for stage in (('acquire', 'bridge', 'target')
                                                    if args.curriculum != 'none' else ('target',))},
                  'initialization': initialization}
    (args.out/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    started = time.perf_counter()
    model.train()
    for step in range(start+1, args.steps+1):
        next_stage = stage_for_step(step, args.curriculum, args.acquire_steps, args.bridge_steps)
        if next_stage != training_stage:
            training_stage = next_stage
            current_cfg = stage_config(data_cfg, training_stage)
            documents = make_documents(current_cfg, seed=args.data_seed, count=4096)
            streams = TaskByteStreams(documents, args.batch)
            coverage = SourceCoverage(documents)
            # New independent documents; learned weights and AdamW continue.
            state = model.initial_state(args.batch)
            credit = make_eligibility(model, args.batch)
        chunk = streams.next_chunk(args.context)
        optimizer.zero_grad(set_to_none=True)
        result, credit, boundaries = window_forward(model, chunk, state, credit,
            control=training_control(args.arm), truncate_every=args.truncate_every)
        loss, answer, text = objective(result, chunk, args.text_weight, args.write_cost)
        loss.backward()
        if args.arm == 'integrated':
            delayed_norm += sum(boundary.apply(model.write_policy) for boundary in boundaries)
        if args.payload_rank:
            current_payload_norm = sum(boundary.apply_payload(model.payload_adapter, args.payload_credit)
                                       for boundary in boundaries)
            payload_norm += current_payload_norm
            payload_credit_steps += int(current_payload_norm > 0)
            previous_payload = model.payload_adapter.coefficients.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if args.payload_rank:
            payload_update += float((model.payload_adapter.coefficients.detach() - previous_payload).norm())
        state = result['state'].detach()
        external += result['external_events']; experts += result['expert_executions']
        counts = coverage.count(chunk, args.truncate_every)
        counts.update(updates=1, external_bytes=result['external_events'],
                      expert_executions=result['expert_executions'],
                      producer_candidates=result['producer_candidates'], producer_commits=result['producer_commits'])
        accumulated = stage_totals.setdefault(training_stage, {key: 0 for key in counts})
        for key, value in counts.items():
            accumulated[key] = accumulated.get(key, 0) + value
        stage_end = training_stage != stage_for_step(step+1, args.curriculum,
                                                     args.acquire_steps, args.bridge_steps)
        if step % args.eval_every == 0 or step == args.steps or stage_end:
            evaluation, _ = evaluate(model, data_cfg, args.data_seed, args.eval_count, args.context,
                                     control=training_control(args.arm))
            stage_evaluation = evaluation if training_stage == 'target' else evaluate(
                model, current_cfg, args.data_seed, args.eval_count, args.context,
                control=training_control(args.arm))[0]
            elapsed = prior_seconds+time.perf_counter()-started
            totals = dict(external_bytes=external, expert_executions=experts,
                          elapsed_seconds=elapsed, delayed_gradient_norm_sum=delayed_norm,
                          payload_gradient_norm_sum=payload_norm, payload_update_norm_sum=payload_update,
                          payload_credit_steps=payload_credit_steps,
                          stages=stage_totals)
            record = dict(step=step, loss=float(loss.detach()), train_answer_nll=float(answer.detach()),
                          train_text_nll=float(text.detach()), validation=evaluation,
                          training_external_bytes=external, training_expert_executions=experts,
                          elapsed_seconds=elapsed, run_seconds=time.perf_counter()-started,
                          delayed_gradient_norm_sum=delayed_norm,
                          payload_gradient_norm_sum=payload_norm, payload_update_norm_sum=payload_update,
                          payload_credit_steps=payload_credit_steps,
                          training_stage=training_stage, stage_validation=stage_evaluation,
                          stage_totals=stage_totals,
                          trace_bytes=credit.storage_bytes())
            with (args.out/'history.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if evaluation['answer_nll'] < best:
                best = evaluation['answer_nll']
                checkpoint(args.out/'best.pt', model, optimizer, state, credit, streams, args, data_cfg, step, best, totals, initialization)
            checkpoint(args.out/'last.pt', model, optimizer, state, credit, streams, args, data_cfg, step, best, totals, initialization)
            if stage_end:
                shutil.copy2(args.out/'last.pt', args.out/(training_stage+'.pt'))
            print(json.dumps({k: record[k] for k in ('step', 'train_answer_nll', 'elapsed_seconds')} |
                             dict(val_nll=evaluation['answer_nll'], val_accuracy=evaluation['accuracy'])), flush=True)
    if args.final_eval:
        final_evaluation(args.out, model, data_cfg, args.data_seed, args.eval_count*2, args.context,
                         training_control(args.arm))


if __name__ == '__main__':
    main()
