"""Matched interventions over whole query windows, outside the live learner.

Snapshots exist only in this evaluator. Labels and event boundaries never enter
the model. Each fork begins immediately before the first query byte; only the
intact stream is carried onward. Holding writes isolates the publications that
were already committed before the query from newly computed query summaries.
In coupled state-commit mode, forced holds also protect producer contents.
"""
import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ..hat_minkron.hat import HATState
from .ports import map_tensors
from .task_memory import TaskMemoryConfig, TaskMemoryModel
from .task_memory_train import packed_documents, training_control
from .task_stream import TaskStreamConfig, make_documents


def select_state(state, rows):
    """Select independent rows, preserving the flattened producer-owner axis."""
    rows = torch.as_tensor(rows, dtype=torch.long, device=state.events_seen.device)
    batch, owners = len(state.events_seen), len(state.publications)
    owner_rows = (rows[:, None]*owners + torch.arange(owners, device=rows.device)).flatten()
    def choose(value):
        if value.shape[0] == batch:
            return value.index_select(0, rows)
        if value.shape[0] == batch*owners:
            return value.index_select(0, owner_rows)
        raise ValueError('Unexpected state tensor batch axis.')
    selected = map_tensors(state, choose)
    return replace(selected, ingress=replace(selected.ingress, batch_size=len(rows)))


def concatenate_states(states):
    """Combine pre-query snapshots; never combine or average their contents."""
    if not states:
        raise ValueError('At least one state is required.')
    def combine(values):
        first = values[0]
        if torch.is_tensor(first):
            return torch.cat(values, 0)
        if isinstance(first, HATState):
            return HATState(sum(v.batch_size for v in values),
                            combine([v.levels for v in values]), combine([v.clocks for v in values]))
        if is_dataclass(first):
            return replace(first, **{f.name: combine([getattr(v, f.name) for v in values]) for f in fields(first)})
        if isinstance(first, dict):
            if any(tuple(v) != tuple(first) for v in values):
                raise ValueError('State namespaces differ.')
            return {key: combine([v[key] for v in values]) for key in first}
        if isinstance(first, tuple):
            if any(len(v) != len(first) for v in values):
                raise ValueError('State tuple lengths differ.')
            return tuple(combine([v[i] for v in values]) for i in range(len(first)))
        if any(v != first for v in values):
            raise ValueError('State metadata differ.')
        return first
    return combine(states)


def shuffle_publication_content(state, donors, mask=None):
    """Exchange payload/header content; preserve recipient ownership and age.

    Donor identities are recorded by the evaluator. Keeping time metadata fixed
    makes this a content intervention, rather than an additional recency change.
    """
    donors = torch.as_tensor(donors, dtype=torch.long, device=state.events_seen.device)
    if mask is None:
        mask = torch.ones((len(donors), len(state.publications)), dtype=torch.bool,
                          device=donors.device)
    def exchange(value, owner):
        selected = mask[:, owner].reshape(-1, *([1]*(value.ndim-1)))
        return torch.where(selected, value[donors], value)
    return replace(state, publications={name: replace(pub, payload=exchange(pub.payload, owner),
        header=map_tensors(pub.header, lambda x: exchange(x, owner)))
        for owner, (name, pub) in enumerate(state.publications.items())})


def aged_publication_shuffle(state, donors, age_threshold):
    """Replace old, existing cells with valid other-document donor content.

    Eligibility and actual packed-content changes are distinct: early producer
    publications can contain identical payloads even when their versions differ.
    """
    if age_threshold < 0:
        raise ValueError('Age threshold must be nonnegative.')
    donors = torch.as_tensor(donors, dtype=torch.long, device=state.events_seen.device)
    existing = torch.stack([p.valid for p in state.publications.values()], 1)
    occurrence = torch.stack([p.occurrence for p in state.publications.values()], 1)
    ages = state.events_seen[:, None]-occurrence
    eligible = existing & (ages > age_threshold)
    replaced = eligible & existing[donors] & (donors != torch.arange(len(donors), device=donors.device))[:, None]
    shuffled = shuffle_publication_content(state, donors, replaced)
    changed = (state.memory_tensor() != shuffled.memory_tensor()).any(-1)
    return shuffled, dict(existing=existing, ages=ages, age_eligible=eligible,
                          replaced=replaced, content_changed=changed)


def shuffle_producer_content(state, donors, mask):
    """Exchange owner recurrent contents while preserving every execution clock."""
    batch, owners = len(state.events_seen), len(state.publications)
    donors = torch.as_tensor(donors, dtype=torch.long, device=state.events_seen.device)
    if donors.shape != (batch,) or mask.shape != (batch, owners) or mask.dtype != torch.bool:
        raise ValueError('Producer donors must be [B] and mask must be bool [B,owners].')
    def exchange(value):
        shaped = value.reshape(batch, owners, *value.shape[1:])
        selected = mask.reshape(batch, owners, *([1] * (value.ndim - 1)))
        return torch.where(selected, shaped[donors], shaped).reshape_as(value)
    return replace(state, producer=replace(state.producer,
        contextualizer=tuple(exchange(x) for x in state.producer.contextualizer),
        blocks=tuple(exchange(x) for x in state.producer.blocks)))


def aged_producer_shuffle(state, donors, age_threshold, state_commit):
    """Age refers to the producer commit, never an unrelated publication age."""
    if age_threshold < 0 or state_commit not in ('always', 'publication'):
        raise ValueError('Nonnegative age and a known state-commit mode are required.')
    batch, owners = len(state.events_seen), len(state.publications)
    donors = torch.as_tensor(donors, dtype=torch.long, device=state.events_seen.device)
    if state_commit == 'publication':
        existing = torch.stack([p.valid for p in state.publications.values()], 1)
        occurrence = torch.stack([p.occurrence for p in state.publications.values()], 1)
        ages = (state.events_seen[:, None] - occurrence) * existing
    else:
        existing = state.events_seen.gt(0)[:, None].expand(batch, owners)
        ages = existing.long()  # Every owner advanced at the preceding event.
    eligible = existing & (ages > age_threshold)
    replaced = eligible & existing[donors] & (donors != torch.arange(batch, device=donors.device))[:, None]
    shuffled = shuffle_producer_content(state, donors, replaced)
    changed = torch.zeros_like(existing)
    for old, new in zip((*state.producer.contextualizer, *state.producer.blocks),
                        (*shuffled.producer.contextualizer, *shuffled.producer.blocks)):
        changed |= (old != new).flatten(1).any(-1).reshape(batch, owners)
    return shuffled, dict(existing=existing, ages=ages, age_eligible=eligible,
                          replaced=replaced, content_changed=changed)


def query_trace(state, result, values, blocked=False):
    """Per-byte publication ages refer to the pre-event read snapshot."""
    occurrence = torch.stack([p.occurrence for p in state.publications.values()], 1)
    valid = torch.stack([p.valid for p in state.publications.values()], 1)
    rows = [[] for _ in range(values.shape[0])]
    for t in range(values.shape[1]):
        clock = state.events_seen+t
        source = result.get('read_source', 'publication')
        actual_valid = result['read_valid'][:, t] if 'read_valid' in result else valid
        ages = torch.zeros_like(occurrence) if source == 'candidate' else clock[:, None]-occurrence
        weights = result['read_weights'][:, t, :-1]*actual_valid
        mass = weights.sum(-1)
        weighted_age = (weights*ages).sum(-1)/mass.clamp_min(1e-12)
        for row in range(values.shape[0]):
            rows[row].append(dict(offset=t, byte=int(values[row, t]),
                read_weights=result['read_weights'][row, t].tolist(), read_source=source,
                read_versions=result['read_versions'][row, t].tolist(),
                read_ages=[int(a) if bool(v) else None for a, v in zip(ages[row], actual_valid[row])],
                read_mass=float(mass[row]), effective_read_mass=0. if blocked else float(mass[row]),
                weighted_publication_age=float(weighted_age[row]) if mass[row] > 0 else None,
                write_actions=result['write_actions'][row, t].tolist(),
                state_actions=result['state_actions'][row, t].tolist() if 'state_actions' in result else None))
        actions = result['write_actions'][:, t]
        occurrence = torch.where(actions, clock[:, None], occurrence)
        valid = valid | actions
    return rows


@torch.no_grad()
def query_forks(model, values, state, donors=None, include_fresh=False, live_control='intact', age_threshold=None,
                include_producer=False):
    """Return query predictions and traces from exactly the same pre-query state."""
    if live_control not in ('intact', 'fresh', 'blocked'):
        raise ValueError('Live control must be intact, fresh, or blocked.')
    outputs = {}
    settings = [('intact', live_control, None), ('blocked', 'blocked', None),
                ('hold', live_control, 'hold'), ('hold_blocked', 'blocked', 'hold')]
    if include_fresh:
        settings.append(('fresh', 'fresh', None))
    for name, control, write_override in settings:
        result = model(values, state=state, control=control, write_override=write_override)
        outputs[name] = dict(result=result, trace=query_trace(state, result, values, control == 'blocked'))
    if donors is not None:
        shuffled = shuffle_publication_content(state, donors)
        for name, write_override in [('shuffle', None), ('hold_shuffle', 'hold')]:
            result = model(values, state=shuffled, control=live_control, write_override=write_override)
            outputs[name] = dict(result=result, trace=query_trace(shuffled, result, values, live_control == 'blocked'))
        if age_threshold is not None:
            aged, metadata = aged_publication_shuffle(state, donors, age_threshold)
            result = model(values, state=aged, control=live_control)
            outputs['aged_shuffle'] = dict(result=result, age_filter=metadata,
                trace=query_trace(aged, result, values, live_control == 'blocked'))
        if include_producer:
            for name, threshold in (('producer_shuffle', 0), ('aged_producer_shuffle', age_threshold)):
                if threshold is None:
                    continue
                shuffled, metadata = aged_producer_shuffle(state, donors, threshold, model.cfg.state_commit)
                result = model(values, state=shuffled, control=live_control)
                outputs[name] = dict(result=result, age_filter=metadata,
                    trace=query_trace(shuffled, result, values, live_control == 'blocked'))
    # Reconstruct the intact prefix and block the terminal answer event alone.
    prefix_state = state
    prefix_executions = 0
    if values.shape[1] > 1:
        prefix = model(values[:, :-1], state=state, control=live_control)
        prefix_state, prefix_executions = prefix['state'], prefix['expert_executions']
    terminal = model(values[:, -1:], state=prefix_state, control='blocked')
    terminal['expert_executions'] += prefix_executions
    outputs['terminal_blocked'] = dict(result=terminal,
        trace=query_trace(prefix_state, terminal, values[:, -1:], True))
    expected = outputs['intact']['result']['expert_executions']
    if any(item['result']['expert_executions'] != expected for item in outputs.values()):
        raise AssertionError('Query interventions must have equal expert budgets.')
    return outputs


@torch.no_grad()
def audit(model, data_config, *, data_seed=123, split='test', count=16, include_fresh=False, live_control='intact',
          age_threshold=16, include_producer=False):
    """Audit fresh documents, with query event labels used for evaluation only."""
    if live_control not in ('intact', 'fresh', 'blocked'):
        raise ValueError('Live control must be intact, fresh, or blocked.')
    if count < 2:
        raise ValueError('At least two independent documents are needed for content shuffling.')
    if age_threshold is not None and age_threshold < 0:
        raise ValueError('Age threshold must be nonnegative.')
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        documents = make_documents(data_config, split=split, seed=data_seed, count=count)
        inputs, _, _, valid, _ = packed_documents(documents)
        starts, ends = {}, {}
        for row, doc in enumerate(documents):
            for event in doc.events:
                if event.query_index is not None:
                    starts.setdefault(event.start, []).append((row, event))
                    ends[(row, event.stop-1)] = (row, event.query_index)
        state = model.initial_state(count)
        snapshots, live_logits = {}, {}
        for position in range(inputs.shape[1]):
            for row, event in starts.get(position, ()):
                snapshots[row, event.query_index] = select_state(state, [row]).detach()
            result = model(inputs[:, position:position+1], state=state, control=live_control)
            for row in range(count):
                if (row, position) in ends:
                    live_logits[ends[row, position]] = result['logits'][row, 0].clone()
            state = result['state']
        grouped = {}
        for row, doc in enumerate(documents):
            for event in doc.events:
                if event.query_index is not None:
                    grouped.setdefault(len(event.text), []).append((row, event))
        records = []
        max_drift = 0.
        argmax_changes = query_executions = 0
        for length, items in sorted(grouped.items()):
            states = [snapshots[row, event.query_index] for row, event in items]
            batch_state = concatenate_states(states)
            values = torch.tensor([list(event.text) for _, event in items], dtype=torch.long)
            donors, donor_valid = [], []
            for i, (row, _) in enumerate(items):
                other = next((j for j in range(i+1, len(items)) if items[j][0] != row),
                             next((j for j in range(i) if items[j][0] != row), None))
                donors.append(i if other is None else other)
                donor_valid.append(other is not None)
            forks = query_forks(model, values, batch_state, donors, include_fresh, live_control, age_threshold,
                                include_producer)
            query_executions += forks['intact']['result']['expert_executions']
            for i, (row, event) in enumerate(items):
                q = documents[row].queries[event.query_index]
                record = dict(document=row, query_index=event.query_index, query_start=event.start,
                    query_stop=event.stop, query_text=event.text.decode('ascii'), **asdict(q),
                    prequery_versions=[int(p.version[i]) for p in batch_state.publications.values()], branches={})
                reference = live_logits[row, event.query_index]
                prediction = forks['intact']['result']['logits'][i, -1]
                max_drift = max(max_drift, float((prediction-reference).abs().max()))
                argmax_changes += int(prediction.argmax() != reference.argmax())
                for name, item in forks.items():
                    logits = item['result']['logits'][i, -1]
                    branch = dict(prediction=int(logits.argmax()), correct=bool(logits.argmax() == q.answer),
                        nll=float(F.cross_entropy(logits[None], torch.tensor([q.answer]))), trace=item['trace'][i])
                    if 'shuffle' in name:
                        branch.update(eligible=donor_valid[i], donor_document=items[donors[i]][0],
                                      donor_query_index=items[donors[i]][1].query_index)
                    if 'age_filter' in item:
                        branch['age_filter'] = {key: value[i].tolist() for key, value in item['age_filter'].items()}
                    record['branches'][name] = branch
                records.append(record)
        records.sort(key=lambda r: (r['document'], r['query_start']))
        if max_drift > 1e-4 or argmax_changes:
            raise AssertionError(f'Rebatched intact fork disagrees with live stream: drift={max_drift}, argmax={argmax_changes}')
        groups = {'all': records}
        for name in ('exact', 'semantic', 'context', 'relation'):
            groups[name] = [r for r in records if r['kind'] == name]
        groups['corrected_exact'] = [r for r in records if r['kind'] == 'exact' and r['correction']]
        def score(rows, branch):
            items = [r['branches'][branch] for r in rows if r['branches'][branch].get('eligible', True)]
            if not items:
                return dict(count=0, accuracy=None, nll=None)
            scored = dict(count=len(items), accuracy=sum(x['correct'] for x in items)/len(items),
                          nll=sum(x['nll'] for x in items)/len(items))
            if branch == 'terminal_blocked':
                return scored  # Its compact trace omits the intact prefix.
            all_mass, prefix_mass, terminal_mass, prequery_mass = [], [], [], []
            age_mass = total_mass = 0.
            for record in rows:
                item = record['branches'][branch]
                if not item.get('eligible', True):
                    continue
                for t, part in enumerate(item['trace']):
                    mass = part['effective_read_mass']
                    all_mass.append(mass)
                    (terminal_mass if t == len(item['trace'])-1 else prefix_mass).append(mass)
                    retained = 0.
                    if part['read_source'] != 'candidate' and mass > 0:
                        retained = sum(weight for weight, version, original, age in zip(
                            part['read_weights'][:-1], part['read_versions'], record['prequery_versions'],
                            part['read_ages']) if version == original and age is not None)
                    prequery_mass.append(retained)
                    if part['weighted_publication_age'] is not None:
                        age_mass += mass*part['weighted_publication_age']; total_mass += mass
            average = lambda values: sum(values)/len(values) if values else None
            scored.update(mean_query_read_mass=average(all_mass),
                          mean_preterminal_read_mass=average(prefix_mass),
                          terminal_read_mass=average(terminal_mass),
                          prequery_publication_read_mass=average(prequery_mass),
                          consumed_read_age=age_mass/total_mass if total_mass else None)
            return scored
        summary = dict(schema=1, split=split, data_seed=data_seed, documents=count,
            state_commit=model.cfg.state_commit,
            live_control=live_control,
            live_read_source={'intact': 'publication', 'fresh': 'candidate', 'blocked': 'disabled'}[live_control],
            query_count=len(records), groups={name: {branch: score(rows, branch) for branch in
                records[0]['branches']} for name, rows in groups.items() if rows},
            live_fork_max_logit_error=max_drift, live_fork_argmax_changes=argmax_changes,
            live_external_bytes=int(valid.sum()), live_executed_bytes=inputs.numel(),
            intact_query_expert_executions=query_executions, seconds=time.perf_counter()-started,
            interventions=dict(intact='Unmodified trained read control: '+live_control,
                blocked='Reads zeroed from first through final query byte.',
                terminal_blocked='Only final query byte has reads zeroed; prefix stays intact.',
                hold='Writes held throughout query, leaving pre-query publications unchanged.',
                hold_blocked='Both held writes and blocked reads throughout query.',
                shuffle='Other-document payload/header replaces initial publication content; normal query writes continue.',
                aged_shuffle='Only existing pre-query cells strictly older than age_threshold receive valid other-document content; metadata is preserved and normal query writes continue.',
                hold_shuffle='Other-document payload/header with publications held throughout query.',
                fresh='Current producer candidates replace read values; memory-aware admission can still consult stored publications.'),
            scope='All forks preserve pre-query ingress and composer recurrence. Publication shuffles also preserve producer recurrence. Producer shuffles change only producer contents. Normal-write shuffle branches retain the trained read control. In publication state-commit mode, hold-versus-intact changes both publication and producer commits; comparisons within the hold branches share the same state hold. Fresh uses current candidates; blocked disables the read contribution.')
        if include_producer:
            summary['interventions'].update(
                producer_shuffle='Other-document committed producer contents replace recipient contents; publications and all clocks are preserved.',
                aged_producer_shuffle='Only producer contents strictly older than age_threshold receive other-document committed contents; publications and all clocks are preserved.')
            def producer_metrics(rows, branch):
                masks = [r['branches'][branch]['age_filter'] for r in rows]
                changed = [r for r, mask in zip(rows, masks) if any(mask['content_changed'])]
                return dict(query_count=len(rows),
                    **{key + '_slot_count': sum(sum(mask[key]) for mask in masks)
                       for key in ('existing', 'age_eligible', 'replaced', 'content_changed')},
                    changed_query_count=len(changed),
                    changed_queries={name: score(changed, name) for name in ('intact', branch)})
            for branch, threshold in (('producer_shuffle', 0), ('aged_producer_shuffle', age_threshold)):
                if threshold is None:
                    continue
                summary[branch] = dict(age_threshold=threshold, age_comparison='strictly_greater',
                    age_source='publication commit' if model.cfg.state_commit == 'publication' else 'preceding event',
                    count_unit='Owner state at query start; repeated owners across queries count separately.',
                    changed_content='Exact difference across producer contextualizer and recurrent block contents; excludes clocks.',
                    groups={name: producer_metrics(rows, branch) for name, rows in groups.items() if rows})
        if age_threshold is not None:
            def age_metrics(rows):
                masks = [r['branches']['aged_shuffle']['age_filter'] for r in rows]
                counts = {key: sum(sum(m[key]) for m in masks)
                          for key in ('existing', 'age_eligible', 'replaced', 'content_changed')}
                queries = {key: sum(any(m[key]) for m in masks)
                           for key in ('age_eligible', 'replaced', 'content_changed')}
                changed_rows = [r for r, m in zip(rows, masks) if any(m['content_changed'])]
                masses = dict(age_eligible=0., content_changed=0.)
                events = 0
                for record, mask in zip(rows, masks):
                    for part in record['branches']['intact']['trace']:
                        events += 1
                        if part['read_source'] != 'publication' or part['effective_read_mass'] <= 0:
                            continue
                        for key in masses:
                            masses[key] += sum(weight for selected, weight, version, original, age in zip(
                                mask[key], part['read_weights'][:-1], part['read_versions'],
                                record['prequery_versions'], part['read_ages'])
                                if selected and version == original and age is not None)
                return dict(query_count=len(rows), existing_slot_count=counts['existing'],
                    age_eligible_slot_count=counts['age_eligible'], age_eligible_query_count=queries['age_eligible'],
                    replaced_slot_count=counts['replaced'], replaced_query_count=queries['replaced'],
                    changed_slot_count=counts['content_changed'], changed_query_count=queries['content_changed'],
                    changed_fraction_of_existing=counts['content_changed']/counts['existing'] if counts['existing'] else None,
                    intact_age_eligible_read_mass=masses['age_eligible']/events if events else None,
                    intact_changed_read_mass=masses['content_changed']/events if events else None,
                    changed_queries={branch: score(changed_rows, branch) for branch in ('intact', 'aged_shuffle')})
            summary['aged_shuffle'] = dict(age_threshold=age_threshold, age_comparison='strictly_greater',
                count_unit='Publication cells at query start; repeated cells across queries count separately.',
                changed_content='Exact difference in model-readable packed payload and soft header memberships.',
                read_mass_scope='Mean intact per-query-byte attention on eligible/changed pre-query versions only; excludes replaced versions and candidate reads.',
                groups={name: age_metrics(rows) for name, rows in groups.items() if rows})
        return summary, records
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--split', choices=('test', 'long_delay'), default='test')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--fresh-control', action='store_true', help='Also read current producer candidates during each query.')
    parser.add_argument('--producer-state', action='store_true', help='Also shuffle committed producer contents and age-filtered producer contents.')
    parser.add_argument('--age-threshold', type=int, help='Old-cell shuffle threshold; defaults to checkpoint training context.')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = TaskMemoryModel(TaskMemoryConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights'])
    config = TaskStreamConfig.from_dict(saved['data_config'])
    summary, records = audit(model, config, data_seed=saved['args']['data_seed'],
                             split=args.split, count=args.count, include_fresh=args.fresh_control,
                             live_control=training_control(saved['args']['arm']),
                             include_producer=args.producer_state,
                             age_threshold=args.age_threshold if args.age_threshold is not None else saved['args'].get('context', 16))
    summary.update(checkpoint=str(args.checkpoint), selected_step=saved['step'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.out/'queries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(summary['groups']['all']))


if __name__ == '__main__':
    main()
