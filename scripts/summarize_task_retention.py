"""Summarize the completed five-arm persistent-publication retention pilot.

Reads JSON evidence only; requires selected and final evaluation plus the
selected-checkpoint whole-query audit before copying or writing any deliverable.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil

from summarize_task_memory import read_jsonl, table, number, pct, trace_metrics, mean

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('legacy', 'conditioned', 'competitive', 'no_credit', 'fresh')
LABELS = dict(legacy='Legacy', conditioned='Conditioned', competitive='Competitive',
              no_credit='No delayed credit', fresh='Trained fresh')
EXPECTED = dict(legacy=('integrated', 'independent', False, False),
                conditioned=('integrated', 'independent', True, True),
                competitive=('integrated', 'competitive', True, True),
                no_credit=('no_credit', 'competitive', True, True),
                fresh=('fresh', 'competitive', True, True))
EVIDENCE = ('invocation.json', 'history.jsonl', 'evaluation.json', 'test_traces.jsonl',
            'long_delay_traces.jsonl', 'final_test_traces.jsonl', 'final_long_delay_traces.jsonl',
            'query_audit/summary.json', 'query_audit/queries.jsonl')
SOURCE_BASE = 'fd4d1ac'
MAX_REPO_TRACE_BYTES = 128*1024


def group(evaluation, kind='exact'):
    return evaluation['groups'][kind]


def audit_group(audit, kind='exact'):
    return audit['groups'][kind]


def query_metrics(records, context):
    """Full-query provenance; never infer retrieval from the terminal alone."""
    groups = {}
    for kind in ('all', 'exact', 'semantic', 'context', 'corrected_exact'):
        subset = [r for r in records if kind == 'all' or r['kind'] == kind or
                  (kind == 'corrected_exact' and r['kind'] == 'exact' and r['correction'])]
        if not subset:
            continue
        old_mass = []
        source_delay = []
        for record in subset:
            source_delay.append(record['delay'])
            for part in record['branches']['intact']['trace']:
                old_mass.append(0. if part['read_source'] == 'candidate' else
                    sum(w for w, age in zip(part['read_weights'][:-1], part['read_ages'])
                        if age is not None and age > context))
        groups[kind] = dict(queries=len(subset), query_byte_count=len(old_mass),
            mean_old_publication_read_mass=mean(old_mass), old_threshold_bytes=context,
            mean_source_fact_delay=mean(source_delay))
    return groups


def load_study(runs, arms=ARMS, initialization_mode='scratch'):
    loaded = {}
    for arm in arms:
        folder = runs/arm
        for filename in EVIDENCE:
            if not (folder/filename).is_file():
                raise FileNotFoundError(f'Incomplete retention study: {folder/filename}')
        invocation = json.loads((folder/'invocation.json').read_text())
        evaluation = json.loads((folder/'evaluation.json').read_text())
        history = read_jsonl(folder/'history.jsonl')
        audit = json.loads((folder/'query_audit/summary.json').read_text())
        audit_traces = read_jsonl(folder/'query_audit/queries.jsonl')
        initialization = invocation.get('initialization')
        if initialization_mode == 'scratch':
            if initialization or invocation.get('init_from'):
                raise ValueError(f'{arm}: scratch cohort has parent initialization.')
        elif not initialization or initialization.get('optimizer') != 'fresh AdamW' or initialization.get('runtime_state') != 'fresh independent streams':
            raise ValueError(f'{arm}: continuation requires common weights with fresh optimizer/runtime.')
        expected = EXPECTED[arm]
        actual = tuple(invocation[k] for k in ('arm', 'write_mode', 'memory_aware_write', 'bootstrap_empty'))
        if actual != expected:
            raise ValueError(f'{arm}: unexpected training/writer modes: {actual}')
        for key in ('write_mode', 'memory_aware_write', 'bootstrap_empty'):
            if invocation[key] != invocation['config'][key]:
                raise ValueError(f'{arm}: invocation/model {key} mismatch.')
        if invocation['write_cost'] != 0 or invocation['stream_preset'] != 'retention':
            raise ValueError(f'{arm}: this report requires the unpenalized retention preset.')
        data = invocation['data_config']
        required = dict(entities='ABCD', attributes=['x'], values='0123', cycles=2,
                        queries_per_cycle=4, delays=[2, 48], interference=True)
        if any(data.get(k) != v for k, v in required.items()):
            raise ValueError(f'{arm}: unexpected retention stream protocol.')
        if (not history or history[-1]['step'] != invocation['steps'] or
                evaluation['final_budget']['step'] != invocation['steps']):
            raise ValueError(f'{arm}: training/final evaluation is incomplete.')
        selected = next((h for h in reversed(history) if h['step'] == evaluation['selected_step']), None)
        if selected is None or audit['selected_step'] != evaluation['selected_step']:
            raise ValueError(f'{arm}: selected history/audit checkpoint mismatch.')
        control = 'fresh' if arm == 'fresh' else 'intact'
        if audit.get('live_control') != control or evaluation['test']['control'] != control:
            raise ValueError(f'{arm}: audit/evaluation does not use its trained read control.')
        if audit['data_seed'] != invocation['data_seed'] or audit['split'] != 'test':
            raise ValueError(f'{arm}: audit uses different test data.')
        if len(audit_traces) != audit['query_count'] or audit['query_count'] != evaluation['test']['answers']:
            raise ValueError(f'{arm}: audit answer count mismatch.')
        if audit['live_fork_argmax_changes'] or audit['live_fork_max_logit_error'] > 1e-4:
            raise ValueError(f'{arm}: intact audit forks disagree with their live rollout.')
        for branch in ('intact', 'blocked', 'hold', 'hold_blocked', 'shuffle', 'hold_shuffle', 'fresh'):
            if branch not in audit_group(audit):
                raise ValueError(f'{arm}: missing exact-query {branch} intervention.')
        traces = {name: read_jsonl(folder/(name+'_traces.jsonl'))
                  for name in ('test', 'long_delay', 'final_test', 'final_long_delay')}
        for name, rows in traces.items():
            split = name.removeprefix('final_')
            metrics = evaluation['final_budget'][split] if name.startswith('final_') else evaluation[split]
            if len(rows) != metrics['answers']:
                raise ValueError(f'{arm}/{name}: trace/evaluation count mismatch.')
            expected_bootstrap = invocation['owners']*invocation['eval_count']*2 if expected[3] else 0
            if metrics['bootstrap_events'] != expected_bootstrap:
                raise ValueError(f'{arm}/{name}: forced startup commits differ from protocol.')
        loaded[arm] = dict(invocation=invocation, evaluation=evaluation, history=history,
            selected_history=selected, audit=audit, audit_derived=query_metrics(audit_traces, invocation['context']),
            audit_samples=sorted(audit_traces, key=lambda r: (r['document'], r['query_start']))[:4],
            raw_trace_counts={**{name+'_traces.jsonl': len(rows) for name, rows in traces.items()},
                              'query_audit/queries.jsonl': len(audit_traces)},
            derived={name: trace_metrics(rows) for name, rows in traces.items()})
    reference = loaded['competitive']
    paired = ('seed', 'data_seed', 'data_digest', 'batch', 'context', 'steps', 'width', 'owners',
              'lr', 'text_weight', 'write_cost', 'eval_count', 'eval_every', 'data_config')
    allowed_config_differences = {'write_mode', 'memory_aware_write', 'bootstrap_empty'}
    for arm, run in loaded.items():
        for key in paired:
            if run['invocation'][key] != reference['invocation'][key]:
                raise ValueError(f'Unpaired retention study: {arm} differs in {key}.')
        for key, value in reference['invocation']['config'].items():
            if key not in allowed_config_differences and run['invocation']['config'][key] != value:
                raise ValueError(f'{arm}: unrelated architecture change in {key}.')
        expected_parameters = reference['invocation']['model_parameters']-(864 if arm == 'legacy' else 0)
        if run['invocation']['model_parameters'] != expected_parameters:
            raise ValueError(f'{arm}: parameter count differs from expected conditioning-only change.')
        for key in ('training_external_bytes', 'training_expert_executions'):
            if run['history'][-1][key] != reference['history'][-1][key]:
                raise ValueError(f'{arm}: unmatched final {key}.')
        for split in ('test', 'long_delay'):
            for key in ('answers', 'external_bytes', 'executed_bytes', 'expert_executions'):
                if run['evaluation'][split][key] != reference['evaluation'][split][key]:
                    raise ValueError(f'{arm}/{split}: unmatched {key}.')
        for key in ('documents', 'query_count', 'live_external_bytes', 'live_executed_bytes',
                    'intact_query_expert_executions'):
            if run['audit'][key] != reference['audit'][key]:
                raise ValueError(f'{arm}: unmatched query audit {key}.')
    return loaded


def aged_evidence(run):
    audit = run['audit']
    metadata = audit.get('aged_shuffle')
    if metadata is None or 'exact' not in metadata['groups']:
        return dict(status='unestablished', reason='No age-filtered exact-query audit.')
    details = metadata['groups']['exact']
    paired = details['changed_queries']
    count = paired['intact']['count']
    delta = None if not count else paired['aged_shuffle']['nll']-paired['intact']['nll']
    exact = audit_group(audit)
    all_delta = None
    if 'aged_shuffle' in exact and exact['aged_shuffle']['count'] == exact['intact']['count']:
        all_delta = exact['aged_shuffle']['nll']-exact['intact']['nll']
    return dict(status='unestablished' if not count else ('directional_positive' if delta > 0 else 'unmet'),
        age_threshold=metadata['age_threshold'], changed_queries=count,
        changed_slots=details['changed_slot_count'], age_eligible_slots=details['age_eligible_slot_count'],
        intact_changed_read_mass=details['intact_changed_read_mass'],
        paired_changed_query_exact_nll_delta=delta, all_query_exact_nll_delta=all_delta,
        meaning='Positive paired loss increase after shuffling only older pre-query content would support its use; zero affected queries provide no test.')


def differences(run, fresh):
    selected, final = run['evaluation']['test'], run['evaluation']['final_budget']['test']
    q = audit_group(run['audit'])
    return dict(selected_exact_gain_vs_trained_fresh=group(fresh['evaluation']['test'])['nll']-group(selected)['nll'],
        final_exact_gain_vs_trained_fresh=group(fresh['evaluation']['final_budget']['test'])['nll']-group(final)['nll'],
        long_exact_gain_vs_trained_fresh=group(fresh['evaluation']['long_delay'])['nll']-group(run['evaluation']['long_delay'])['nll'],
        normal_write_content_shuffle_exact_delta=q['shuffle']['nll']-q['intact']['nll'],
        aged_content_evidence=aged_evidence(run),
        age_filtered_audit=run['audit'].get('aged_shuffle'),
        optional_age_filtered_shuffle={name: dict(nll=value['nll'], count=value['count'],
                                      comparison='Use paired affected-query contrast in age_filtered_audit; no subtraction from unmatched all-query intact loss.')
                                      for name, value in q.items() if 'shuffle' in name and name not in ('shuffle', 'hold_shuffle')},
        whole_query_blocked_exact_delta=q['blocked']['nll']-q['intact']['nll'],
        held_publication_exact_gain=q['hold_blocked']['nll']-q['hold']['nll'],
        held_content_shuffle_exact_delta=q['hold_shuffle']['nll']-q['hold']['nll'])


def commits_per_thousand(metrics):
    return 1000*metrics['learned_writes']/max(metrics['external_bytes']-metrics['bootstrap_events'], 1)


def load_continuation(path, scratch):
    arms = ('conditioned', 'competitive', 'fresh')
    if (path/'no_credit').exists():
        arms = ('conditioned', 'competitive', 'no_credit', 'fresh')
    warm = load_study(path, arms, initialization_mode='continuation')
    reference = scratch['competitive']
    initializations = [run['invocation']['initialization'] for run in warm.values()]
    if len({value['parent_step'] for value in initializations}) != 1:
        raise ValueError('Continuation arms disagree on the selected parent step.')
    if len({value['sha256'] for value in initializations}) != 1:
        raise ValueError('Continuation arms do not share an identical parent checkpoint.')
    for value in initializations:
        if not Path(value['path']).as_posix().endswith('artifacts/task_memory/pilot_v2/integrated/best.pt'):
            raise ValueError('Unexpected continuation parent checkpoint.')
    for arm, run in warm.items():
        for key in ('seed', 'data_seed', 'data_digest', 'batch', 'context', 'steps', 'lr', 'text_weight', 'data_config'):
            if run['invocation'][key] != reference['invocation'][key]:
                raise ValueError(f'Continuation {arm}: differs from retention protocol in {key}.')
        for key in ('training_external_bytes', 'training_expert_executions'):
            if run['history'][-1][key] != reference['history'][-1][key]:
                raise ValueError(f'Continuation {arm}: unmatched final {key}.')
    return warm


def directional_support(compare):
    return [arm for arm, d in compare.items() if d['selected_exact_gain_vs_trained_fresh'] > 0
            and d['normal_write_content_shuffle_exact_delta'] > 0]


def validation_evidence(artifact_root):
    regression = artifact_root/'regression.log'
    focused = artifact_root/'final_focused.log'
    main_text, focused_text = regression.read_text(), focused.read_text()
    suites = [int(n) for n in re.findall(r'^Ran (\d+) tests?', main_text, re.MULTILINE)]
    focused_runs = [int(n) for n in re.findall(r'^Ran (\d+) tests?', focused_text, re.MULTILINE)]
    if (len(suites) != 5 or main_text.count('\nOK\n') != 5 or
            'All five experiment suites passed.' not in main_text or 'FAILED (' in main_text):
        raise ValueError('Five-suite regression log is incomplete or failing.')
    if len(focused_runs) != 1 or '\nOK\n' not in focused_text or 'FAILED (' in focused_text:
        raise ValueError('Final focused-check log is incomplete or failing.')
    return dict(suite_counts=suites, full_regression_checks=sum(suites), final_focused_checks=focused_runs[0],
                logs={p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
                      for p in (regression, focused)})


def parent_training_budget(warm):
    initialization = warm['competitive']['invocation']['initialization']
    parent = Path(initialization['path'])
    if not parent.is_absolute():
        parent = ROOT/parent
    history = read_jsonl(parent.parent/'history.jsonl')
    selected = next((r for r in reversed(history) if r['step'] == initialization['parent_step']), None)
    if selected is None:
        raise ValueError('Selected parent has no matching training-budget record.')
    return dict(selected_step=selected['step'], external_bytes=selected['training_external_bytes'],
                expert_executions=selected['training_expert_executions'], checkpoint_sha256=initialization['sha256'])


def summarize(runs, output_root, baseline, continuation=None):
    study = load_study(runs)
    validation = validation_evidence(runs.parent)
    warm = load_continuation(continuation, study) if continuation else None
    baseline_path = baseline/'aggregate.json'
    historical = json.loads(baseline_path.read_text())
    if historical['seeds'] != [17, 18, 19]:
        raise ValueError('Expected historical three-seed query audit.')
    cfg = study['competitive']['invocation']
    compare = {arm: differences(run, study['fresh']) for arm, run in study.items() if arm != 'fresh'}
    supported = directional_support(compare)
    credit_gain = dict(selected_exact_nll=group(study['no_credit']['evaluation']['test'])['nll']-
                      group(study['competitive']['evaluation']['test'])['nll'],
                      final_exact_nll=group(study['no_credit']['evaluation']['final_budget']['test'])['nll']-
                      group(study['competitive']['evaluation']['final_budget']['test'])['nll'])
    summary = dict(schema=1, source_base=SOURCE_BASE, protocol='docs/SHoEbox_Task_Retention_Protocol.md',
        scope='One paired initialization/data seed, five arms, fixed training budget',
        validation=validation, paired_seed=cfg['seed'], paired_data_seed=cfg['data_seed'], data_digest=cfg['data_digest'],
        primary_metric='Selected held-out exact-value NLL versus the separately trained fresh-candidate control',
        comparisons=compare, delayed_credit_exact_gain=credit_gain,
        phase_status=dict(persistent_publication_directional_support=supported,
            replicated=False, delayed_credit_selected_exact_gain=credit_gain['selected_exact_nll'] > 0,
            usefulness_of_publications_older_than_chunk={arm: aged_evidence(run) for arm, run in study.items()}),
        interpretation=dict(query_audit='All query forks preserve pre-query ingress/producer/composer recurrence. Held-publication contrasts isolate direct access to pre-query content, not historical information already absorbed into recurrent state.',
            fresh_control='Separately trained current-candidate reads retain recurrent experts and matched sparse execution. This is not an otherwise-identical trained function.',
            primary_causal_criterion='Normal-write initial-publication content shuffling versus intact, combined with exact-task advantage over trained fresh recurrence.',
            held_diagnostics='Holding all publications also removes legitimate fast query-time computation; held contrasts are diagnostic rather than a required support criterion.',
            old_mass=f'Full-query probability mass on publications older than {cfg["context"]} bytes; provenance, not a causal usefulness score.',
            bootstrap='First four generic owner commits per stream are forced, excluded from learned gate credit and reported separately.',
            time='Evaluation and audit wall times include their diagnostic forks; live execution counts exclude those extra forwards.'),
        arms={arm: {k: v for k, v in run.items() if k != 'audit_samples'} for arm, run in study.items()},
        historical_basic_stream_query_audit=historical)
    warm_compare = {arm: differences(run, warm['fresh']) for arm, run in warm.items() if arm != 'fresh'} if warm else {}
    warm_supported = directional_support(warm_compare)
    if warm:
        summary.update(scope='Separate scratch and common-parent continuation cohorts; one paired retention seed each',
            continuation=dict(initialization=warm['competitive']['invocation']['initialization'],
                all_parameters_trainable=True, optimizer_and_state_reset=True,
                inherited_parent_budget=parent_training_budget(warm),
                comparisons=warm_compare, persistent_publication_directional_support=warm_supported,
                delayed_credit_exact_gain=None if 'no_credit' not in warm else dict(
                    selected=group(warm['no_credit']['evaluation']['test'])['nll']-group(warm['competitive']['evaluation']['test'])['nll'],
                    final=group(warm['no_credit']['evaluation']['final_budget']['test'])['nll']-group(warm['competitive']['evaluation']['final_budget']['test'])['nll']),
                arms={arm: {k: v for k, v in run.items() if k != 'audit_samples'} for arm, run in warm.items()}))
    if warm:
        summary['overall_outcome'] = dict(stable_retention_advance=False,
            trained_fresh_selected_exact_nll=group(warm['fresh']['evaluation']['test'])['nll'],
            competitive_selected_exact_nll=group(warm['competitive']['evaluation']['test'])['nll'],
            interpretation='The continuation must beat its separately trained fresh control to establish the intended retention advantage; small content-intervention effects alone are insufficient.')
    destination = output_root/'experiments/text_regions/results/task_retention'
    destination.mkdir(parents=True, exist_ok=True)
    hashes, manifest = {}, {}
    cohorts = [(Path(), runs, study)]
    if warm:
        cohorts.append((Path('continuation'), continuation, warm))
    for prefix, source_root, cohort in cohorts:
        for arm, run in cohort.items():
            for name in EVIDENCE:
                src, dest = source_root/arm/name, destination/prefix/arm/name
                digest = hashlib.sha256(src.read_bytes()).hexdigest()
                is_audit_trace = name == 'query_audit/queries.jsonl'
                oversized_trace = name.endswith('_traces.jsonl') and src.stat().st_size > MAX_REPO_TRACE_BYTES
                copied = not is_audit_trace and not oversized_trace
                key = (prefix/arm/name).as_posix()
                manifest[key] = dict(sha256=digest, bytes=src.stat().st_size,
                    records=run['raw_trace_counts'].get(name), repository_copy=copied,
                    full_source='downloadable artifacts archive')
                if copied:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    hashes[key] = digest
            sample = destination/prefix/arm/'query_audit/query_samples.jsonl'
            sample.parent.mkdir(parents=True, exist_ok=True)
            sample.write_text(''.join(json.dumps(r)+'\n' for r in run['audit_samples']))
            sample_key = (prefix/arm/'query_audit/query_samples.jsonl').as_posix()
            hashes[sample_key] = hashlib.sha256(sample.read_bytes()).hexdigest()
            full_key = (prefix/arm/'query_audit/queries.jsonl').as_posix()
            manifest[full_key]['repository_sample'] = sample_key
            manifest[full_key]['sample_records'] = len(run['audit_samples'])
    shutil.copy2(baseline_path, destination/'historical_query_audit.json')
    hashes['historical_query_audit.json'] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    for seed in historical['seeds']:
        source = baseline/str(seed)
        target = destination/'historical_query_audit'/str(seed)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/'summary.json', target/'summary.json')
        hashes[f'historical_query_audit/{seed}/summary.json'] = hashlib.sha256((target/'summary.json').read_bytes()).hexdigest()
        records = read_jsonl(source/'queries.jsonl')
        sample = sorted(records, key=lambda r: (r['document'], r['query_start']))[:4]
        (target/'query_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sample))
        key = f'historical_query_audit/{seed}/queries.jsonl'
        manifest[key] = dict(sha256=hashlib.sha256((source/'queries.jsonl').read_bytes()).hexdigest(),
            bytes=(source/'queries.jsonl').stat().st_size, records=len(records), repository_copy=False,
            repository_sample=f'historical_query_audit/{seed}/query_samples.jsonl', sample_records=len(sample),
            full_source='downloadable artifacts archive')
        hashes[f'historical_query_audit/{seed}/query_samples.jsonl'] = hashlib.sha256((target/'query_samples.jsonl').read_bytes()).hexdigest()
    for filename, entry in validation['logs'].items():
        target = destination/'validation'/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runs.parent/filename, target)
        hashes['validation/'+filename] = entry['sha256']
    summary['evidence_sha256'], summary['evidence_manifest'] = hashes, manifest
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    verdict = ('Small direction-only signs favor ' + ', '.join(LABELS[a] for a in supported) + '; a stable retention advance remains unestablished.'
               if supported else 'Persistent publications do not yet show combined exact-task advantage over trained fresh recurrence and normal-write initial-content sensitivity.')
    lines = ['# SHoEbox: persistent-publication retention', '',
        f'November 6, 2026. Source base `{SOURCE_BASE}`; paired initialization/data seeds {cfg["seed"]}/{cfg["data_seed"]}. Scratch cohort: {verdict}', '',
        '## Changes and comparison', '',
        'The write policy can condition on the current owned publication, validity, age and candidate similarity. Competitive writes '
        'choose one owner or none; cross-owner delayed derivatives follow that competition. Generic startup fills each of four invalid '
        'owners once. These forced commits receive no learned gate credit. Common initialization is preserved; conditioned arms have '
        f'{cfg["model_parameters"]:,} parameters, and legacy has 864 fewer.', '',
        'Four recurring entities receive changing values, questions, corrections and updates irrelevant to the current question but useful '
        'later. Delays are 2/48 bytes; longer-delay evaluation triples them. Inputs contain no answers, slot IDs or task labels. '
        'The fresh arm is trained separately to read current expert candidates, while retaining producer/composer recurrence and the same sparse execution budget.', '',
        '## Scratch cohort: selected and final tasks', '',
        table(['Arm', 'Selected step', 'Exact NLL', 'Exact accuracy', 'Final exact NLL', 'Long exact NLL'], [
            [LABELS[arm], run['evaluation']['selected_step'], number(group(run['evaluation']['test'])['nll']),
             pct(group(run['evaluation']['test'])['accuracy']), number(group(run['evaluation']['final_budget']['test'])['nll']),
             number(group(run['evaluation']['long_delay'])['nll'])] for arm, run in study.items()]), '',
        'NLL is nats per scored answer byte. Checkpoints minimize validation all-answer NLL; exact-value NLL is the declared retention '
        'comparison. Semantic/context scores and both complete checkpoints’ evaluations are in JSON. Positive gains below favor stored publications.', '',
        table(['Stored arm', 'Selected exact gain vs fresh', 'Final exact gain vs fresh'], [
            [LABELS[arm], number(d['selected_exact_gain_vs_trained_fresh'], 6), number(d['final_exact_gain_vs_trained_fresh'], 6)]
            for arm, d in compare.items()]), '',
        '## Whole-query causal checks', '',
        table(['Arm', 'Normal-write shuffle delta', 'Held-content gain', 'Held-shuffle delta'], [
            [LABELS[arm], number(d['normal_write_content_shuffle_exact_delta'], 6), number(d['held_publication_exact_gain']),
             number(d['held_content_shuffle_exact_delta'])] for arm, d in compare.items()]), '',
        'Normal-write shuffle replaces pre-query publication content while allowing ordinary query computation; positive exact-NLL delta '
        'supports content use. Held-content gain is hold+blocked minus hold; held-shuffle delta is hold+shuffled minus hold. Holding '
        'every publication also removes fast query-time computation, so these are separate diagnostics, not mandatory proof. Forks '
        'preserve earlier recurrent state and isolate direct pre-query content use, not all historical information in recurrence. ', '']
    older = aged_evidence(study['competitive'])
    if older.get('changed_queries', 0):
        older_verdict = ('Older-publication usefulness remains unmet.' if older['status'] == 'unmet' else
                         'The affected-query contrast provides one-seed directional evidence for older-content use.')
        lines += [f'Competitive age-filtered shuffling changed {older["changed_slots"]} older slots across '
            f'{older["changed_queries"]} exact queries. Exact NLL changed by {number(older["all_query_exact_nll_delta"], 6)} '
            f'across all queries and {number(older["paired_changed_query_exact_nll_delta"], 6)} on the matched changed-query subset. '
            f'{older_verdict} The intact read mass on changed older versions was {pct(older["intact_changed_read_mass"])}; '
            'their age and consumption did not by themselves establish usefulness.', '']
    h = historical['groups']['exact']
    lines += [f'The earlier three-seed basic-stream audit had exact NLL {number(h["intact"]["nll"])} intact, '
              f'{number(h["blocked"]["nll"])} query-blocked, {number(h["hold"]["nll"])} held, and '
              f'{number(h["hold_blocked"]["nll"])} held+blocked. It motivated separating continuing recurrent computation from '
              'useful held publications; it is a different stream, not a paired quality baseline.', '',
              '## Retention behavior and cost', '',
              table(['Arm', 'Forced commits', 'Learned commits/1k bytes', 'Query age', 'Old-query mass', 'Revised exact'], [
                  [LABELS[arm], run['evaluation']['test']['bootstrap_events'],
                   number(commits_per_thousand(run['evaluation']['test']), 1),
                   number(audit_group(run['audit'])['intact'].get('consumed_read_age'), 1),
                   pct(run['audit_derived']['exact']['mean_old_publication_read_mass']),
                   pct(run['derived']['test']['corrected_exact_accuracy'])] for arm, run in study.items()]), '',
              'Commit rate excludes startup events and counts all owners. Query age is read-mass-weighted bytes since publication over '
              f'exact query windows; old mass counts publications older than {cfg["context"]} bytes. Fact-to-query delay is separate. '
              'Lower writes or older occupied state alone do not establish useful retention.', '']
    lag = study['competitive']['derived']['test']['correction_lag']
    lines += [f'Competitive corrections resolve {lag["resolved"]}/{lag["versions"]} queried revised versions; '
              f'{lag["unresolved"]} remain unresolved. Among resolved versions, first success averages '
              f'{number(lag["mean_first_correct_query"], 2)} query opportunities and '
              f'{number(lag["mean_first_correct_byte_lag"], 1)} bytes. These are sampled opportunities, not continuous recovery timing.', '',
              f'Each arm trains {cfg["steps"]} updates, {study["competitive"]["history"][-1]["training_external_bytes"]:,} bytes and '
              f'{study["competitive"]["history"][-1]["training_expert_executions"]:,} sparse expert calls; batches/chunks are '
              f'{cfg["batch"]}/{cfg["context"]}. Training seconds (legacy/conditioned/competitive/no-credit/fresh): ' +
              '/'.join(number(study[a]['history'][-1]['elapsed_seconds'], 1) for a in ARMS) + '.', '',
              f'Competitive state/eligibility storage is {study["competitive"]["evaluation"]["test"]["state_bytes"]:,}/'
              f'{study["competitive"]["history"][-1]["trace_bytes"]:,} bytes for evaluation/training batches. '
              f'Evaluation/audit took {number(study["competitive"]["evaluation"]["test"]["seconds"], 2)}/'
              f'{number(study["competitive"]["audit"]["seconds"], 2)} seconds including diagnostic forwards; live execution counts '
              'exclude those extra forwards. Selected budgets and owner utilization are in JSON. Full raw query traces ship in the attached archive; '
              'the repository contains summaries and four deterministic query samples per arm.', '',
              '## Scratch outcome', '',
              verdict + f' Competitive delayed-credit exact NLL gain over credit-disabled training is '
              f'{number(credit_gain["selected_exact_nll"])} selected and {number(credit_gain["final_exact_nll"])} final. '
              + ('The selected exact-loss credit criterion is met in this pilot. ' if credit_gain['selected_exact_nll'] > 0 else
                 'The selected exact-loss credit criterion remains unmet. ') +
              'Three-seed retention replication and 24.9M scaling remain open. No age or null-mass threshold is substituted for task usefulness.', '']
    if warm:
        warm_verdict = ('Directional support: ' + ', '.join(LABELS[a] for a in warm_supported) + '.' if warm_supported else
                        'The continuation does not meet the combined trained-fresh advantage and normal-write content-sensitivity criterion.')
        initial = warm['competitive']['invocation']['initialization']
        inherited = summary['continuation']['inherited_parent_budget']
        lines += ['## Separate common-parent continuation', '',
            f'All continuation arms start from the same previously trained seed-17 checkpoint (parent step {initial["parent_step"]}; '
            f'SHA256 `{initial["sha256"][:12]}`). All weights remain trainable; optimizer, runtime and stream state restart. '
            f'Each inherits {inherited["external_bytes"]:,} earlier training bytes and receives {study["competitive"]["history"][-1]["training_external_bytes"]:,} new '
            f'retention bytes ({cfg["steps"]} updates). Results remain separate from scratch initialization.', '',
            table(['Arm', 'Selected step', 'Exact NLL', 'Final exact NLL', 'Gain vs fresh', 'Shuffle delta'], [
                [LABELS[arm], run['evaluation']['selected_step'], number(group(run['evaluation']['test'])['nll']),
                 number(group(run['evaluation']['final_budget']['test'])['nll']),
                 number(warm_compare[arm]['selected_exact_gain_vs_trained_fresh'], 6) if arm != 'fresh' else 'reference',
                 number(warm_compare[arm]['normal_write_content_shuffle_exact_delta'], 6) if arm != 'fresh' else 'n/a']
                for arm, run in warm.items()]), '', warm_verdict + ' Full selected/final, age-filtered shuffle diagnostics when available, '
            'read ages, correction and resource metrics remain separate in JSON. Three-seed confirmation remains open.', '']
    if warm:
        aged = aged_evidence(warm['competitive'])
        if aged.get('changed_queries', 0):
            lines += [f'Competitive continuation age-filtered shuffling changes {aged["changed_slots"]} old slots across '
                f'{aged["changed_queries"]} exact questions; paired changed-query NLL delta is '
                f'{number(aged["paired_changed_query_exact_nll_delta"], 6)}. This causal effect is reported separately from '
                'whether stored publications improve performance over the trained fresh control.', '']
    if warm:
        competitive_loss = group(warm['competitive']['evaluation']['test'])['nll']
        fresh_loss = group(warm['fresh']['evaluation']['test'])['nll']
        main_result = ('Trained fresh recurrence performs better on exact questions: '
                       f'{number(fresh_loss, 6)} NLL versus {number(competitive_loss, 6)} competitive publications. '
                       'The intended retention advantage is not established.' if fresh_loss <= competitive_loss else
                       'The competitive continuation improves exact-task loss; the full paired causal results follow below.')
        lines[2] = (f'November 6, 2026. Source base `{SOURCE_BASE}`; seeds {cfg["seed"]}/{cfg["data_seed"]}. '
                    + main_result + ' Scratch and common-parent continuation cohorts are reported separately.')
    lines += ['## Validation and next target', '',
        f'All five regression suites pass ({validation["full_regression_checks"]} checks); '
        f'{validation["final_focused_checks"]} final focused checks pass. Logs and hashes accompany the evidence.', '',
        'Next, train acquisition and payload representations at short write/read intervals inside an active gradient chunk, then '
        'increase delay and interference. Current detached traces update write-policy heads only; historical payload/feature learning '
        'remains truncated. Keep the trained-fresh, normal-content-shuffle and age-filtered causal controls. Establish a reproducible '
        'task gain before scaling.', '']
    report = output_root/'docs/SHoEbox_Task_Retention_Report.md'  
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines))
    return report, destination/'summary.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT/'artifacts/task_retention/pilot')
    parser.add_argument('--baseline-audits', type=Path, default=ROOT/'artifacts/task_memory/query_audit_baseline')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--continuation', type=Path, help='Optional conditioned/competitive/fresh common-parent cohort; no_credit included when present.')
    args = parser.parse_args()
    report, summary = summarize(args.runs, args.output_root, args.baseline_audits, args.continuation)
    print(json.dumps(dict(report=str(report), summary=str(summary))))


if __name__ == '__main__':
    main()
