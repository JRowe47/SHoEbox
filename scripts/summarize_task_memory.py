"""Summarize a completed paired task-memory pilot without loading checkpoints.

Run only after all three arms finish --final-eval. Copies JSON evidence into the
repository and writes a compact report; binary checkpoints remain in artifacts.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('integrated', 'no_credit', 'no_memory')
SELECTIVE_ARMS = ('selective', 'selective_no_credit')
PAIRED_FIELDS = ('seed', 'data_seed', 'data_digest', 'batch', 'context', 'steps', 'width', 'owners',
                 'lr', 'text_weight', 'config', 'data_config', 'model_parameters', 'eval_count')
LABELS = {'integrated': 'Integrated', 'no_credit': 'No delayed credit', 'no_memory': 'No memory'}
SOURCE_VERSION = '7707348b550c64e2d685f07c8e67c058cb169de5'
EVIDENCE = ('invocation.json', 'history.jsonl', 'evaluation.json', 'test_traces.jsonl',
            'long_delay_traces.jsonl', 'final_test_traces.jsonl', 'final_long_delay_traces.jsonl')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def number(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def pct(value):
    return 'n/a' if value is None else f'{100*value:.1f}%'


def accuracy(records):
    return mean(r['prediction'] == r['answer'] for r in records)


def trace_metrics(records):
    by_kind = defaultdict(list)
    revisions = defaultdict(list)
    for r in records:
        by_kind[r['kind']].append(r)
        if r['correction'] and r['kind'] == 'exact':
            identity = (r['document'], r['entity'], r['attribute'], r['source_position'], r['source_version'])
            revisions[identity].append(r)
    lag = []
    for identity, rows in sorted(revisions.items()):
        rows.sort(key=lambda r: r['position'])
        successful = next(((i, r) for i, r in enumerate(rows, 1) if r['prediction'] == r['answer']), None)
        lag.append(dict(document=identity[0], entity=identity[1], attribute=identity[2],
                        source_position=identity[3], source_version=identity[4],
                        query_opportunities=len(rows), status='resolved' if successful else 'unresolved',
                        first_correct_query=successful[0] if successful else None,
                        first_correct_byte_lag=successful[1]['position']-identity[3] if successful else None))
    corrected = [r for r in records if r['correction'] and r['kind'] == 'exact']
    memory = [r for r in records if r['kind'] != 'context']
    return dict(null_mass_by_kind={kind: mean(r['read_weights'][-1] for r in rows)
                                   for kind, rows in by_kind.items()},
                null_mass_memory=mean(r['read_weights'][-1] for r in memory),
                accuracy_by_kind={kind: accuracy(rows) for kind, rows in by_kind.items()},
                count_by_kind={kind: len(rows) for kind, rows in by_kind.items()},
                corrected_exact_accuracy=accuracy(corrected), corrected_exact_count=len(corrected),
                corrected_exact_blocked_nll_delta=mean(r['blocked_nll']-r['nll'] for r in corrected if 'blocked_nll' in r),
                correction_lag=dict(versions=len(lag), resolved=sum(r['status']=='resolved' for r in lag),
                    unresolved=sum(r['status']=='unresolved' for r in lag),
                    mean_first_correct_query=mean(r['first_correct_query'] for r in lag if r['status']=='resolved'),
                    mean_first_correct_byte_lag=mean(r['first_correct_byte_lag'] for r in lag if r['status']=='resolved'),
                    records=lag))


def selected_group(evaluation, group, metric='accuracy'):
    return evaluation['groups'].get(group, {}).get(metric)


def intervention_delta(evaluation, kind='blocked'):
    item = evaluation.get('interventions', {}).get(kind)
    return None if item is None else item['nll']-evaluation['answer_nll']


def positive(value):
    return value is not None and value > 0


def causal_examples(records, limit=2):
    """First two document/entity revisions, without selecting by correctness."""
    examples, seen = [], set()
    for revision in sorted(records, key=lambda r: (r['document'], r['position'])):
        if not revision['correction'] or revision['kind'] != 'exact':
            continue
        identity = (revision['document'], revision['entity'], revision['attribute'], revision['source_version'])
        if identity in seen:
            continue
        seen.add(identity)
        initial = [r for r in records if r['document'] == revision['document'] and r['entity'] == revision['entity']
                   and r['attribute'] == revision['attribute'] and r['kind'] == 'exact'
                   and r['source_version'] < revision['source_version']]
        if not initial:
            continue
        examples.append(dict(initial=min(initial, key=lambda r: r['position']), revised=revision))
        if len(examples) == limit:
            break
    return examples


def byte_name(value):
    return repr(chr(value)) if 32 <= value < 127 else f'byte {value}'


def load_runs(runs, arms=ARMS, expected_cost=0., expected_modes=None):
    loaded = {}
    # Validate every required input before producing any report/copy.
    for arm in arms:
        folder = runs/arm
        for filename in EVIDENCE:
            if not (folder/filename).is_file():
                raise FileNotFoundError(f'Incomplete pilot: {folder/filename}; finish --final-eval first.')
        invocation = json.loads((folder/'invocation.json').read_text())
        expected_mode = (expected_modes or {}).get(arm, arm)
        if invocation['arm'] != expected_mode:
            raise ValueError(f'{arm}: invocation arm does not match expected mode {expected_mode}.')
        if invocation.get('write_cost', 0.) != expected_cost:
            raise ValueError(f'{arm}: expected write_cost={expected_cost}.')
        evaluation = json.loads((folder/'evaluation.json').read_text())
        history = read_jsonl(folder/'history.jsonl')
        if not history or history[-1]['step'] != invocation['steps']:
            raise ValueError(f'{arm}: history has not reached the invocation budget.')
        if evaluation['final_budget']['step'] != invocation['steps']:
            raise ValueError(f'{arm}: final checkpoint evaluation does not match budget.')
        selected = next((h for h in reversed(history) if h['step'] == evaluation['selected_step']), None)
        if selected is None:
            raise ValueError(f'{arm}: selected checkpoint has no history record.')
        traces = {split: read_jsonl(folder/(split+'_traces.jsonl'))
                  for split in ('test', 'long_delay', 'final_test', 'final_long_delay')}
        for name, rows in traces.items():
            split = name.removeprefix('final_')
            evaluated = evaluation['final_budget'][split] if name.startswith('final_') else evaluation[split]
            if len(rows) != evaluated['answers']:
                raise ValueError(f'{arm}/{name}: trace count differs from evaluation.')
        loaded[arm] = dict(invocation=invocation, evaluation=evaluation, history=history,
                           selected_history=selected, traces=traces,
                           derived={split: trace_metrics(rows) for split, rows in traces.items()})
    reference = loaded[arms[0]]
    for arm in arms[1:]:
        assert_paired(loaded[arm], reference, arm)
    return loaded


def assert_paired(run, reference, label):
    for name in PAIRED_FIELDS:
        if run['invocation'][name] != reference['invocation'][name]:
            raise ValueError(f'Unpaired pilot: {label} differs in {name}.')
    for name in ('training_external_bytes', 'training_expert_executions'):
        if run['history'][-1][name] != reference['history'][-1][name]:
            raise ValueError(f'Unpaired final compute: {label} differs in {name}.')
    for split in ('test', 'long_delay'):
        for name in ('answers', 'external_bytes', 'executed_bytes', 'expert_executions'):
            if run['evaluation'][split][name] != reference['evaluation'][split][name]:
                raise ValueError(f'Unpaired evaluation: {label}/{split} differs in {name}.')


def owner_diagnostics(evaluation):
    owners = evaluation['owners']
    fractions = [o['write_fraction'] for o in owners.values()]
    return dict(owners=owners, write_fraction_range=[min(fractions), max(fractions)],
                occupied_owners=sum(o['valid_fraction'] > 0 for o in owners.values()),
                query_mean_weighted_publication_age=evaluation['query_mean_weighted_publication_age'])


def load_selective(runs, primary):
    loaded = load_runs(runs, SELECTIVE_ARMS, expected_cost=.02,
                       expected_modes={'selective': 'integrated', 'selective_no_credit': 'no_credit'})
    for arm, run in loaded.items():
        assert_paired(run, primary['integrated'], arm)
        for split in ('test', 'long_delay'):
            for evaluation in (run['evaluation'][split], run['evaluation']['final_budget'][split]):
                owner_diagnostics(evaluation)  # Require the new provenance diagnostics.
    trigger = next((r for r in primary['integrated']['history'] if r['step'] == 100), None)
    if trigger is None:
        raise ValueError('Write-cost followup requires the primary 100-update diagnostic.')
    trigger_rate = trigger['validation']['write_rate']
    if trigger_rate < .95:
        raise ValueError('The declared writing-collapse trigger is not present in this primary run.')
    return loaded, dict(step=100, validation_write_rate=trigger_rate,
                        reason='Observed near-universal writing triggered the separate write-cost followup.')


def load_replications(path, primary):
    """Two additional seed/data pairs with the same finite training protocol."""
    reference = primary['integrated']
    base_seed, base_data = reference['invocation']['seed'], reference['invocation']['data_seed']
    studies = {str(base_seed): primary}
    for offset in (1, 2):
        seed = base_seed + offset
        study = load_runs(path/str(seed))
        run = study['integrated']
        if run['invocation']['seed'] != seed or run['invocation']['data_seed'] != base_data + offset:
            raise ValueError(f'Replication {seed}: unexpected model/data seed pair.')
        for key in PAIRED_FIELDS:
            if key not in ('seed', 'data_seed', 'data_digest') and run['invocation'][key] != reference['invocation'][key]:
                raise ValueError(f'Replication {seed}: training protocol differs in {key}.')
        for key in ('training_external_bytes', 'training_expert_executions'):
            if run['history'][-1][key] != reference['history'][-1][key]:
                raise ValueError(f'Replication {seed}: final compute differs in {key}.')
        studies[str(seed)] = study
    if len({study['integrated']['invocation']['data_digest'] for study in studies.values()}) != 3:
        raise ValueError('Replications must use three distinct paired stream datasets.')
    return studies


def comparisons(study):
    a = study['integrated']['evaluation']
    b = study['no_credit']['evaluation']
    c = study['no_memory']['evaluation']
    d = study['integrated']['derived']['test']
    null_context, null_memory = d['null_mass_by_kind'].get('context'), d['null_mass_memory']
    correction = d['corrected_exact_accuracy']
    correction_control = study['no_memory']['derived']['test']['corrected_exact_accuracy']
    return dict(selected_memory_nll_gain=c['test']['answer_nll']-a['test']['answer_nll'],
                selected_delayed_credit_nll_gain=b['test']['answer_nll']-a['test']['answer_nll'],
                final_memory_nll_gain=c['final_budget']['test']['answer_nll']-a['final_budget']['test']['answer_nll'],
                final_delayed_credit_nll_gain=b['final_budget']['test']['answer_nll']-a['final_budget']['test']['answer_nll'],
                blocked_nll_delta=intervention_delta(a['test']),
                shuffled_nll_delta=intervention_delta(a['test'], 'shuffle'),
                stale_nll_delta=intervention_delta(a['test'], 'stale'),
                long_delay_memory_nll_gain=c['long_delay']['answer_nll']-a['long_delay']['answer_nll'],
                context_null_gap=None if null_context is None or null_memory is None else null_context-null_memory,
                corrected_accuracy_gain=None if correction is None or correction_control is None else correction-correction_control,
                corrected_blocked_nll_delta=d['corrected_exact_blocked_nll_delta'])


def directional_exits(gains, three_seeds=False):
    return dict(useful_retained_experience=positive(gains['selected_memory_nll_gain']) and positive(gains['blocked_nll_delta']),
                context_selectivity=None,
                revision_response=positive(gains['corrected_accuracy_gain']) and positive(gains['corrected_blocked_nll_delta']),
                delayed_credit_value=positive(gains['selected_delayed_credit_nll_gain']),
                longer_delay_memory_value=positive(gains['long_delay_memory_nll_gain']),
                three_seed_confirmation=three_seeds)


def aggregate_studies(studies):
    per_seed = []
    for seed, study in studies.items():
        gains = comparisons(study)
        per_seed.append(dict(seed=int(seed), data_seed=study['integrated']['invocation']['data_seed'],
                             gains=gains, directional_exits=directional_exits(gains)))
    mean_gains = {key: mean(row['gains'][key] for row in per_seed if row['gains'][key] is not None)
                  for key in per_seed[0]['gains']}
    aggregate = {}
    for arm in ARMS:
        rows = [study[arm] for study in studies.values()]
        aggregate[arm] = dict(
            answer_nll=mean(r['evaluation']['test']['answer_nll'] for r in rows),
            accuracy=mean(r['evaluation']['test']['accuracy'] for r in rows),
            exact_accuracy=mean(selected_group(r['evaluation']['test'], 'exact') for r in rows),
            category_accuracy=mean(selected_group(r['evaluation']['test'], 'semantic') for r in rows
                                   if selected_group(r['evaluation']['test'], 'semantic') is not None),
            corrected_exact_accuracy=mean(r['derived']['test']['corrected_exact_accuracy'] for r in rows),
            long_delay_nll=mean(r['evaluation']['long_delay']['answer_nll'] for r in rows),
            long_delay_accuracy=mean(r['evaluation']['long_delay']['accuracy'] for r in rows),
            long_delay_exact_accuracy=mean(selected_group(r['evaluation']['long_delay'], 'exact') for r in rows),
            final_answer_nll=mean(r['evaluation']['final_budget']['test']['answer_nll'] for r in rows),
            final_accuracy=mean(r['evaluation']['final_budget']['test']['accuracy'] for r in rows))
    return dict(seeds=[row['seed'] for row in per_seed], data_seeds=[row['data_seed'] for row in per_seed],
                aggregation='Unweighted mean across paired seeds; finite-budget directional comparisons.',
                arms=aggregate, paired_gains=per_seed, mean_gains=mean_gains,
                positive_seed_counts={key: sum(positive(row['gains'][key]) for row in per_seed) for key in mean_gains},
                directional_exits=directional_exits(mean_gains, three_seeds=True),
                exit_support_counts={key: sum(row['directional_exits'][key] is True for row in per_seed)
                                     for key in per_seed[0]['directional_exits'] if key != 'three_seed_confirmation'},
                completion_note='three_seed_confirmation records completion; it does not mean every paired result is positive.')


def summarize(runs, output_root, selective_runs=None, replications=None):
    loaded = load_runs(runs)
    studies = load_replications(replications, loaded) if replications else None
    aggregate = aggregate_studies(studies) if studies else None
    supplement, trigger = load_selective(selective_runs, loaded) if selective_runs else ({}, None)
    integrated = loaded['integrated']
    cfg = integrated['invocation']
    selected = {arm: run['evaluation']['test'] for arm, run in loaded.items()}
    final = {arm: run['evaluation']['final_budget']['test'] for arm, run in loaded.items()}
    d = integrated['derived']['test']
    blocked = intervention_delta(selected['integrated'])
    gains = comparisons(loaded)
    exits = directional_exits(gains)
    summary = dict(schema=1, source_version=SOURCE_VERSION, scope='one-seed paired pilot',
                   units=dict(nll='nats per scored target byte', accuracy='fraction', time='seconds',
                              state_bytes='bytes for entire evaluation batch', trace_bytes='bytes for training batch',
                              evaluation_seconds='wall time including counterfactual intervention forwards; live execution counts exclude diagnostics'),
                   paired_fields={key: cfg[key] for key in ('seed', 'data_seed', 'data_digest', 'batch', 'context', 'steps')},
                   comparisons=gains, directional_pilot_exits=exits,
                   exit_rule_note='Directional pilot comparisons, not significance tests; no universal recall floor.',
                   context_selectivity_status='Unestablished: terminal-token null rates do not measure retrieval throughout a query prefix.',
                   publication_content_sensitivity='Positive matched shuffle NLL delta measures dependence on publication content, separately from query-window selectivity.',
                   arms={arm: {key: value for key, value in run.items() if key != 'traces'} for arm, run in loaded.items()},
                   causal_examples=causal_examples(integrated['traces']['test']))
    if aggregate:
        summary.update(scope='three-seed paired study', seed17_comparisons=gains,
            seed17_pilot_exits=exits, comparisons=aggregate['mean_gains'],
            directional_pilot_exits=aggregate['directional_exits'], three_seed=aggregate,
            exit_rule_note='Unweighted seed means with paired support counts; finite-budget comparisons, not significance tests.',
            replications={seed: {arm: {k: v for k, v in run.items() if k != 'traces'} for arm, run in study.items()}
                          for seed, study in studies.items() if int(seed) != cfg['seed']})
    if supplement:
        a, b = supplement['selective'], supplement['selective_no_credit']
        summary['supplementary_write_cost'] = dict(write_cost=.02, primary_write_cost=0., trigger=trigger,
            comparisons=dict(selected_credit_nll_gain=b['evaluation']['test']['answer_nll']-a['evaluation']['test']['answer_nll'],
                final_credit_nll_gain=b['evaluation']['final_budget']['test']['answer_nll']-a['evaluation']['final_budget']['test']['answer_nll'],
                selected_nll_gain_over_primary=selected['integrated']['answer_nll']-a['evaluation']['test']['answer_nll']),
            arms={arm: {key: value for key, value in run.items() if key != 'traces'} for arm, run in supplement.items()},
            owner_diagnostics={arm: dict(selected=owner_diagnostics(run['evaluation']['test']),
                final=owner_diagnostics(run['evaluation']['final_budget']['test'])) for arm, run in supplement.items()})
    output = output_root/'experiments/text_regions/results/task_memory' 
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for arm in ARMS:
        (output/arm).mkdir(exist_ok=True)
        for filename in EVIDENCE:
            src, dest = runs/arm/filename, output/arm/filename
            shutil.copy2(src, dest)
            hashes[f'{arm}/{filename}'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if studies:
        for seed in studies:
            if int(seed) == cfg['seed']:
                continue
            for arm in ARMS:
                destination = output/'replications'/seed/arm
                destination.mkdir(parents=True, exist_ok=True)
                for filename in EVIDENCE:
                    src, dest = replications/seed/arm/filename, destination/filename
                    shutil.copy2(src, dest)
                    hashes[f'replications/{seed}/{arm}/{filename}'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if supplement:
        for arm in SELECTIVE_ARMS:
            destination = output/'supplementary'/arm
            destination.mkdir(parents=True, exist_ok=True)
            for filename in EVIDENCE:
                src, dest = selective_runs/arm/filename, destination/filename
                shutil.copy2(src, dest)
                hashes[f'supplementary/{arm}/{filename}'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    summary['evidence_sha256'] = hashes
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    main_gains = aggregate['mean_gains'] if aggregate else gains
    scope = ('Three paired seeds (' + ', '.join(map(str, aggregate['seeds'])) + ')' if aggregate
             else f'One paired seed ({cfg["seed"]})')
    result = ('The integrated arm improves mean selected held-out task NLL versus both training controls.'
              if positive(main_gains['selected_memory_nll_gain']) and positive(main_gains['selected_delayed_credit_nll_gain'])
              else 'The integrated loop is implemented; it does not beat both training controls on mean selected held-out task NLL.')
    lines = ['# SHoEbox: task-driven memory study', '',
             f'November 6, 2026. Source base `{SOURCE_VERSION[:7]}`. {scope}, '
             f'{cfg["model_parameters"]:,} parameters, {cfg["owners"]} owned publications, '
             f'{cfg["context"]}-byte gradient chunks. {result}', '',
             '## Integrated behavior', '',
             'Ordinary bytes drive recurrent SHoE producers with separately owned state. A dense soft read over current publications plus '
             'a null option feeds the composer’s existing sparse routing and experts. Reads precede committed writes. The stream tests '
             'conflicting values, distractors, corrections and context-only questions; answers and memory-slot IDs never enter model input.', '',
             'Task loss trains payloads, read heads and gates within chunks. Bounded conditional straight-through sensitivities carry later '
             'publication gradients across detach boundaries to the write-policy head; historical candidate/feature gradients remain '
             'truncated. Read heads learn at retrieval events. Replacement removes old responsibility; holds decay it by 0.995. '
             'This is not sampled policy-gradient estimation. Matched read-blocked forwards measure causal gain separately.', '',
             '## Seed 17 pilot detail' if aggregate else '## Paired held-out results', '',
             table(['Arm', 'NLL', 'Exact', 'Category', 'Revised exact', 'Long exact', 'Blocked delta'], [
                 [LABELS[arm], number(selected[arm]['answer_nll']), pct(selected_group(selected[arm], 'exact')),
                  pct(selected_group(selected[arm], 'semantic')), pct(loaded[arm]['derived']['test']['corrected_exact_accuracy']),
                  pct(selected_group(loaded[arm]['evaluation']['long_delay'], 'exact')), number(intervention_delta(selected[arm]))]
                 for arm in ARMS]), '',
             f'NLL is nats per answer byte; accuracies use their respective query subsets. Each test arm scores '
             f'{selected["integrated"]["answers"]} answers from {cfg["eval_count"]*2} fresh independent streams. '
             'Positive blocked delta means memory helps. Interventions fork identical pre-query state at matched expert count. '
             'Long-delay evaluation multiplies declared delays by '
             f'{cfg["data_config"]["long_delay_multiplier"]}; new entity/attribute combinations are not measured by this pilot.', '',
             f'At the final matched budget, NLL is {number(final["integrated"]["answer_nll"])} integrated, '
             f'{number(final["no_credit"]["answer_nll"])} without delayed credit, and '
             f'{number(final["no_memory"]["answer_nll"])} without memory. Selected checkpoints minimize validation answer NLL. '
             f'Integrated shuffled/stale deltas are {number(gains["shuffled_nll_delta"])}/{number(gains["stale_nll_delta"])} nats. '
             f'Ordinary next-byte NLL is {number(selected["integrated"]["ordinary_nll"])}.', '',
             table(['Arm', 'Steps sel/final', 'Bytes sel/final', 'Expert calls', 'Seconds'], [
                 [LABELS[arm], f'{run["evaluation"]["selected_step"]}/{cfg["steps"]}',
                  f'{run["selected_history"]["training_external_bytes"]:,}/{run["history"][-1]["training_external_bytes"]:,}', f'{run["history"][-1]["training_expert_executions"]:,}',
                  number(run['history'][-1]['elapsed_seconds'], 1)] for arm, run in loaded.items()]), '',
             'Expert counts and time cover the final budget, including periodic validation time. Full curves and selected budgets are in summary.json.', '', '---', '',
             '## Seed 17 selectivity and correction' if aggregate else '## Selectivity and correction', '',
             table(['Arm', 'Terminal null: exact', 'Terminal null: context', 'Writes', 'Read mass'], [
                 [LABELS[arm], pct(run['derived']['test']['null_mass_by_kind'].get('exact')),
                  pct(run['derived']['test']['null_mass_by_kind'].get('context')),
                  pct(selected[arm]['write_rate']), pct(selected[arm]['effective_read_mass'])]
                 for arm, run in loaded.items()]), '',
             'Null mass averages the terminal-token no-read probability; writes count commits per owner-byte. Retrieval can occur during entity/attribute '
             'prefixes, so these terminal rates do not establish query-window relevance selection; detailed context-selective retrieval remains '
             'unestablished here. A positive shuffle penalty separately shows that publication content matters.', '']
    full_blocked = selected['integrated'].get('full_blocked_rollout')
    if full_blocked:
        lines += [f'Seed 17 intact/single-query-blocked/full-blocked accuracy: {pct(selected["integrated"]["accuracy"])}/'
                  f'{pct(selected["integrated"]["interventions"]["blocked"]["accuracy"])}/{pct(full_blocked["accuracy"])}. '
                  f'Query publications average {number(selected["integrated"].get("query_mean_weighted_publication_age"), 2)} bytes old '
                  f'with credit and {number(selected["no_credit"].get("query_mean_weighted_publication_age"), 2)} without: useful recurrent '
                  'summaries are rewritten frequently. Long observation delays do not imply old persistent publications.', '']
    lag = d['correction_lag']
    lines += [f'For integrated exact correction tasks, {lag["resolved"]}/{lag["versions"]} revised fact versions receive a correct '
              f'answer at a measured later query; {lag["unresolved"]} remain unresolved. Among resolved versions, the first success '
              f'occurs at mean query opportunity {number(lag["mean_first_correct_query"], 2)} and '
              f'{number(lag["mean_first_correct_byte_lag"], 1)} bytes after the correction value. This is sampled correction lag: '
              'unobserved recovery times are not inferred.', '',
              '## Causal trace excerpts', '',
              'First two eligible revisions, ordered by document/position; positions are zero-based bytes.', '']
    for example in summary['causal_examples']:
        first, revised = example['initial'], example['revised']
        lines += [f'- Stream {first["document"]}, {first["entity"]}.{first["attribute"]}: observe '
                  f'{byte_name(first["answer"])} at byte {first["source_position"]}; query at {first["position"]} predicts '
                  f'{byte_name(first["prediction"])}. Correct to {byte_name(revised["answer"])} at '
                  f'{revised["source_position"]}; query at {revised["position"]} predicts {byte_name(revised["prediction"])} '
                  f'(reads blocked: {byte_name(revised.get("blocked_prediction", revised["prediction"]))}).']
    if supplement:
        supplementary_labels = {'selective': 'Cost + credit', 'selective_no_credit': 'Cost, no credit'}
        a, b = supplement['selective'], supplement['selective_no_credit']
        af, bf = a['evaluation']['final_budget']['test'], b['evaluation']['final_budget']['test']
        owner_ranges = [owner_diagnostics(run['evaluation']['test'])['write_fraction_range'] for run in (a, b)]
        lines += ['', '## Separate seed 17 write-cost followup', '',
            f'Observed {pct(trigger["validation_write_rate"])} writing at update 100 triggered two additional paired arms with '
            '`0.02 * mean(write_probability)` added to the objective. Seeds, bytes, model and final compute match the primary cost-zero pilot.', '',
            table(['Arm', 'Step', 'NLL', 'Exact', 'Writes', 'Query age'], [
                ['Primary (cost 0)', integrated['evaluation']['selected_step'], number(selected['integrated']['answer_nll']),
                 pct(selected_group(selected['integrated'], 'exact')), pct(selected['integrated']['write_rate']),
                 number(selected['integrated'].get('query_mean_weighted_publication_age'), 1)], *[
                [supplementary_labels[arm], run['evaluation']['selected_step'], number(run['evaluation']['test']['answer_nll']),
                 pct(selected_group(run['evaluation']['test'], 'exact')), pct(run['evaluation']['test']['write_rate']),
                 number(run['evaluation']['test']['query_mean_weighted_publication_age'], 1)]
                for arm, run in supplement.items()]]), '',
            f'Final-budget cost+credit/cost-without-credit NLL is {number(af["answer_nll"])}/{number(bf["answer_nll"])}; '
            f'writes {pct(af["write_rate"])}/{pct(bf["write_rate"])}; '
            f'query ages {number(af["query_mean_weighted_publication_age"], 1)}/{number(bf["query_mean_weighted_publication_age"], 1)} bytes. '
            'Age is read-weighted time since publication, conditional on nonnull reads. '
            f'Selected owner write ranges are {pct(owner_ranges[0][0])}–{pct(owner_ranges[0][1])} and '
            f'{pct(owner_ranges[1][0])}–{pct(owner_ranges[1][1])}; occupancy/read fractions are in JSON. '
            'Lower writing alone does not establish useful selective retention; primary phase exits remain unchanged.']
    lines += ['', '## Phase exits (three-seed means)' if aggregate else '## Phase exits (primary pilot)', '']
    names = {'useful_retained_experience': 'Useful retained experience versus no-memory training and blocked reads',
             'context_selectivity': 'Detailed context-selective retrieval',
             'revision_response': 'Revision response above the no-memory control with positive blocked-read gain',
             'delayed_credit_value': 'Delayed credit improves selected task NLL over credit-disabled training',
             'longer_delay_memory_value': 'Memory improves longer-delay task NLL over no-memory training'}
    for key, name in names.items():
        result_exits = aggregate['directional_exits'] if aggregate else exits
        support = (f" ({aggregate['exit_support_counts'][key]}/3 paired seeds support)"
                   if aggregate and result_exits[key] is not None else '')
        status = 'Unestablished' if result_exits[key] is None else ('Directional evidence' if result_exits[key] else 'Unmet')
        lines.append(f'- {status}{support}: {name}.')
    completion = ('- Complete: three paired seeds; this does not imply three positive results. The 24.9M scale-up remains open.'
                  if aggregate else '- Unmet: three-seed confirmation and the 24.9M scale-up.')
    lines += [completion, '', 
              f'Integrated state occupies {selected["integrated"]["state_bytes"]:,} bytes for the full evaluation batch; '
              f'eligibility traces occupy {integrated["history"][-1]["trace_bytes"]:,} bytes for the training batch. '
              f'Seed 17 evaluation including intervention forwards took {number(selected["integrated"]["seconds"], 2)} seconds for '
              f'{selected["integrated"]["external_bytes"]:,} valid bytes '
              f'({selected["integrated"]["executed_bytes"]:,} live bytes including padding; diagnostic execution counts excluded). '
              'Resumable checkpoints accompany the downloadable artifacts; JSON evidence is committed separately.', '']
    if aggregate:
        means = aggregate['arms']
        details_index = lines.index('## Seed 17 pilot detail')
        main_table = ['## Three-seed held-out means', '',
            table(['Arm', 'NLL', 'Accuracy', 'Exact', 'Long NLL', 'Long exact'], [
                [LABELS[arm], number(means[arm]['answer_nll']), pct(means[arm]['accuracy']),
                 pct(means[arm]['exact_accuracy']), number(means[arm]['long_delay_nll']),
                 pct(means[arm]['long_delay_exact_accuracy'])] for arm in ARMS]), '',
            table(['Seed/data', 'Memory NLL gain', 'Delayed-credit NLL gain', 'Blocked delta'], [
                [f"{r['seed']}/{r['data_seed']}", number(r['gains']['selected_memory_nll_gain']),
                 number(r['gains']['selected_delayed_credit_nll_gain']), number(r['gains']['blocked_nll_delta'])]
                for r in aggregate['paired_gains']] + [
                ['Mean', number(main_gains['selected_memory_nll_gain']), number(main_gains['selected_delayed_credit_nll_gain']),
                 number(main_gains['blocked_nll_delta'])]]), '',
            f"Positive gains favor integrated training. Memory/credit gains are positive in "
            f"{aggregate['positive_seed_counts']['selected_memory_nll_gain']}/3 and "
            f"{aggregate['positive_seed_counts']['selected_delayed_credit_nll_gain']}/3 pairs. "
            f"Final-budget mean NLL (integrated/no-credit/no-memory): {number(means['integrated']['final_answer_nll'])}/"
            f"{number(means['no_credit']['final_answer_nll'])}/{number(means['no_memory']['final_answer_nll'])}. "
            f"Each arm receives {cfg['steps']} updates and {integrated['history'][-1]['training_external_bytes']:,} bytes. "
            'Means weight seeds equally; the JSON preserves selected/final budgets and every per-seed result.', '']
        lines[details_index:details_index] = main_table
    validation_path = output/'validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines += ['## Validation', '',
                  f'All {validation["full_suites_passed"]} regression suites passed ({validation["full_suite_total"]} checks in that run). '
                  f'The final {validation["final_focused_task_checks"]} task-memory checks also pass, including exact optimizer/stream/credit continuation '
                  'and the conditional detached-gradient fixture at 1e-12 tolerance.', '']
    report = output_root/'docs/SHoEbox_Task_Memory_Report.md' 
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines))
    return report, output/'summary.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT/'artifacts/task_memory/pilot_v2')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--selective-runs', type=Path, help='Optional parent of selective and selective_no_credit runs.')
    parser.add_argument('--replications', type=Path, help='Parent of the next two seed directories, each containing three paired arms.')
    args = parser.parse_args()
    report, summary = summarize(args.runs, args.output_root, args.selective_runs, args.replications)
    print(json.dumps(dict(report=str(report), summary=str(summary))))


if __name__ == '__main__':
    main()
