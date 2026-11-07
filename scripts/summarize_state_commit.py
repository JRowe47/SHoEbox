"""Validate and summarize every arm of the producer-state factorial study."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('always', 'held', 'always_fresh', 'held_fresh')


def read(path):
    return json.loads(path.read_text())


def changed_contrast(audit, branch):
    details = audit[branch]['groups']['exact']
    changed = details['changed_queries']
    left, right = changed['intact']['nll'], changed[branch]['nll']
    return dict(delta=None if left is None or right is None else right-left,
                changed_queries=details['changed_query_count'],
                changed_cells=details.get('changed_slot_count', details.get('content_changed_slot_count')),
                details=details)


def summarize(root, seeds):
    rows = []
    for seed in seeds:
        matched = []
        for arm in ARMS:
            folder = root / str(seed) / arm
            invocation = read(folder/'invocation.json')
            history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
            last = history[-1]
            evaluation = read(folder/'evaluation.json')
            expected_mode = 'publication' if arm.startswith('held') else 'always'
            expected_read = 'fresh' if arm.endswith('fresh') else 'integrated'
            config = dict(invocation['config'])
            if config.pop('state_commit', 'always') != expected_mode or invocation['arm'] != expected_read:
                raise ValueError(f'Wrong state/read arm: {folder}')
            if invocation['payload_credit'] or config['payload_rank'] != 4:
                raise ValueError(f'Unmatched payload treatment: {folder}')
            selected = min(history, key=lambda x: x['validation']['answer_nll'])['step']
            if selected != evaluation['selected_step']:
                raise ValueError(f'Checkpoint selection mismatch: {folder}')
            if last['step'] != invocation['steps'] or evaluation['final_budget']['step'] != last['step']:
                raise ValueError(f'Incomplete final budget: {folder}')
            stage_coverage = {stage: {k: v for k, v in data.items()
                              if k not in ('producer_candidates', 'producer_commits')}
                              for stage, data in last['stage_totals'].items()}
            training_keys = ('seed', 'data_seed', 'batch', 'context', 'lr', 'text_weight', 'write_cost',
                             'curriculum', 'acquire_steps', 'bridge_steps', 'truncate_every',
                             'write_mode', 'memory_aware_write', 'bootstrap_empty', 'eval_count', 'eval_every')
            match = dict(parameters=invocation['model_parameters'], digest=invocation['data_digest'],
                         config=config, stages=stage_coverage,
                         training={key: invocation[key] for key in training_keys},
                         updates=last['step'], bytes=last['training_external_bytes'],
                         experts=last['training_expert_executions'])
            matched.append(json.dumps(match, sort_keys=True))
            candidates = last['training_external_bytes'] * config['owners']
            commits = candidates if expected_mode == 'always' else sum(
                data['producer_commits'] for data in last['stage_totals'].values())
            row = dict(seed=seed, arm=arm, parameters=invocation['model_parameters'],
                updates=last['step'], bytes=last['training_external_bytes'],
                expert_executions=last['training_expert_executions'],
                selected_step=selected, trace_bytes=last['trace_bytes'],
                producer_candidates=candidates, producer_commits=commits,
                state_commit_fraction=commits/candidates, train_validate_seconds=last['elapsed_seconds'],
                reused_baseline=expected_mode == 'always', evaluation=evaluation, audits={})
            for checkpoint in ('best', 'last'):
                for split in ('test', 'long_delay'):
                    audit = read(folder/f'state_audit_{checkpoint}_{split}'/'summary.json')
                    if audit['live_fork_argmax_changes'] or audit['live_fork_max_logit_error'] > 1e-4:
                        raise ValueError(f'Unreliable query fork: {folder}')
                    ev = evaluation if checkpoint == 'best' else evaluation['final_budget']
                    exact = audit['groups']['exact']['intact']
                    original = ev[split]['groups']['exact']
                    if abs(exact['nll'] - original['nll']) > 1e-5 or exact['accuracy'] != original['accuracy']:
                        raise ValueError(f'Audit does not reproduce saved evaluation: {folder}, {checkpoint}, {split}')
                    row['audits'][f'{checkpoint}_{split}'] = dict(
                        intact=exact, corrected_exact=audit['groups'].get('corrected_exact'),
                        publication_shuffle_delta=audit['groups']['exact']['shuffle']['nll']-exact['nll'],
                        producer_shuffle_delta=audit['groups']['exact']['producer_shuffle']['nll']-exact['nll'],
                        aged_publication=changed_contrast(audit, 'aged_shuffle'),
                        aged_producer=changed_contrast(audit, 'aged_producer_shuffle'),
                        live_fork_max_logit_error=audit['live_fork_max_logit_error'],
                        live_fork_argmax_changes=audit['live_fork_argmax_changes'])
            rows.append(row)
        if len(set(matched)) != 1:
            raise ValueError(f'Unmatched initialization, curriculum, evaluation, or compute budget: seed {seed}')
    contrasts = []
    for seed in seeds:
        pair = {row['arm']: row for row in rows if row['seed'] == seed}
        for checkpoint in ('selected', 'final'):
            for split in ('test', 'long_delay'):
                scores = {}
                for arm in ARMS:
                    evaluation = pair[arm]['evaluation']
                    if checkpoint == 'final':
                        evaluation = evaluation['final_budget']
                    scores[arm] = evaluation[split]['groups']['exact']['nll']
                stored_effect = scores['held'] - scores['always']
                fresh_effect = scores['held_fresh'] - scores['always_fresh']
                contrasts.append(dict(seed=seed, checkpoint=checkpoint, split=split, exact_nll=scores,
                    state_effect_stored=stored_effect, state_effect_fresh=fresh_effect,
                    publication_effect_always=scores['always']-scores['always_fresh'],
                    publication_effect_held=scores['held']-scores['held_fresh'],
                    interaction=stored_effect-fresh_effect))
    means = []
    for checkpoint in ('selected', 'final'):
        for split in ('test', 'long_delay'):
            selected = [r for r in contrasts if r['checkpoint'] == checkpoint and r['split'] == split]
            means.append(dict(checkpoint=checkpoint, split=split,
                exact_nll={arm: statistics.mean(r['exact_nll'][arm] for r in selected) for arm in ARMS},
                **{key: statistics.mean(r[key] for r in selected) for key in
                   ('state_effect_stored', 'state_effect_fresh', 'publication_effect_always',
                    'publication_effect_held', 'interaction')}))
    source = [ROOT/'experiments/text_regions'/f'{name}.py' for name in
              ('task_memory', 'task_state_commit', 'task_query_audit', 'task_memory_train', 'task_curriculum')]
    source += [Path(__file__), ROOT/'scripts/run_state_commit.py']
    return dict(schema=1, source_base='150c8a0', seeds=seeds, rows=rows, contrasts=contrasts, means=means,
        nll_direction='Lower is better. Positive shuffle deltas indicate helpful intact contents.',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[41, 43])
    args = p.parse_args()
    summary = summarize(args.root, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['means'], indent=2))


if __name__ == '__main__':
    main()
