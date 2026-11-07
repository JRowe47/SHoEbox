"""Collect completed paired payload-credit runs without selecting favorable arms."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('gate_only', 'payload', 'fresh')


def read(path):
    return json.loads(path.read_text())


def summarize(root, seeds):
    rows = []
    for seed in seeds:
        matched = []
        for arm in ARMS:
            folder = root / str(seed) / arm
            completion = read(folder / 'completion.json')
            if not completion['completed']:
                raise ValueError(f'Incomplete arm: {folder}')
            invocation = read(folder / 'invocation.json')
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            last = history[-1]
            evaluation = read(folder / 'evaluation.json')
            matched.append((invocation['model_parameters'], invocation['data_digest'],
                            last['step'], last['training_external_bytes'], last['training_expert_executions']))
            if last['step'] != invocation['steps'] or evaluation['final_budget']['step'] != last['step']:
                raise ValueError(f'Final budget mismatch: {folder}')
            row = dict(seed=seed, data_seed=invocation['data_seed'], arm=arm,
                parameters=invocation['model_parameters'], updates=last['step'],
                bytes=last['training_external_bytes'], expert_executions=last['training_expert_executions'],
                train_validate_seconds=last['elapsed_seconds'], complete_seconds=completion['seconds'],
                trace_bytes=last['trace_bytes'], payload_credit_steps=last['payload_credit_steps'],
                payload_gradient_norm_sum=last['payload_gradient_norm_sum'],
                payload_update_norm_sum=last['payload_update_norm_sum'],
                selected_step=evaluation['selected_step'], evaluation=evaluation, audits={})
            if arm != 'fresh':
                for checkpoint in ('best', 'last'):
                    for split in ('test', 'long_delay'):
                        audit = read(folder / f'audit_{checkpoint}_{split}' / 'summary.json')
                        group = audit['groups']['exact']
                        older = audit['aged_shuffle']['groups']['exact']
                        changed = older['changed_queries']
                        a, b = changed['intact']['nll'], changed['aged_shuffle']['nll']
                        row['audits'][f'{checkpoint}_{split}'] = dict(
                            exact_nll=group['intact']['nll'],
                            shuffle_delta=group['shuffle']['nll']-group['intact']['nll'],
                            older_changed_delta=None if a is None or b is None else b-a,
                            older=older, corrected_exact=audit['groups']['corrected_exact'],
                            live_fork_max_logit_error=audit['live_fork_max_logit_error'],
                            live_fork_argmax_changes=audit['live_fork_argmax_changes'])
                        older64 = read(folder / f'audit_{checkpoint}_{split}_age64' / 'summary.json')
                        details = older64['aged_shuffle']['groups']['exact']
                        changed64 = details['changed_queries']
                        left, right = changed64['intact']['nll'], changed64['aged_shuffle']['nll']
                        row['audits'][f'{checkpoint}_{split}']['beyond_window'] = dict(
                            age_threshold=64, older=details,
                            older_changed_delta=None if left is None or right is None else right-left)
            rows.append(row)
        if len(set(matched)) != 1:
            raise ValueError(f'Unmatched initialization/data/compute budgets for seed {seed}.')
    contrasts = []
    for seed in seeds:
        pair = {r['arm']: r for r in rows if r['seed'] == seed}
        for checkpoint in ('selected', 'final'):
            def evaluation(arm):
                value = pair[arm]['evaluation']
                return value if checkpoint == 'selected' else value['final_budget']
            for split in ('test', 'long_delay'):
                scores = {arm: evaluation(arm)[split]['groups']['exact']['nll'] for arm in ARMS}
                contrasts.append(dict(seed=seed, checkpoint=checkpoint, split=split, exact_nll=scores,
                    payload_minus_gate=scores['payload']-scores['gate_only'],
                    payload_minus_fresh=scores['payload']-scores['fresh']))
    means = []
    for checkpoint in ('selected', 'final'):
        for split in ('test', 'long_delay'):
            entries = [c for c in contrasts if c['checkpoint'] == checkpoint and c['split'] == split]
            means.append(dict(checkpoint=checkpoint, split=split,
                exact_nll={arm: statistics.mean(c['exact_nll'][arm] for c in entries) for arm in ARMS},
                payload_minus_gate=statistics.mean(c['payload_minus_gate'] for c in entries),
                payload_minus_fresh=statistics.mean(c['payload_minus_fresh'] for c in entries)))
    sources = ['experiments/text_regions/'+name+'.py' for name in
               ('task_memory', 'task_payload', 'task_memory_train', 'task_curriculum', 'task_stream')]
    sources += ['scripts/run_payload_credit.py', 'scripts/summarize_payload_credit.py']
    return dict(schema=1, source_base='d47ddc6', seeds=seeds, rows=rows,
                contrasts=contrasts, means=means,
                source_sha256={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 43])
    args = parser.parse_args()
    summary = summarize(args.root, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['means'], indent=2))


if __name__ == '__main__':
    main()
