"""Validate the matched recall continuation and summarize its raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    study = root/'study'
    provenance = read(study/'provenance.json')
    if hashlib.sha256((study/'parent.pt').read_bytes()).hexdigest() != provenance['parent_sha256']:
        raise ValueError('Parent checkpoint hash mismatch.')
    for name in ('task_memory.py', 'task_memory_train.py', 'task_recall.py', 'task_curriculum.py'):
        relative = 'experiments/text_regions/'+name
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != provenance['source_sha256'][relative]:
            raise ValueError(f'Training implementation changed during the study: {name}')
    parent = torch.load(study/'parent.pt', map_location='cpu', weights_only=False)
    parent_step = parent['step']
    rows, matched = [], []
    for layout in ('distinct', 'pooled'):
        folder = study/layout
        invocation = read(folder/'invocation.json')
        history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        evaluation = read(folder/'evaluation.json')
        completion = read(folder/'completion.json')
        if not completion['complete'] or not completion['audited']:
            raise ValueError(f'Incomplete run: {layout}')
        last = history[-1]
        config = dict(invocation['config'])
        if config.pop('recall_layout') != layout or config['recall_slots'] < 1:
            raise ValueError('Wrong recall treatment.')
        if invocation['initialization']['sha256'] != provenance['parent_sha256']:
            raise ValueError('Wrong parent initialization.')
        selected = min(history, key=lambda row: row['validation']['answer_nll'])['step']
        if selected != evaluation['selected_step'] or last['step'] != invocation['steps']:
            raise ValueError('Incomplete budget or wrong selected checkpoint.')
        if evaluation['final_budget']['step'] != last['step']:
            raise ValueError('Wrong final evaluation budget.')
        matched.append(dict(config=config, data_digest=invocation['data_digest'],
            parameters=invocation['model_parameters'], steps=last['step'],
            bytes=last['training_external_bytes'], experts=last['training_expert_executions'],
            coverage=last['stage_totals'],
            training={k: invocation[k] for k in ('seed', 'data_seed', 'batch', 'context', 'lr',
                'text_weight', 'write_cost', 'curriculum', 'truncate_every', 'eval_every', 'eval_count')}))
        row = dict(layout=layout, parameters=invocation['model_parameters'], updates=last['step'],
            bytes=last['training_external_bytes'], expert_executions=last['training_expert_executions'],
            selected_step=selected, train_validate_seconds=last['elapsed_seconds'],
            trace_bytes=last['trace_bytes'], evaluation=evaluation, audits={})
        for checkpoint in ('best', 'last'):
            saved = torch.load(folder/(checkpoint+'.pt'), map_location='cpu', weights_only=False)
            merge = saved['weights']['recall.payload_merge.weight']
            d, slots = saved['config']['width'], saved['config']['recall_slots']
            blocks = merge.reshape(d, slots, d)
            row[checkpoint+'_role_map_spread'] = float((blocks-blocks.mean(1, keepdim=True)).norm())
            for split in ('test', 'long_delay'):
                audit = read(folder/f'audit_{checkpoint}_{split}'/'summary.json')
                if audit['live_fork_argmax_changes'] or audit['live_fork_max_logit_error'] > 1e-4:
                    raise ValueError('Query forks do not reproduce the live stream.')
                ev = evaluation if checkpoint == 'best' else evaluation['final_budget']
                original = ev[split]['groups']['exact']
                exact = audit['groups']['exact']
                if abs(original['nll']-exact['intact']['nll']) > 1e-5 or original['accuracy'] != exact['intact']['accuracy']:
                    raise ValueError('Audit/evaluation disagreement.')
                row['audits'][f'{checkpoint}_{split}'] = dict(
                    exact=exact, corrected_exact=audit['groups'].get('corrected_exact'),
                    deltas={name: exact[name]['nll']-exact['intact']['nll'] for name in
                        ('recall_pooled', 'recall_no_feedback', 'shuffle', 'blocked')},
                    aged_publications=audit['aged_shuffle']['groups']['exact'],
                    live_fork_max_logit_error=audit['live_fork_max_logit_error'])
        rows.append(row)
    if matched[0] != matched[1]:
        raise ValueError('Recall arms do not have matched configuration/data/compute budgets.')
    seven = root/'seven_slot'
    seven_invocation = read(seven/'invocation.json')
    seven_history = [json.loads(line) for line in (seven/'history.jsonl').read_text().splitlines()]
    if seven_history[-1]['step'] != seven_invocation['steps']:
        raise ValueError('Seven-slot integration run is incomplete.')
    return dict(schema=1, purpose='Architecture integration and short training shakedown; retain the feature for larger-scale ablations.',
        parent_sha256=provenance['parent_sha256'], parent_step=parent_step, rows=rows,
        seven_slot=dict(invocation=seven_invocation, final=seven_history[-1]),
        read_mass_definition='Aggregate mass is averaged over slots. Active-slot counts sum nonnull choices; unique counts merge repeated owners.',
        nll_direction='Lower is better. Positive intervention delta means intact behavior helped.',
        training_source_sha256=provenance['source_sha256'],
        audit_source_sha256=hashlib.sha256((ROOT/'experiments/text_regions/task_query_audit.py').read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    summary = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({row['layout']: dict(selected=row['selected_step'],
        test=row['evaluation']['test']['groups']['exact']['nll'],
        final_test=row['evaluation']['final_budget']['test']['groups']['exact']['nll']) for row in summary['rows']}))


if __name__ == '__main__':
    main()
