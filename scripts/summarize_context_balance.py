"""Validate completed paired exposure and summarize the logged measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def retention_cohorts(path, metric, segment, long_threshold):
    traces = [json.loads(line) for line in path.read_text().splitlines()]
    exact = [r for r in traces if r['kind'] == 'exact']
    if len(traces) != metric['answers'] or len(exact) != metric['groups']['exact']['count'] or abs(
            sum(r['nll'] for r in exact)/len(exact)-metric['groups']['exact']['nll']) > 1e-10:
        raise ValueError('Archived query traces disagree with aggregate scores.')
    groups = dict(acquisition=[r for r in exact if r['cycle'] == -1],
        beyond_segment=[r for r in exact if r['delay'] > segment],
        beyond_long_threshold=[r for r in exact if r['delay'] > long_threshold],
        corrected_beyond_segment=[r for r in exact if r['correction'] and r['delay'] > segment])
    return dict(segment_bytes=segment, long_threshold_bytes=long_threshold,
        groups={key:dict(count=len(rows), nll=sum(r['nll'] for r in rows)/len(rows) if rows else None,
            accuracy=sum(r['prediction'] == r['answer'] for r in rows)/len(rows) if rows else None)
            for key,rows in groups.items()})


def summarize(root):
    arms = {}
    for mode in ('group_clip', 'sum'):
        path = root/mode
        protocol = load(path/'protocol.json')
        evaluation = load(path/'evaluation.json')
        for tag, measurement in evaluation.items():
            for split in ('test','long_delay'):
                measurement[split]['exact_cohorts'] = retention_cohorts(
                    path/f"{measurement['signature']}_{split}_traces.jsonl", measurement[split],
                    protocol['protocol']['truncate_every'], max(protocol['data_config']['delays']))
        history = [json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
        steps = protocol['protocol']['steps']
        if evaluation['last']['step'] != steps or [r['step'] for r in history] != list(range(1,steps+1)):
            raise ValueError('Incomplete or duplicated retained trajectory: '+mode)
        for source in ('language','memory'):
            total = evaluation['last']['totals'][source]
            if total['updates'] != steps or total['bytes'] != steps*protocol['protocol']['batch']*protocol['protocol']['context']:
                raise ValueError('Retained source exposure disagrees with budget.')
        initial = load(path/'initial.json')
        selected = min([(initial['score'], 0)] + [(r['validation']['score'], r['step'])
            for r in history if 'validation' in r])
        if evaluation['best']['step'] != selected[1]:
            raise ValueError('Selected checkpoint disagrees with fixed validation rule.')
        gradients = {}
        for name in history[0]['gradient_groups']:
            rows = [r['gradient_groups'][name] for r in history]
            cosines = [r['cosine'] for r in rows if r['cosine'] is not None]
            gradients[name] = dict(language_norm_median=statistics.median(r['text_norm'] for r in rows),
                memory_norm_median=statistics.median(r['memory_norm'] for r in rows),
                language_clipped_fraction=sum(r['text_scale'] < 1 for r in rows)/len(rows),
                memory_clipped_fraction=sum(r['memory_scale'] < 1 for r in rows)/len(rows),
                mean_cosine=statistics.mean(cosines) if cosines else None,
                conflicting_fraction=sum(v < 0 for v in cosines)/max(len(cosines),1))
        arms[mode] = dict(protocol=protocol, initial=initial, evaluation=evaluation,
            validation=[dict(step=0, **initial)] + [dict(step=r['step'], **r['validation'])
                                                   for r in history if 'validation' in r],
            gradients=gradients, finite_norm_recoveries=sum(r['fp64_norm_recovered'] for r in history))
    a, b = arms['group_clip'], arms['sum']
    for name in ('config','data_config','parameters','raw_sha256','initialization','group_parameters'):
        if a['protocol'][name] != b['protocol'][name]:
            raise ValueError('Pair mismatch: '+name)
    stripped = [{k:v for k,v in arm['protocol']['protocol'].items() if k != 'gradient_mode'} for arm in (a,b)]
    if stripped[0] != stripped[1] or a['initial'] != b['initial']:
        raise ValueError('Pair has different exposure or initial evaluation.')
    first = [json.loads((root/mode/'history.jsonl').read_text().splitlines()[0]) for mode in ('group_clip','sum')]
    for group in first[0]['gradient_groups']:
        for key in ('text_norm','memory_norm','dot','cosine'):
            if first[0]['gradient_groups'][group][key] != first[1]['gradient_groups'][group][key]:
                raise ValueError('Pair first gradients differ before the experimental intervention.')
    for source in ('language','memory'):
        for key in ('updates','bytes','experts','answers'):
            if a['evaluation']['last']['totals'][source][key] != b['evaluation']['last']['totals'][source][key]:
                raise ValueError('Pair realized exposure mismatch.')
    selected_a, selected_b = a['evaluation']['best'], b['evaluation']['best']
    delta = dict(language_nll=selected_a['language']['nll']-selected_b['language']['nll'],
        answer_nll=selected_a['test']['answer_nll']-selected_b['test']['answer_nll'],
        long_exact_nll=selected_a['long_delay']['groups']['exact']['nll']-selected_b['long_delay']['groups']['exact']['nll'])
    checks = load(root/'checks.json')
    if checks['returncode'] or not checks['complete']:
        raise ValueError('Full checks are incomplete or failed.')
    environment = load(root/'environment.json')
    for name, expected in environment['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Training source changed after launch: '+name)
    parent = load(root/'parent_evaluation.json')
    if set(parent['measurements']) != {'language','test','long_delay','legacy_test'} or (
        parent['identity']['parent_sha256'] != a['protocol']['protocol']['parent_sha256'] or
        parent['identity']['raw_sha256'] != a['protocol']['raw_sha256']):
        raise ValueError('Parent comparison is incomplete or mismatched.')
    for split in ('test','long_delay'):
        metric = parent['measurements'][split]
        metric['exact_cohorts'] = retention_cohorts(root/f'parent_{split}_traces.jsonl', metric,
            a['protocol']['protocol']['truncate_every'], max(a['protocol']['data_config']['delays']))
    return dict(study='Contextual retention with paired task/group balancing',
        base_commit='7f3ce944547d800e133e999046a77d4e029a1851', arms=arms,
        group_clip_minus_sum=delta, checks=checks, environment=environment,
        paired_exposure_verified=True, parent=parent,
        recovery=load(root/'recovery.json') if (root/'recovery.json').exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/context_balance')
    parser.add_argument('--out', type=Path, default=ROOT/'experiments/text_regions/results/context_balance/summary.json')
    args = parser.parse_args()
    summary = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['group_clip_minus_sum']))


if __name__ == '__main__':
    main()
