"""Verify matched exposure, selection and source identity before reporting."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def summarize(root):
    load=lambda path:json.loads(path.read_text())
    evaluation=load(root/'evaluation.json')
    if set(evaluation)!={'parent','state_best','state_last','publication_best','publication_last'}:
        raise ValueError('Final evaluation is incomplete.')
    environment=load(root/'environment.json')
    for path,sha in environment['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Training source changed: '+path)
    arms={}
    for mode in ('state','publication'):
        folder=root/mode; protocol=load(folder/'protocol.json')
        initial=load(folder/'initial.json')
        history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        steps=protocol['protocol']['steps']
        if [r['step'] for r in history]!=list(range(1,steps+1)):
            raise ValueError('Incomplete or duplicate retained history: '+mode)
        expected=min([(initial['score'],0)]+[(r['validation']['score'],r['step']) for r in history if 'validation' in r])[1]
        if evaluation[mode+'_best']['step']!=expected or evaluation[mode+'_last']['step']!=steps:
            raise ValueError('Selection/final step mismatch.')
        totals=history[-1]['totals']
        for source in totals:
            if totals[source]['updates']!=steps or totals[source]['bytes']!=steps*protocol['protocol']['batch']*protocol['protocol']['context']:
                raise ValueError('Source exposure mismatch.')
        arms[mode]=dict(protocol=protocol,initial=initial,totals=totals,
            best=evaluation[mode+'_best'],last=evaluation[mode+'_last'],
            validation=[dict(step=0,**initial)]+[dict(step=r['step'],**r['validation']) for r in history if 'validation' in r],
            state_gradient_median={s:statistics.median(r[s]['state_gradient_norm'] for r in history) for s in totals})
    a,b=arms['state'],arms['publication']
    if a['initial']!=b['initial']:
        raise ValueError('Arms differ before learning.')
    pa,pb=[dict(x['protocol']['protocol']) for x in (a,b)]
    pa.pop('state_credit');pb.pop('state_credit')
    if pa!=pb or a['protocol']['raw_sha256']!=b['protocol']['raw_sha256']:
        raise ValueError('Matched protocols differ.')
    for source in a['totals']:
        for key in ('updates','bytes','experts','answers'):
            if a['totals'][source][key]!=b['totals'][source][key]:
                raise ValueError('Matched exposure differs.')
    checks=load(root/'checks.json')
    if not checks['passed']:
        raise ValueError('Verification incomplete.')
    return dict(study='Conditional credit through producer-state transitions',
        base_commit='4d919d1d4e23d6af3fb052987c2fe1cdd0bcad59',parent=evaluation['parent'],arms=arms,
        state_minus_publication=dict(language_nll=a['best']['language']['nll']-b['best']['language']['nll'],
            answer_nll=a['best']['test']['answer_nll']-b['best']['test']['answer_nll'],
            delayed_exact_nll=a['best']['long_delay']['exact_beyond_96']['nll']-b['best']['long_delay']['exact_beyond_96']['nll']),
        checks=checks,environment=environment,trace_benchmark=load(root/'trace_benchmark.json'),
        paired_exposure_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/state_credit')
    p.add_argument('--out',type=Path,default=ROOT/'experiments/text_regions/results/state_credit/summary.json')
    a=p.parse_args();summary=summarize(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['state_minus_publication']))


if __name__=='__main__':
    main()
