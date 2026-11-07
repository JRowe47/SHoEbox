"""Assemble completed matched-run evidence without rerunning trained models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def query_retention(path, segment):
    selected = weighted_age = old_weight = 0.
    by_receiver = {}
    for line in path.read_text().splitlines():
        query = json.loads(line)
        for receiver, reads in query['reads'].items():
            r = by_receiver.setdefault(receiver,dict(selected=0.,older_than_segment=0.,weighted_age=0.))
            for weights, valid, occurrences in zip(reads['weights'],reads['valid'],reads['occurrences']):
                for slot in weights:
                    for weight, present, occurrence in zip(slot[:-1],valid,occurrences):
                        if not present:
                            continue
                        age = query['position']-occurrence
                        if age < 0:
                            raise ValueError('Read refers to a future publication.')
                        selected += weight; weighted_age += weight*age
                        old_weight += weight*(age > segment)
                        r['selected'] += weight; r['weighted_age'] += weight*age
                        r['older_than_segment'] += weight*(age > segment)
    return dict(selected_slot_reads=selected,mean_selected_age=weighted_age/max(selected,1),
        fraction_older_than_segment=old_weight/max(selected,1),segment_bytes=segment,
        receivers={name:dict(mean_selected_age=r['weighted_age']/max(r['selected'],1),
            fraction_older_than_segment=r['older_than_segment']/max(r['selected'],1))
            for name,r in by_receiver.items()},
        definition='At answer queries, selected valid edge publications across all receivers and rounds; age is query byte minus publication byte.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts/graph_publication')
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/text_regions/results/graph_publication')
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text())
    arms = {}
    for arm in ('credit','no_credit'):
        folder = args.root/arm
        protocol = load(folder/'protocol.json')
        evaluation = load(folder/'evaluation.json')
        if evaluation['last']['step'] != protocol['protocol']['steps']:
            raise ValueError('Incomplete training budget.')
        segment = protocol['protocol']['truncate_every'] or protocol['protocol']['context']
        arms[arm] = dict(protocol=protocol, initial=load(folder/'initial.json'),
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()],
            evaluation=evaluation, query_retention=query_retention(folder/'best_test_traces.jsonl',segment))
    a,b = [arms[n] for n in ('credit','no_credit')]
    for key in ('config','data_config','raw_sha256','parameters','initialization'):
        if a['protocol'][key] != b['protocol'][key]:
            raise ValueError('Pair mismatch: '+key)
    policies = [{k:v for k,v in arm['protocol']['protocol'].items() if k != 'delayed_credit'} for arm in (a,b)]
    if policies[0] != policies[1]:
        raise ValueError('Pair training budgets differ.')
    if a['initial'] != b['initial']:
        raise ValueError('Pair starting metrics differ.')
    for source in ('language','memory'):
        for key in ('updates','bytes','experts','answers'):
            if a['evaluation']['last']['totals'][source][key] != b['evaluation']['last']['totals'][source][key]:
                raise ValueError('Pair realized exposure differs.')
    checks = load(args.root/'checks.json')
    if not checks['complete'] or checks['returncode']:
        raise ValueError('Checks have not passed.')
    summary = dict(study='Unified graph task publication and causal demand',
        parent_sha256=hashlib.sha256((args.root/'parent.pt').read_bytes()).hexdigest(),
        paired_exposure_verified=True,arms=arms,checks=checks)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for arm,results in arms.items():
        best = results['evaluation']['best']
        print(json.dumps(dict(arm=arm,step=best['step'],language=best['language']['nll'],
            answer=best['memory']['test']['answer_nll'],exact=best['memory']['test']['groups']['exact']['nll'],
            older_reads=results['query_retention']['fraction_older_than_segment'])))


if __name__ == '__main__':
    main()
