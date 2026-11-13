"""Read realized teaching and eligibility delivery from the tested checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,nargs='+',default=[ROOT/f'results/relay_full_relays_seed{s}/last.npz' for s in (1,2)])
    a=p.parse_args();rows=[]
    for seed,path in enumerate(a.checkpoints,1):
        m=Model.load(path);teaching=m.relay_teaching_status();lace=m.lace_metrics()
        composition=m.composition_metrics();stats=m.stats()
        for r,name in enumerate(core.NAMES):
            t=teaching[name]
            assert t['scheduled']==sum(t[k] for k in ('neural_targets','relay_targets','nonlearning_targets','expired_producers'))+int(t['pending_tick']>0)
            assert t['duplicate_schedules']==0 and t['expired_producers']==0
            rows.append(dict(seed=seed,region=name,neural_executions=int(stats['executions'][r]),
                             teaching=t,lace=lace[name],composition=composition[name]))
        m.close()
    result=dict(checkpoint_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in a.checkpoints],
                note='Saved training counters; no extra training. A1/B1 use their original local teacher and do not populate relay forecast counters.',regions=rows)
    (ROOT/'results/contribution_credit.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        if row['seed']==1 and row['region'] in ('L2','L3','E4','F4','V4','G5','EC-C','Y'):
            print(row['region'],'local targets',row['teaching']['neural_targets']+row['teaching']['relay_targets'],
                  'credit edges',row['lace']['credit_edges'],'missing parents',row['lace']['missing_parent'],
                  'expired input edges',row['composition']['expired_input_edges'])


if __name__=='__main__':main()
