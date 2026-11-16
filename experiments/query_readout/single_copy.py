"""Initialization-only control: one supplied value, one query, original padding.

Derived worlds retain their original two-record split group. This deliberately
tests copying the only supplied value, not unseen key/value generalization.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from .adapter import Adapter
from .run import train,evaluate,write
from experiments.mad_next.tasks import Spec,World,dataset,encode,parse,VALUE,solve,fingerprint


def single_cases(spec,split,count):
    result=[]
    for ep in dataset(spec,count,split):
        records,queries=parse(ep);key=queries[0]
        record=next((k,v-VALUE) for _,k,v in records if k==key)
        world=World('binding',(record,));one=encode(world,spec,query_order=[key],group_key=ep.split_key)
        assert solve(one,spec)==one.answer
        result.append(one)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan={'experiment':'EXP01 single-supplied-value copy control','cores':['minimal','full'],
          'seeds':[1,7,19],'parameters':500000,'budget':250000,
          'scope':'paired source-world folds; copying only supplied value; not unseen one-key mappings'}
    write(a.out.with_suffix('.plan.json'),plan)
    spec=Spec('binding',vocab=2);tr=single_cases(spec,'train',2048);dv=single_cases(spec,'dev',512);te=single_cases(spec,'test',1024)
    rows=[]
    for core in plan['cores']:
        for seed in plan['seeds']:
            with Adapter(seed=seed,core=core,kind='native') as m:
                row=train(m,tr,dv,plan['budget'],plan['budget'],1067)
                row.update(core=core,seed=seed,inventory=m.inventory(),test=evaluate(m,te))
                rows.append(row);print(core,seed,row['test']['native_rollout']['native_exact'],flush=True)
    write(a.out,{'plan':plan,'rows':rows,'fingerprints':{k:fingerprint(v) for k,v in [('train',tr),('dev',dv),('test',te)]}})


if __name__=='__main__':main()
