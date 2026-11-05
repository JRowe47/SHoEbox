"""Recoverable held-out evaluation and standalone inference exports."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.retained_composition import evaluate,evaluate_revision
from experiments.text_regions.retained_composition_train import load
from experiments.text_regions.composition_train import atomic_save


def conditions():
    conditions={}
    for gap in (16,64):
        for kind,control,name in (('recall','carried','recall'),('correction','carried','correction'),
                                   ('correction','withheld','withheld')):
            conditions[f'{name}_{gap}']=dict(gap=gap,kind=kind,memory_control=control)
    conditions.update(k1=dict(rounds=1),k4=dict(rounds=4),no_exchange=dict(feedback=False),
        no_edge=dict(query_control='no_edge'),reset_memory=dict(query_control='reset_memory'),
        shuffled_memory=dict(query_control='shuffle_memory'),forced_always=dict(policy='always'),
        repeated_queries=dict(requests=4))
    return conditions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);a=p.parse_args()
    torch.set_num_threads(1);model,_,saved=load(a.path/'best.pt');seed=saved['protocol']['seed']
    path=a.path/'evaluation.json';records=json.loads(path.read_text()) if path.exists() else {}
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(records,indent=2)+'\n');temp.replace(path)
    for name,kwargs in conditions().items():
        if name in records:continue
        records[name]=evaluate(model,'test',seed,**kwargs);save()
        print(json.dumps(dict(condition=name,accuracy=records[name]['accuracy'])),flush=True)
    if 'revision_after_queries' not in records:
        records['revision_after_queries']=evaluate_revision(model,'test',seed);save()
        print(json.dumps({k:v['accuracy'] for k,v in records['revision_after_queries'].items()}),flush=True)
    atomic_save(a.path/'inference.pt',dict(schema='shoebox-retained-composition-inference-v1',
        config=model.cfg.to_dict(),weights=model.state_dict(),selected_step=saved['step']))
    last,_,_=load(a.path/'last.pt')
    if 'final_endpoint' not in records:
        records['final_endpoint']=evaluate(last,'test',seed);save()


if __name__=='__main__':main()
