"""Held-out matched comparison and exact standalone inference exports."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_retained_train import load
from experiments.text_regions.joint_retained import publication_metrics
from experiments.text_regions.retained_composition import evaluate,evaluate_revision
from experiments.text_regions.composition_train import atomic_save
from evaluate_retained_composition import conditions
from audit_retained_composition import cached_source_memory


def evaluate_run(path):
    torch.set_num_threads(1);model,_,saved=load(path/'best.pt');seed=saved['protocol']['seed']
    target=path/'evaluation.json';records=json.loads(target.read_text()) if target.exists() else {}
    def save():
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(records,indent=2)+'\n');tmp.replace(target)
    with cached_source_memory():
        for name,kwargs in conditions().items():
            if name not in records:
                records[name]=evaluate(model,'test',seed,**kwargs);save()
                print(json.dumps(dict(condition=name,accuracy=records[name]['accuracy'])),flush=True)
        if 'revision_after_queries' not in records:
            records['revision_after_queries']=evaluate_revision(model,'test',seed);save()
        for gap in (16,64):
            name=f'publication_{gap}'
            if name not in records:
                records[name]=publication_metrics(model,'test',seed,gap=gap);save()
        if 'initial' not in records:
            initial,_,_=load(path/'initial.pt')
            records['initial']={name:evaluate(initial,'test',seed,**kwargs)
                for name,kwargs in conditions().items() if name in ('recall_16','correction_16','withheld_16')}
            save()
        if 'final_endpoint' not in records:
            final,_,_=load(path/'last.pt');records['final_endpoint']=evaluate(final,'test',seed);save()
    atomic_save(path/'inference.pt',dict(schema='shoebox-joint-retained-inference-v1',
        config=model.cfg.to_dict(),weights=model.state_dict(),selected_step=saved['step']))
    print(json.dumps(dict(selected_step=saved['step'],complete=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
    evaluate_run(p.parse_args().path)
