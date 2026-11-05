"""Reproduce the separate 80-update development pilot; not a confirmatory run."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.retained_composition import build,tables,update,evaluate
from experiments.text_regions.publication_data import make_batch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory.')
    a.out.mkdir(parents=True);torch.set_num_threads(1)
    parents=ROOT/'artifacts/retained_composition/parents'
    model=build(parents/'composer.pt',parents/'producer_20261120.pt',20261120)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=.01)
    history=[];tick=time.perf_counter()
    for step in range(1,81):
        start=(step-1)*32;b=make_batch('train',start,32,20261120,(0,4,8,16)[(step-1)%4],
                                     'correction' if step%2==0 else 'recall')
        tb=tables('train',start,32,20261120,b.answers);r=update(model,opt,b,tb);r['step']=step;history.append(r)
        if step%20==0:
            scores={k:evaluate(model,'validation',20261120,64,16,k)['accuracy'] for k in ('recall','correction')}
            print(json.dumps(dict(step=step,validation=scores,seconds=time.perf_counter()-tick)),flush=True)
            torch.save(dict(config=model.cfg.to_dict(),weights=model.state_dict(),optimizer=opt.state_dict(),history=history),a.out/'last.pt')
            (a.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')


if __name__=='__main__':main()
