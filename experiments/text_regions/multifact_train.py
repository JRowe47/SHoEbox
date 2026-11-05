"""Resume local multi-field acquisition learning with a frozen composer."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .config import GraphConfig
from .model import RegionLanguageModel
from .composition_train import atomic_save,source_hashes
from .multifact import build,optimizer_for,update,table_batches
from .multifact_data import make_batch,KINDS
from .multifact_eval import validation


def load(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['schema']!='shoebox-multifact-v1':raise ValueError('Unknown checkpoint schema.')
    model=RegionLanguageModel(GraphConfig.from_dict(saved['config']));model.load_state_dict(saved['weights'])
    p=saved['protocol'];opt=optimizer_for(model,p['producer_lr'],p['edge_lr']);opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng']);return model,opt,saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=80)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--producer-lr',type=float,default=.0003)
    p.add_argument('--edge-lr',type=float,default=.001);p.add_argument('--validate-every',type=int,default=20)
    p.add_argument('--validation-count',type=int,default=64);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args();torch.set_num_threads(1)
    if min(a.steps,a.batch,a.validate_every,a.validation_count)<1 or not 0<a.producer_lr<1 or not 0<a.edge_lr<1:p.error('Invalid dimensions or rate.')
    if a.stop_after is not None and not 1<=a.stop_after<=a.steps:p.error('Invalid stopping point.')
    protocol={k:v for k,v in vars(a).items() if k not in ('parent','out','resume','stop_after')}
    protocol.update(parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),source_hashes=source_hashes(),
        gaps=[0,4,8,16],segment=16,selection='Validation minimum of 15 task/field accuracies, mean accuracy, lower mean NLL; step zero eligible',
        interface='Explicit three-field query address; latest valid shared publication; raw source bytes reach both local field producers',
        initialization='Clone each owner from its matching joint-retention parent; private input embeddings swap colon and equals')
    protocol['gate_objective']='0.5 own writes + 0.25 other-field writes + 0.25 ordinary holds, separately normalized'
    a.out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        model,opt,s=load(a.out/'last.pt')
        if s['protocol']!=protocol:raise ValueError('Resume requires identical source and protocol.')
        step,history,best,totals=s['step'],s['history'],s['best'],s['totals']
    else:
        if (a.out/'last.pt').exists():raise ValueError('Use --resume or a fresh output directory.')
        torch.manual_seed(a.seed);model=build(a.parent,a.seed);opt=optimizer_for(model,a.producer_lr,a.edge_lr)
        step=0;history=[];best=[-1.,-1.,-1e30];totals=dict(expert_executions=0,external_events=0,source_occurrences=0,training_seconds=0.)
    def save(name):
        atomic_save(a.out/name,dict(schema='shoebox-multifact-v1',config=model.cfg.to_dict(),weights=model.state_dict(),
            optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),protocol=protocol,step=step,history=history,best=best,totals=totals))
    if not a.resume:
        key,scores=validation(model,a.seed,a.validation_count);best=list(key)
        history.append(dict(step=0,validation=scores));save('initial.pt');save('best.pt');save('last.pt')
        print(json.dumps(dict(step=0,best=best,parameters=model.parameter_summary())),flush=True)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for step in range(step+1,(a.stop_after or a.steps)+1):
        kind=KINDS[(step-1)%5];gap=protocol['gaps'][((step-1)//5)%4];start=(step-1)*a.batch
        batch=make_batch('train',start,a.batch,a.seed,gap,kind);tb=table_batches(batch,'train',start,a.seed)
        tick=time.perf_counter();row=update(model,opt,batch,tb);row.update(step=step,kind=kind,gap=gap,training_seconds=time.perf_counter()-tick)
        history.append(row)
        for k in totals:totals[k]+=row[k]
        if step%a.validate_every==0 or step==a.steps:
            key,scores=validation(model,a.seed,a.validation_count);row['validation']=scores
            if key>tuple(best):best=list(key);save('best.pt')
            print(json.dumps(dict(step=step,validation=list(key),best=best)),flush=True)
        if step==1 or step%10==0:print(json.dumps({k:v for k,v in row.items() if k not in ('validation','table_keys')}),flush=True)
        if step%10==0 or step==(a.stop_after or a.steps):
            save('last.pt');(a.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(dict(complete=step==a.steps,step=step,best=best,**totals)),flush=True)


if __name__=='__main__':main()
