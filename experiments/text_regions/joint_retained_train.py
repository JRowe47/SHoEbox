"""Resumable matched continuation of all three retained-composition parents."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .composition_train import atomic_save, source_hashes
from .config import GraphConfig
from .model import RegionLanguageModel
from .publication_data import make_batch
from .retained_composition import tables
from .retained_composition_train import validation
from .joint_retained import optimizer_for, update, publication_metrics


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['schema'] != 'shoebox-joint-retained-v1': raise ValueError('Unknown checkpoint schema.')
    model = RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights']); p = saved['protocol']
    opt = optimizer_for(model, p['mode'], p['producer_lr'], p['edge_lr'])
    opt.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['torch_rng'])
    return model, opt, saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--mode', choices=('joint','edge'), required=True)
    p.add_argument('--steps', type=int, default=80); p.add_argument('--batch', type=int, default=32)
    p.add_argument('--producer-lr', type=float, default=.0003); p.add_argument('--edge-lr', type=float, default=.001)
    p.add_argument('--validate-every', type=int, default=20); p.add_argument('--validation-count', type=int, default=128)
    p.add_argument('--stop-after', type=int); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); torch.set_num_threads(1)
    if min(a.steps,a.batch,a.validate_every,a.validation_count)<1 or not 0<a.producer_lr<1 or not 0<a.edge_lr<1:
        p.error('Invalid dimensions/rate.')
    if a.stop_after is not None and not 1<=a.stop_after<=a.steps: p.error('Invalid stopping step.')
    protocol = {k:v for k,v in vars(a).items() if k not in ('parent','out','resume','stop_after')}
    protocol.update(parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(), source_hashes=source_hashes(),
                    gaps=[0,4,8,16], segment=16,
                    objective='Retained and write-candidate normalized MSE to frozen A ingress; balanced gate BCE; final byte CE; original router regularizers',
                    selection='Validation minimum task accuracy, mean accuracy, lower mean NLL; includes unchanged parent',
                    optimizer='AdamW beta=(.9,.95), decay=.01, global trainable gradient clip=1')
    a.out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        model,opt,saved = load(a.out/'last.pt')
        if saved['protocol']!=protocol: raise ValueError('Resume requires identical source and protocol.')
        step,history,best,totals = saved['step'],saved['history'],saved['best'],saved['totals']
    else:
        if (a.out/'last.pt').exists(): raise ValueError('Use --resume or a new output directory.')
        torch.manual_seed(a.seed); parent=torch.load(a.parent,map_location='cpu',weights_only=True)
        model=RegionLanguageModel(GraphConfig.from_dict(parent['config']));model.load_state_dict(parent['weights'])
        opt=optimizer_for(model,a.mode,a.producer_lr,a.edge_lr)
        step,history,best=0,[],[-1.,-1.,-1e30]
        totals=dict(expert_executions=0,external_events=0,source_occurrences=0,training_seconds=0.)
    def save(name):
        atomic_save(a.out/name,dict(schema='shoebox-joint-retained-v1',config=model.cfg.to_dict(),
            weights=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            protocol=protocol,step=step,history=history,best=best,totals=totals))
    if not a.resume:
        key,scores=validation(model,a.seed,a.validation_count);best=list(key)
        history.append(dict(step=0,validation=scores));save('initial.pt');save('best.pt');save('last.pt')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for step in range(step+1,(a.stop_after or a.steps)+1):
        kind='recall' if step%2 else 'correction';gap=protocol['gaps'][((step-1)//2)%4]
        start=(step-1)*a.batch;b=make_batch('train',start,a.batch,a.seed,gap,kind)
        tb=tables('train',start,a.batch,a.seed,b.answers)
        tick=time.perf_counter();row=update(model,opt,b,tb,protocol['segment'])
        row.update(step=step,kind=kind,gap=gap,training_seconds=time.perf_counter()-tick);history.append(row)
        for k in totals:totals[k]+=row[k]
        if step%a.validate_every==0 or step==a.steps:
            key,scores=validation(model,a.seed,a.validation_count);row['validation']=scores
            if key>tuple(best):best=list(key);save('best.pt')
            print(json.dumps(dict(step=step,validation=[r['accuracy'] for r in scores],best=best)),flush=True)
        if step==1 or step%10==0:
            print(json.dumps({k:v for k,v in row.items() if k not in ('keys','validation')}),flush=True)
        if step%10==0 or step==(a.stop_after or a.steps):
            save('last.pt');(a.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(dict(complete=step==a.steps,step=step,best=best,**totals)),flush=True)


if __name__=='__main__':main()
