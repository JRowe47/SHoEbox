"""Delayed, source-and-slot-specific byte retrieval: two targeted learning checks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from events import ByteBatch
from hat import HAT
from model import HATSHoE
from train import optimizer_for
from verify import tiny_config


def episode(generator,batch=32,delay=8):
    stored=torch.randint(0,8,(batch,2,2),generator=generator)
    query_source=int(torch.randint(0,2,(),generator=generator))
    query_slot=int(torch.randint(0,2,(),generator=generator))
    values=torch.cat((stored[:,:,0],stored[:,:,1],
                      torch.randint(16,32,(batch,delay),generator=generator),torch.full((batch,1),255)),1)
    links=torch.tensor([1,2,1,2]+[1+i%2 for i in range(delay)]+[query_source+1])[None].expand(batch,-1)
    slots=torch.tensor([0,0,1,1]+[4]*delay+[query_slot+2])[None].expand(batch,-1)
    targets=torch.full_like(values,-100);targets[:,-1]=stored[:,query_source,query_slot]
    return ByteBatch(values,links,slots),targets


class HATReadout(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.hat=HAT(cfg);self.head=nn.Linear(cfg.width,8)
    def forward(self,e,history=True):
        r=self.hat(e,memory_enabled=history)
        return self.head(r['payload']),r['state']


@torch.no_grad()
def evaluate(model,arm,seed=8001,batches=16,history=True,erase_source=False,erase_slot=False):
    model.eval();gen=torch.Generator().manual_seed(seed);correct=total=0;loss=0.
    for _ in range(batches):
        e,y=episode(gen)
        if erase_source: e=ByteBatch(e.values,torch.ones_like(e.links),e.slots)
        if erase_slot: e=ByteBatch(e.values,e.links,torch.zeros_like(e.slots))
        if arm=='hat': logits=model(e,history)[0][:,-1]
        else: logits=model(e,hat_memory=False,expert_memory=history,forced_child=0,force_discovery=True)['logits'][:,-1]
        correct+=int((logits.argmax(-1)==y[:,-1]).sum());total+=len(y)
        loss+=float(F.cross_entropy(logits,y[:,-1]))
    return {'accuracy':correct/total,'nll':loss/batches,'episodes':total}


def run(arm,steps,out,seed):
    torch.manual_seed(seed);gen=torch.Generator().manual_seed(seed+1);cfg=tiny_config(exploration=0)
    model=HATReadout(cfg) if arm=='hat' else HATSHoE(cfg)
    optimizer=optimizer_for(model,.003);start=time.perf_counter();history=[]
    initial=evaluate(model,arm)
    for step in range(1,steps+1):
        model.train();e,y=episode(gen);optimizer.zero_grad(set_to_none=True)
        if arm=='hat': logits,_=model(e);loss=F.cross_entropy(logits[:,-1],y[:,-1])
        else:
            r=model(e,y,hat_memory=False,forced_child=0,force_discovery=True,reinforce=False)
            loss=r['ce']+.001*r['aux']
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)): raise RuntimeError('Nonfinite gradients.')
        optimizer.step()
        if step%100==0 or step==steps:
            row={'arm':arm,'step':step,'train_loss':float(loss.detach()),**evaluate(model,arm,batches=4)}
            history.append(row);print(json.dumps(row),flush=True)
    results={'arm':arm,'steps':steps,'batch_size':32,'delay_events':8,'chance_accuracy':.125,
        'initial':initial,'normal':evaluate(model,arm),'history_disabled':evaluate(model,arm,history=False),
        'source_identity_removed':evaluate(model,arm,erase_source=True),
        'slot_identity_removed':evaluate(model,arm,erase_slot=True),
        'seconds':time.perf_counter()-start,'history':history,
        'protocol':'Four random writes to two links x two slots; eight distractors; query supplies identical 0xff with addressed source and slot. Fresh held-out episodes, seed8001.',
        'scope':'HAT alone + 8-class readout' if arm=='hat' else 'Two-child model with HAT history disabled and child0 forced; byte decoder; tests expert-block memory in isolation.'}
    folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(results,indent=2))
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'data_rng':gen.get_state(),'results':results},folder/'checkpoint.pt')
    print(json.dumps(results),flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=500)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--arm',choices=['hat','expert','both'],default='both')
    p.add_argument('--out',default='results/learning');args=p.parse_args();torch.set_num_threads(args.threads)
    results=[]
    for i,arm in enumerate(['hat','expert'] if args.arm=='both' else [args.arm]):
        results.append(run(arm,args.steps,Path(args.out),40201+(0 if arm=='hat' else 1)))
