"""Byte-level language shakedown through both static and live HAT addresses."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
import torch
if __package__:
    from .events import ByteBatch
    from .model import HATConfig,HATSHoE
else:
    from events import ByteBatch
    from model import HATConfig,HATSHoE


def data_load(path):
    raw=Path(path).read_bytes();a,b=int(.9*len(raw)),int(.95*len(raw))
    values=torch.tensor(list(raw[:b]),dtype=torch.long)
    return values[:a],values[a:b],{'sha256':hashlib.sha256(raw).hexdigest(),
        'bytes':len(raw),'train_bytes':a,'validation_bytes':b-a,'untouched_test_bytes':len(raw)-b,
        'split':'contiguous 90/5/5; test unused','alphabet':'all 256 bytes; no vocabulary fitting'}


def batch(data,size,context,generator,interface='mixed'):
    starts=torch.randint(len(data)-context,(size,),generator=generator)
    z=data[starts[:,None]+torch.arange(context+1)]
    x,y=z[:,:-1],z[:,1:]
    if interface=='static': return ByteBatch.static(x),y
    if interface=='live': return ByteBatch.stream(x,1,0),y
    links=torch.zeros_like(x);slots=x.clone()
    # Identical source bytes/targets; only address transport differs by row.
    for row in range(size):
        if row%4 in (1,2):
            links[row]=row%4;slots[row]=0 if row%4==1 else 7
    return ByteBatch(x,links,slots),y


def optimizer_for(model,lr):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.ndim<2 or getattr(p,'_no_weight_decay',False) or 'embedding' in name: no_decay.append(p)
        else: decay.append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.01},
                             {'params':no_decay,'weight_decay':0}],lr=lr,betas=(.9,.95))


@torch.no_grad()
def evaluate(model,data,batches=4,size=4,seed=333,interface='mixed',**kwargs):
    old=model.training;model.eval();gen=torch.Generator().manual_seed(seed)
    losses=[]
    for _ in range(batches):
        e,y=batch(data,size,model.cfg.context,gen,interface)
        losses.append(float(model(e,y,**kwargs)['ce']))
    model.train(old)
    return sum(losses)/len(losses)


@torch.no_grad()
def diagnose(model,data):
    model.eval();e,y=batch(data,4,model.cfg.context,torch.Generator().manual_seed(5151))
    r=model(e,y,trace=True);route=r['route']
    counts=torch.bincount(route['choice'].flatten(),minlength=2)
    expert_usage=[[torch.zeros(n,dtype=torch.long) for n in model.cfg.child_config().expert_counts] for _ in range(2)]
    for execution in r['executions']:
        for layer,tr in enumerate(execution['trace']):
            expert_usage[execution['child']][layer]+=torch.bincount(tr['experts'].flatten(),minlength=len(expert_usage[execution['child']][layer]))
    unique=0
    for l,n in enumerate(model.shared_counts):
        unique+=int(((expert_usage[0][l][:n]+expert_usage[1][l][:n])>0).sum())
        unique+=sum(int((expert_usage[j][l][n:]>0).sum()) for j in range(2))
    return {'events':e.values.numel(),'child_counts':counts.tolist(),
        'child_fractions':(counts/counts.sum()).tolist(),'fast_path_fraction':float(route['fast'].float().mean()),
        'head_probes_per_event':float(route['probed'].sum(-1).float().mean()),
        'unique_physical_experts_used':unique,'parent_semantic_loss':float(r['semantic_loss']),
        'hat_symbols_used_per_level':[len(torch.unique(r['hat']['symbol_ids'][:,:,i])) for i in range(model.cfg.hat_levels)],
        'stream_state_bytes_batch4_three_links':r['state'].storage_bytes(),
        'expert_usage':[[v.tolist() for v in row] for row in expert_usage]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--context',type=int,default=128);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=.002);p.add_argument('--out',default='results/language')
    p.add_argument('--data',default='data/tiny_shakespeare.txt');p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--eval-every',type=int,default=50);args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    train,val,data=data_load(args.data);model=HATSHoE(HATConfig(context=args.context,seed=args.seed))
    optimizer=optimizer_for(model,args.lr);gen=torch.Generator().manual_seed(args.seed+1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    protocol={'args':vars(args),'config':asdict(model.cfg),'parameters':model.parameter_summary(),
        'data':data,'torch':str(torch.__version__),'python':platform.python_version(),
        'optimizer':'AdamW beta=(.9,.95), decay .01 on matrix weights except embeddings and marked recurrence parameters; clip1',
        'schedule':'warmup min(40,steps/10), plateau to 80%, linear cooldown to 20% peak',
        'input_mix':'rows 0/3 static byte bank, row1 live link1 slot0, row2 live link2 slot7',
        'state_policy':'independent training windows reset state; full BPTT inside each window; inference carries state',
        'evaluation':'fixed windows seed333, mean cross-entropy nats/byte',
        'initial_parameter_sha256':hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    initial=evaluate(model,val);print(json.dumps({'initial_validation_loss':initial,'parameters':model.parameter_summary()}),flush=True)
    history=[{'step':0,'bytes':0,'validation_loss':initial}];seconds=0.;recent=[];started=time.perf_counter()
    stream_hash=hashlib.sha256()
    with (out/'history.jsonl').open('w') as log:
        log.write(json.dumps(history[0])+'\n')
        for step in range(1,args.steps+1):
            model.train();e,y=batch(train,args.batch,args.context,gen)
            stream_hash.update(e.values.numpy().tobytes());stream_hash.update(y.numpy().tobytes())
            warmup=min(40,max(1,args.steps//10));scale=min(1,step/warmup)
            if step>.8*args.steps: scale*=1-.8*(step-.8*args.steps)/(.2*args.steps)
            for group in optimizer.param_groups: group['lr']=args.lr*scale
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            r=model(e,y)
            if not bool(torch.isfinite(r['loss'])): raise RuntimeError('Nonfinite training loss.')
            r['loss'].backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(gn)): raise RuntimeError('Nonfinite gradients.')
            optimizer.step();model.commit_feedback(r['packet']);seconds+=time.perf_counter()-tick
            recent.append(float(r['ce'].detach()))
            if step%args.eval_every==0 or step==args.steps:
                v=evaluate(model,val)
                row={'step':step,'bytes':step*args.batch*args.context,'training_loss_recent':sum(recent)/len(recent),
                     'validation_loss':v,'training_seconds':seconds,'elapsed_seconds':time.perf_counter()-started,
                     'lr':args.lr*scale,'gradient_norm':float(gn)}
                history.append(row);log.write(json.dumps(row)+'\n');log.flush();recent=[];print(json.dumps(row),flush=True)
    model.eval()
    summary={'initial_validation_loss_2048_bytes':initial,
        'final_validation_loss_8192_bytes':evaluate(model,val,16),
        'static_dictionary_validation_loss':evaluate(model,val,16,interface='static'),
        'live_link_validation_loss':evaluate(model,val,16,interface='live'),
        'frozen_no_hat_history_loss':evaluate(model,val,16,hat_memory=False),
        'frozen_no_expert_history_loss':evaluate(model,val,16,expert_memory=False),
        'frozen_no_history_loss':evaluate(model,val,16,hat_memory=False,expert_memory=False),
        'train_probe_loss':evaluate(model,train,16,seed=334),'training_bytes':args.steps*args.batch*args.context,
        'training_seconds':seconds,'training_bytes_per_second':args.steps*args.batch*args.context/seconds,
        'minibatch_sha256':stream_hash.hexdigest(),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'note':'Memory removals are frozen interventions, not separately trained controls; byte alphabet and input interface differ from previous character runs.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'routing.json').write_text(json.dumps(diagnose(model,val),indent=2))
    tokens,state=model.generate(ByteBatch.text('ROMEO:\n'),count=240)
    (out/'sample.txt').write_text(bytes(tokens[0].tolist()).decode('utf-8',errors='replace'))
    checkpoint={'config':asdict(model.cfg),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':args.steps,'torch_rng':torch.get_rng_state(),'batch_rng':gen.get_state(),'protocol':protocol,
                'sample_stream_state':state.detach().to_dict()}
    torch.save(checkpoint,out/'checkpoint.pt')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
