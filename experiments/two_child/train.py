"""Matched Tiny Shakespeare shakedown for the three-SHoE hierarchy."""
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
from model import HierarchicalSHoE,HierarchyConfig


def load_data(path):
    raw=Path(path).read_text();a,b=int(.9*len(raw)),int(.95*len(raw))
    chars=sorted(set(raw[:a]));stoi={c:i for i,c in enumerate(chars)}
    data=torch.tensor([stoi[c] for c in raw[:b]],dtype=torch.long)
    return data[:a],data[a:b],chars,{'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'characters':len(raw),'train_characters':a,'validation_characters':b-a,
        'test_characters_untouched':len(raw)-b,'split':'contiguous 90/5/5; test unused',
        'tokenizer':'65-character vocabulary fitted on training split'}


def batch(data,size,context,generator):
    starts=torch.randint(len(data)-context,(size,),generator=generator)
    chunk=data[starts[:,None]+torch.arange(context+1)]
    return chunk[:,:-1],chunk[:,1:]


@torch.no_grad()
def evaluate(model,data,batches=16,size=4,seed=333,**kwargs):
    old=model.training;model.eval();gen=torch.Generator().manual_seed(seed)
    losses=[]
    for _ in range(batches):
        x,y=batch(data,size,model.cfg.context,gen)
        losses.append(float(model(x,y,**kwargs)['ce']))
    model.train(old)
    return sum(losses)/len(losses)


@torch.no_grad()
def diagnose(model,data,batches=8,reinforce=True):
    model.eval();gen=torch.Generator().manual_seed(5151)
    branch=torch.zeros(2,dtype=torch.long);probe_count=fast=total=shared_uses=expert_uses=0
    experts=[[torch.zeros(n,dtype=torch.long) for n in [342,341,341]] for _ in range(2)]
    parent_laces=torch.zeros(64,dtype=torch.long)
    child_laces=[torch.zeros(64,dtype=torch.long) for _ in range(2)]
    semantic=utility=0.;example=None
    for _ in range(batches):
        x,y=batch(data,4,model.cfg.context,gen)
        r=model(x,y,reinforce=reinforce,trace=True)
        route=r['route'];choice=route['choice']
        branch+=torch.bincount(choice.flatten(),minlength=2)
        total+=x.numel();probe_count+=int(route['probed'].sum());fast+=int(route['fast'].sum())
        parent_laces+=torch.bincount(route['parent_lace_ids'].flatten(),minlength=64)
        semantic+=float(r['semantic_loss']);utility+=float(r['utility_loss'])
        for which in range(2):
            child_laces[which]+=torch.bincount(r['executed_lace_ids'][choice==which].flatten(),minlength=64)
        for execution in r['executions']:
            which=execution['child']
            for layer,trace in enumerate(execution['trace']):
                ids=trace['experts'].flatten()
                experts[which][layer]+=torch.bincount(ids,minlength=experts[which][layer].numel())
                shared_uses+=int((ids<model.shared_counts[layer]).sum());expert_uses+=ids.numel()
        if example is None:
            example={'tokens':x[0].tolist(),'branch_choices':choice[0].tolist(),
                     'head_probes':route['probed'][0].tolist(),
                     'parent_laces':route['parent_lace_ids'][0].tolist(),
                     'executed_child_laces':r['executed_lace_ids'][0].tolist()}
    shared_unique=private_unique=0
    private_per_child=[0,0]
    for l,n in enumerate(model.shared_counts):
        shared_unique+=int(((experts[0][l][:n]+experts[1][l][:n])>0).sum())
        for which in range(2):private_per_child[which]+=int((experts[which][l][n:]>0).sum())
    private_unique=sum(private_per_child)
    return {'probe_characters':total,'child_token_counts':branch.tolist(),
            'child_token_fractions':(branch/total).tolist(),'head_probes':probe_count,
            'mean_child_head_probes_per_token':probe_count/total,'fast_path_fraction':fast/total,
            'expert_activations':expert_uses,'expected_expert_activations':total*48,
            'shared_expert_selection_fraction':shared_uses/expert_uses,
            'unique_shared_experts_used':shared_unique,'dedicated_experts_used_per_child':private_per_child,
            'unique_physical_experts_used':shared_unique+private_unique,
            'logical_experts_used_per_child_layer':[[int((x>0).sum()) for x in row] for row in experts],
            'parent_laces_used':int((parent_laces>0).sum()),
            'child_executed_laces_used':[int((x>0).sum()) for x in child_laces],
            'parent_semantic_feedback_loss':semantic/batches,'utility_prediction_loss':utility/batches,
            'feedback_updates':int(model.parent.feedback_updates),
            'route_memory_mass_mean':float(model.parent.route_mass.mean()),
            'expert_usage':[[x.tolist() for x in row] for row in experts],
            'parent_lace_usage':parent_laces.tolist(),'child_lace_usage':[x.tolist() for x in child_laces],
            'example':example}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/tiny_shakespeare.txt');p.add_argument('--out',default='results/full')
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--no-reinforcement',action='store_true');args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    train,val,chars,data=load_data(args.data)
    model=HierarchicalSHoE(HierarchyConfig(vocab_size=len(chars),seed=args.seed))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.01)
    enabled=not args.no_reinforcement
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    protocol={'args':vars(args),'config':asdict(model.cfg),'parameters':model.parameter_summary(),'data':data,
              'torch':str(torch.__version__),'python':platform.python_version(),
              'initial_parameter_sha256':hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest(),
              'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['model.py','semantic_core.py','train.py']},
              'optimizer':'AdamW, beta=(0.9,0.95), weight_decay=.01, clip_norm=1',
              'schedule':'warmup 40 steps; plateau to 80%; linear cooldown to 20% peak',
              'routing':'causal token-level hard top-1; selected-gate straight-through task gradient',
              'feedback':'executed lace KL + selected utility calibration + eligibility-weighted next-pass route memory',
              'control':'retains live bottom-up semantic bids and top-down messages; disables explicit reinforcement losses, persistent route memory, and fast-path reuse',
              'evaluation':'fixed sampled windows, seed333; 2048 periodic and 8192 final character presentations'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    initial=evaluate(model,val,batches=4,reinforce=enabled)
    print(json.dumps({'initial_validation':initial,'parameters':model.parameter_summary()}),flush=True)
    gen=torch.Generator().manual_seed(args.seed+1);stream=hashlib.sha256()
    training_seconds=0.;start=time.perf_counter();recent=[]
    with (out/'history.jsonl').open('w') as log:
        log.write(json.dumps({'step':0,'tokens':0,'validation_loss':initial})+'\n');log.flush()
        for step in range(1,args.steps+1):
            model.train();warmup=min(40,max(1,args.steps//10));scale=min(1,step/warmup)
            if step>.8*args.steps:scale*=1-.8*(step-.8*args.steps)/(.2*args.steps)
            for group in optimizer.param_groups:group['lr']=args.lr*scale
            x,y=batch(train,args.batch,model.cfg.context,gen);stream.update(x.numpy().tobytes())
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            r=model(x,y,reinforce=enabled)
            if not torch.isfinite(r['loss']):raise RuntimeError(f'nonfinite loss at {step}')
            r['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if enabled:model.commit_feedback(r['packet'])
            training_seconds+=time.perf_counter()-tick
            recent.append([float(r[k].detach()) for k in ['ce','semantic_loss','utility_loss']]
                          +[float(r['route']['fast'].float().mean()),float((r['route']['choice']==0).float().mean())])
            if step%args.eval_every==0 or step==args.steps:
                means=torch.tensor(recent).mean(0).tolist();recent=[]
                row={'step':step,'tokens':step*args.batch*model.cfg.context,'training_loss_recent':means[0],
                     'semantic_loss':means[1],'utility_loss':means[2],'fast_path_fraction':means[3],
                     'child_a_fraction':means[4],'validation_loss':evaluate(model,val,batches=4,reinforce=enabled),
                     'lr':args.lr*scale,'grad_norm':float(grad),'training_seconds':training_seconds,
                     'elapsed_seconds':time.perf_counter()-start}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    model.eval()
    final=evaluate(model,val,reinforce=enabled)
    summary={'initial_validation_loss':initial,'final_validation_loss':final,
             'final_train_probe_loss':evaluate(model,train,seed=334,reinforce=enabled),
             'forced_discovery_validation_loss':evaluate(model,val,reinforce=enabled,force_discovery=True),
             'no_upward_messages_validation_loss':evaluate(model,val,reinforce=enabled,force_discovery=True,disable_upward=True),
             'scrambled_upward_messages_validation_loss':evaluate(model,val,reinforce=enabled,force_discovery=True,scramble_upward=True),
             'feedback_memory_disabled_validation_loss':evaluate(model,val,reinforce=False),
             'training_tokens':args.steps*args.batch*model.cfg.context,'training_seconds':training_seconds,
             'training_characters_per_second':args.steps*args.batch*model.cfg.context/training_seconds,
             'character_perplexity':math.exp(final),'training_input_sha256':stream.hexdigest(),
             'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'trained_with_explicit_reinforcement':enabled}
    routing=diagnose(model,val,reinforce=enabled)
    prompt='ROMEO:\n';tokens=torch.tensor([[chars.index(c) for c in prompt]])
    sample=model.generate(tokens,count=240,reinforce=enabled)
    (out/'sample.txt').write_text(''.join(chars[i] for i in sample[0].tolist()))
    summary['total_seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'routing.json').write_text(json.dumps(routing,indent=2))
    torch.save({'config':asdict(model.cfg),'model':model.state_dict(),'chars':chars,'step':args.steps,
                'protocol':protocol,'optimizer':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
                'batch_rng_state':gen.get_state()},out/'checkpoint.pt')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
