"""Continue regional learning with preserved optimizer/RNG and capability gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .composition import encode_sources, answer_query
from .composition_data import batch_from_keys, evaluation_keys, training_batch
from .composition_train import load_checkpoint, save_checkpoint, source_hashes, update


def completed_updates(saved):
    return saved['protocol'].get('optimizer_step_offset',0)+saved['step']


@torch.no_grad()
def evaluate_progress(model, split='validation', batch_size=64):
    """One pass over unique cases; local targets are only used for scoring."""
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    if not hasattr(model,'regional_readout') or not hasattr(model,'source_reconstruction'):
        raise ValueError('Progress scoring requires the existing training-only heads.')
    keys=evaluation_keys(split)
    names=('overall','lookup_b','lookup_c','composition','p_first','q_second',
           'prefix_b','prefix_c','query_b','query_c')
    records={name:dict(correct=0,count=0) for name in names}
    nll=0.;was_training=model.training;model.eval()
    try:
        for start in range(0,len(keys),batch_size):
            batch=batch_from_keys(keys[start:start+batch_size])
            prefix=encode_sources(model,batch)
            query=answer_query(model,batch,prefix['state'],2,return_round_activity=True)
            logits=query['logits'][:,-1]
            correct=logits.argmax(-1).eq(batch.answers)
            nll+=float(F.cross_entropy(logits,batch.answers,reduction='sum'))
            masks={'overall':torch.ones(batch.size,dtype=torch.bool),
                   'lookup_b':torch.tensor([a!=0 and b==0 for a,b,q in batch.keys]),
                   'lookup_c':torch.tensor([a==0 and b!=0 for a,b,q in batch.keys]),
                   'composition':torch.tensor([a!=0 and b!=0 for a,b,q in batch.keys])}
            for name,mask in masks.items():
                records[name]['correct']+=int(correct[mask].sum())
                records[name]['count']+=int(mask.sum())
            intermediate=batch.first.gather(1,batch.query-65).flatten()-65
            for name,node,k,target in [('p_first','P',0,intermediate),
                                       ('q_second','Q',1,batch.answers-65)]:
                values=query['round_activity'][node]['payload'][:,0,k]
                pred=model.regional_readout(values).argmax(-1)
                records[name]['correct']+=int(pred.eq(target).sum())
                records[name]['count']+=batch.size
            for stage,result in [('prefix',prefix),('query',query)]:
                for node,target in [('B',batch.first),('C',batch.second)]:
                    values=result['state'].ports.latest[node].payload
                    pred=model.source_reconstruction(values).reshape(-1,4,4).argmax(-1)
                    correct_table=pred.eq(target-65).all(-1)
                    row=records[stage+'_'+node.lower()]
                    row['correct']+=int(correct_table.sum());row['count']+=batch.size
        for row in records.values():
            row['accuracy']=row['correct']/row['count'] if row['count'] else None
        return {'metrics':records,'nats_per_answer':nll/len(keys),'unique_cases':len(keys)}
    finally:
        model.train(was_training)


def gate_failures(progress, threshold=.9):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Threshold must be finite and in (0,1].')
    failures=[]
    for name in ('overall','lookup_b','lookup_c','composition','p_first','q_second',
                 'prefix_b','prefix_c','query_b','query_c'):
        row=progress['metrics'].get(name,{})
        value=row.get('accuracy')
        needed=1. if name.startswith(('prefix_','query_')) else threshold
        if value is None or not math.isfinite(value) or value < needed:
            failures.append(name)
    return failures


def record_validation(progress, phase_step, counters, threshold, patience):
    failures=gate_failures(progress,threshold)
    counters['gate_streak']=0 if failures else counters['gate_streak']+1
    if counters['gate_streak']>=patience and counters['mastered_step'] is None:
        counters['mastered_step']=phase_step
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--patience',type=int,default=2)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--stop-on-mastery',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.validate_every,args.save_every,args.patience,args.threads)<1:
        p.error('Counts must be positive.')
    if not math.isfinite(args.threshold) or not 0 < args.threshold <= 1:
        p.error('Threshold must be in (0,1].')
    if args.stop_after is not None and args.stop_after<1:p.error('Stop step must be positive.')
    torch.set_num_threads(args.threads)
    model,optimizer,rng,parent=load_checkpoint(args.parent)
    old=parent['protocol']
    if old['rounds']!=2 or not old['feedback'] or not old.get('regional_supervision') or not old.get('source_reconstruction'):
        p.error('Parent must be a two-round model with both regional and source objectives.')
    protocol={key:old[key] for key in ('lr','warmup','rounds','feedback','source_reconstruction','regional_supervision','batch')}
    protocol.update({k:v for k,v in vars(args).items() if k not in ('out','parent','resume','stop_after')})
    protocol.update(parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                    parent_source_hashes=old.get('source_hashes',{}),source_hashes=source_hashes(),
                    optimizer_step_offset=completed_updates(parent),
                    optimizer_continuation='Preserve all moments, per-parameter step counters and RNG states',
                    selection='Lowest validation NLL meeting every capability gate; fallback candidate explicitly labeled',
                    source_gate='100 percent exact tables before and after query',
                    curriculum='Continue the full original training-pair stream without resetting the sampler')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        model,optimizer,rng,saved=load_checkpoint(args.out/'last.pt',protocol)
        start,best,counters,history=saved['step'],saved['best_validation'],saved['counters'],saved['history']
    else:
        if (args.out/'initial.pt').exists():p.error('Output exists; use --resume or a new directory.')
        start,best,history=0,float('inf'),[]
        counters=dict(episodes=0,expert_executions=0,source_occurrences=0,training_seconds=0.,
                      gate_streak=0,mastered_step=None,best_candidate_nats=float('inf'))
        progress=evaluate_progress(model)
        failures=record_validation(progress,0,counters,args.threshold,args.patience)
        history.append({'phase_step':0,'step':protocol['optimizer_step_offset'],
                        'validation':progress,'gate_failures':failures})
        counters['best_candidate_nats']=progress['nats_per_answer']
        save_checkpoint(args.out/'initial.pt',model,optimizer,rng,0,protocol,best,counters,history)
        save_checkpoint(args.out/'best_candidate.pt',model,optimizer,rng,0,protocol,best,counters,history)
        if not failures:
            best=progress['nats_per_answer']
            save_checkpoint(args.out/'best.pt',model,optimizer,rng,0,protocol,best,counters,history)
        print(json.dumps({'phase_step':0,'validation':progress,'gate_failures':failures}),flush=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    stop=min(args.steps,args.stop_after) if args.stop_after else args.steps
    finished=start
    model.train()
    for phase in range(start,stop):
        if args.stop_on_mastery and counters['mastered_step'] is not None:break
        tick=time.perf_counter()
        batch=training_batch(rng,protocol['batch'])
        row=update(model,optimizer,batch,protocol['optimizer_step_offset']+phase,protocol)
        finished=phase+1;row.update(phase_step=finished,seconds=time.perf_counter()-tick)
        counters['episodes']+=batch.size;counters['training_seconds']+=row['seconds']
        for key in ('expert_executions','source_occurrences'):counters[key]+=row[key]
        history.append(row)
        if phase==start or finished%25==0:print(json.dumps(row),flush=True)
        if finished%args.validate_every==0 or finished==args.steps:
            progress=evaluate_progress(model)
            failures=record_validation(progress,finished,counters,args.threshold,args.patience)
            row.update(validation=progress,gate_failures=failures)
            nats=progress['nats_per_answer']
            if nats<counters['best_candidate_nats']:
                counters['best_candidate_nats']=nats
                save_checkpoint(args.out/'best_candidate.pt',model,optimizer,rng,finished,protocol,best,counters,history)
            if not failures and nats<best:
                best=nats
                save_checkpoint(args.out/'best.pt',model,optimizer,rng,finished,protocol,best,counters,history)
            if counters['mastered_step']==finished:
                save_checkpoint(args.out/'mastered.pt',model,optimizer,rng,finished,protocol,best,counters,history)
            print(json.dumps({'phase_step':finished,'validation':progress,'gate_failures':failures,
                              'gate_streak':counters['gate_streak']}),flush=True)
        save_now=(finished%args.save_every==0 or finished==stop or
                  (args.stop_on_mastery and counters['mastered_step'] is not None))
        if save_now:
            save_checkpoint(args.out/'last.pt',model,optimizer,rng,finished,protocol,best,counters,history)
            (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    selected=args.out/('best.pt' if (args.out/'best.pt').exists() else 'best_candidate.pt')
    selection=dict(finished_step=finished,optimizer_step=protocol['optimizer_step_offset']+finished,
                   mastered_step=counters['mastered_step'],qualified_checkpoint=selected.name=='best.pt',
                   selected_checkpoint=selected.name,selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
                   **{k:v for k,v in counters.items() if k!='mastered_step'})
    (args.out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(selection),flush=True)


if __name__=='__main__':main()
