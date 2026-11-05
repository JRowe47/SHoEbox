"""Byte predictions and state-isolation checks for multiple retained fields."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .multifact_data import make_batch, OWNERS, FIELD, KINDS
from .multifact import ingest, query, table_batches, COUNTERS, segment
from .ports import tensor_list
from .composition_data import batch_from_keys


def memory_tensors(state):
    return tensor_list(tuple((state.nodes[o],state.ingress[o],state.ports.latest[o]) for o in OWNERS))


@torch.no_grad()
def evaluate(model,split,seed,count=512,gap=16,kind='recall',source_control='intact',
             query_control='intact',rounds=2,feedback=True,policy='learned',shared_policy='latest',
             requests=1,batch_size=64,live=False,cache=None):
    records=[dict(predictions=[],targets=[],counterfactual_targets=[],old_targets=[],selected=[]) for _ in range(3)]
    before=[dict(predictions=[],targets=[]) for _ in range(3)]
    costs=dict.fromkeys(COUNTERS,0);exact=True;nll=[0.,0.,0.]
    for start in range(0,count,batch_size):
        batch=make_batch(split,start,min(batch_size,count-start),seed,gap,kind,source_control)
        cache_key=(id(model),split,seed,count,gap,kind,source_control,policy,batch_size,start,live)
        if cache is not None and cache_key in cache:
            state,budget=cache[cache_key]
        else:
            state,budget=ingest(model,batch,policy=policy,end=batch.boundary if live else None)
            if cache is not None:cache[cache_key]=(state,budget)
        for k in costs:costs[k]+=budget[k]
        if live:
            old=replace(batch,answers=batch.old_answers)
            for field,tb in enumerate(table_batches(old,split,start,seed,77)):
                saved=[x.clone() for x in memory_tensors(state)]
                r=query(model,tb.first,tb.second,state,torch.full_like(batch.answers[:,0],field))
                before[field]['predictions'].extend(r['logits'][:,-1].argmax(-1).tolist())
                before[field]['targets'].extend(tb.answers.tolist());state=r['state']
                exact &= all(torch.equal(a,b) for a,b in zip(saved,memory_tensors(state)))
                for k in costs:costs[k]+=r[k]
            state,budget=ingest(model,batch,state,policy,start=batch.boundary)
            for k in costs:costs[k]+=budget[k]
        for request in range(requests):
            for field,tb in enumerate(table_batches(batch,split,start,seed,request)):
                snapshot=[x.clone() for x in memory_tensors(state)]
                result=query(model,tb.first,tb.second,state,torch.full_like(batch.answers[:,0],field),
                             rounds,feedback,query_control,shared_policy)
                logits=result['logits'][:,-1];row=records[field]
                row['predictions'].extend(logits.argmax(-1).tolist());row['targets'].extend(tb.answers.tolist())
                row['selected'].extend(result['selected_owner'].tolist())
                cf=batch_from_keys([(a,b,int(q)-65) for (a,b,_),q in zip(tb.keys,batch.answers[:,field].roll(1))])
                row['counterfactual_targets'].extend(cf.answers.tolist())
                old=batch_from_keys([(a,b,int(q)-65) for (a,b,_),q in zip(tb.keys,batch.old_answers[:,field])])
                row['old_targets'].extend(old.answers.tolist())
                nll[field]+=float(F.cross_entropy(logits,tb.answers,reduction='sum'))
                if query_control in ('intact','no_edge'):exact &= all(torch.equal(a,b) for a,b in zip(snapshot,memory_tensors(result['state'])))
                # Each intervention starts from the same retained contents;
                # shuffled-memory controls must not accumulate permutations.
                if query_control=='intact':state=result['state']
                for k in costs:costs[k]+=result[k]
    for field,row in enumerate(records):
        row['accuracy']=sum(a==b for a,b in zip(row['predictions'],row['targets']))/len(row['targets'])
        row['counterfactual_accuracy']=sum(a==b for a,b in zip(row['predictions'],row['counterfactual_targets']))/len(row['targets'])
        row['old_accuracy']=sum(a==b for a,b in zip(row['predictions'],row['old_targets']))/len(row['targets'])
        row['nats']=nll[field]/len(row['targets'])
    for row in before:
        if row['targets']:row['accuracy']=sum(a==b for a,b in zip(row['predictions'],row['targets']))/len(row['targets'])
    return dict(fields=records,before=before,exact_memories_preserved=exact if query_control in ('intact','no_edge') else None,split=split,seed=seed,count=count,
        gap=gap,kind=kind,source_control=source_control,query_control=query_control,rounds=rounds,
        feedback=feedback,policy=policy,shared_policy=shared_policy,requests=requests,batch_size=batch_size,live=live,**costs)


@torch.no_grad()
def validation(model,seed,count=64):
    scores=[evaluate(model,'validation',seed,count,16,kind) for kind in KINDS]
    acc=[r['accuracy'] for score in scores for r in score['fields']]
    nll=[r['nats'] for score in scores for r in score['fields']]
    return (min(acc),sum(acc)/len(acc),-sum(nll)/len(nll)),scores


@torch.no_grad()
def publication_metrics(model,seed,count=256,gap=16):
    records={o:dict(tp=0,fp=0,fn=0,hold_pairs=0,exact_hold_pairs=0) for o in OWNERS}
    for start in range(0,count,64):
        batch=make_batch('test',start,min(64,count-start),seed,gap,KINDS[1+(start//64)%4])
        state=None;payloads={o:[] for o in OWNERS};writes={o:[] for o in OWNERS}
        for offset in range(0,batch.inputs.shape[1],16):
            r=segment(model,batch,offset,min(offset+16,batch.inputs.shape[1]),state,True);state=r['state'].detach()
            for o in OWNERS:
                payloads[o].append(r['node_activity'][o]['payload']);writes[o].append(r['publication_activity'][o]['written'][:,:,0])
        for o in OWNERS:
            predicted=torch.cat(writes[o],1);truth=batch.writes[o];row=records[o]
            row['tp']+=int((predicted&truth).sum());row['fp']+=int((predicted&~truth).sum());row['fn']+=int((~predicted&truth).sum())
            held=~predicted[:,1:]&~batch.reset[:,1:];payload=torch.cat(payloads[o],1)
            equal=payload[:,1:].eq(payload[:,:-1]).all(-1)
            row['hold_pairs']+=int(held.sum());row['exact_hold_pairs']+=int((held&equal).sum())
    for row in records.values():
        row['precision']=row['tp']/max(1,row['tp']+row['fp']);row['recall']=row['tp']/max(1,row['tp']+row['fn'])
    return records
