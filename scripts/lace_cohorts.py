"""Lock rendered paired inputs before training; audit actual byte identities."""
import hashlib,json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.paired_recall import paired_batch
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.data import split_bytes
from scripts.run_memory_selectivity import write_json


def digest(values,answers):
    return hashlib.sha256(bytes(values)+b'\0target\0'+bytes(answers)).hexdigest()


def render(data,raw,split,seed,start,template,gap,wording='native'):
    b=paired_batch(data,split=split,seed=seed,start=start+template//2,count=1,gap=gap,distractor_bytes=raw)
    if template%2:
        a,c=b['history'][0,0].item(),b['history'][0,6].item()
        for key in ('history','query'):
            x=b[key];b[key]=torch.where(x==a,c,torch.where(x==c,a,x))
    # A natural-text lead-in makes full cases new despite the finite compact
    # grammar. It precedes both facts and does not add to the retention delay.
    rng=random.Random(f'lace-prefix:{split}:{seed}:{start}:{template}')
    offset=rng.randrange(len(raw)-24)
    prefix=torch.tensor(list(raw[offset:offset+24])).expand(len(b['history']),-1)
    b['history']=torch.cat((prefix,b['history']),1)
    if wording=='unseen':
        lead=torch.tensor(list(b'Value of ')).expand(len(b['query']),-1)
        b['query']=torch.cat((lead,b['query']),1)
    b['full']=torch.cat((b['history'].index_select(0,b['rows']),b['query']),1)
    b['syntax']=torch.full_like(b['full'],-100);b['syntax'][:,:-1]=b['full'][:,1:]
    return b


def from_record(record):
    b={k:torch.tensor(record[k],dtype=torch.long) for k in ('history','query','answers')}
    b['rows']=torch.arange(len(b['history'])).repeat_interleave(2)
    b['full']=torch.cat((b['history'].index_select(0,b['rows']),b['query']),1)
    b['syntax']=torch.full_like(b['full'],-100);b['syntax'][:,:-1]=b['full'][:,1:]
    return b


def main():
    raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes();train,val,test=split_bytes(raw)
    saved=torch.load(ROOT/'artifacts/hippocampus_binding/ca1/bounded/last.pt',weights_only=True,map_location='cpu')
    data=TaskStreamConfig.from_dict(saved['data_config'])
    # Reconstruct available prior paired evaluation banks, including diagnostics.
    banks=[('validation',683,2000,8),('validation',683,3000,8),('validation',881,4000,2),
           ('test',1031,30000,16),('test',917,20000,16),('train',683,40000,16),('validation',683,50000,16),('validation',683,60000,1),('test',1201,80000,16),('test',1511,120000,16),('test',1699,160000,8)]
    previous=set()
    for split,seed,start,count in banks:
        for gap in (0,24,32,64):
            for t in range(count):
                b=paired_batch(data,split=split,seed=seed,start=start+t,count=1,gap=gap,distractor_bytes=train if split=='train' else val if split=='validation' else test)
                for x,y in zip(b['full'].tolist(),b['answers'].tolist()):previous.add(digest(x,[y]))
    # All available training document identities up to and including this pilot.
    for index in range(saved['next_document']+56):
        for gap in (0,4,8,16,32):
            b=paired_batch(data,split='train',seed=683,start=index,count=1,gap=gap,distractor_bytes=train)
            for x,y in zip(b['full'].tolist(),b['answers'].tolist()):previous.add(digest(x,[y]))
    result=dict(schema=1,raw_sha256=hashlib.sha256(raw).hexdigest(),
        primary='Fixed update 32; native wording; distinct targets; delays 0/32/64',
        prefix_bytes=24,cohorts={},retention=dict(validation_seed=683,validation_start=3000,
        validation_single_templates=8,validation_correction_templates=8,validation_gaps=[0,32],
        test_seed=1699,test_start=160000,test_single_templates=16,test_correction_templates=8,test_gap=64,
        validation_language_targets=256,test_language_targets=1024),
        audit=dict(previous_rendered_answers=len(previous),prior_banks=banks,
        training_document_end_exclusive=saved['next_document']+56))
    identities={};all_seen=set();native_duplicates=0
    for name,split,seed,start,count,heldout in [('diagnostic','validation',2309,180000,2,val),
            ('validation','validation',2411,200000,8,val),('test','test',2539,240000,16,test)]:
        records=[];hashes=set()
        for wording in ('native','unseen') if name!='diagnostic' else ('native',):
            for gap in (0,32,64) if name!='diagnostic' else (0,32):
                for t in range(count):
                    batch=render(data,heldout,split,seed,start,t,gap,wording)
                    case_hashes=[digest(x,[y]) for x,y in zip(batch['full'].tolist(),batch['answers'].tolist())]
                    assert not (set(case_hashes)&previous),'Historical byte overlap'
                    assert not (set(case_hashes)&all_seen),'Cohort byte overlap'
                    all_seen.update(case_hashes);hashes.update(case_hashes)
                    records.append(dict(template=t,gap=gap,wording=wording,case_sha256=case_hashes,
                        **{k:batch[k].tolist() for k in ('history','query','answers')}))
                    if wording=='native':
                        native_duplicates+=sum(digest(x[24:],[y]) in previous for x,y in zip(batch['full'].tolist(),batch['answers'].tolist()))
        result['cohorts'][name]=dict(split=split,seed=seed,start=start,templates=count,records=records,
            sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())
        identities[name]=hashes
    result['audit'].update(historical_intersections=0,cross_cohort_intersections=0,
        unprefixed_native_answers_repeating_history=native_duplicates,
        source_orders='Every base template plus its entity-order reversal; full 4x4 value product',
        limitation='Historical audit covers reconstructed available compact-task ledgers; semantic fact combinations necessarily repeat.')
    out=ROOT/'artifacts/lace_separation/cohorts.json';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Cohort lock already exists.')
    write_json(out,result)
    print(json.dumps(result['audit']),flush=True)

if __name__=='__main__':main()
