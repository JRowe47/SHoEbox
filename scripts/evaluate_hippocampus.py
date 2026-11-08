"""Locked held-out banks and causal inference interventions for the H3 pilot."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig,make_documents
from experiments.text_regions.memory_selectivity import selective_cases,summarize_values
from experiments.text_regions.paired_recall import paired_batch,paired_forward
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language,evaluate_memory
from scripts.run_memory_selectivity import write_json


@torch.no_grad()
def evaluate_pairs(model,data,raw,*,split='test',seed=1201,start=80000,count=16,gap=0):
    rows=[]; total_nll=0.; work=[]
    for template in range(count):
        batch=paired_batch(data,split=split,seed=seed,start=start+template,count=1,gap=gap,distractor_bytes=raw)
        result=paired_forward(model,batch)
        logits=result['logits'][:,-1]
        answers=batch['answers']; predictions=logits.argmax(-1)
        nll=F.cross_entropy(logits,answers,reduction='sum')
        total_nll+=float(nll)
        for i in range(len(answers)//2):
            a,b=answers[2*i:2*i+2].tolist(); x,y=predictions[2*i:2*i+2].tolist()
            rows.append(dict(template=start+template,older=a,newer=b,older_prediction=x,newer_prediction=y))
    conflicting=[r for r in rows if r['older']!=r['newer']]
    sensitivity={}
    for direction,other in (('older','newer'),('newer','older')):
        groups={}
        for r in rows: groups.setdefault((r['template'],r[other]),[]).append(r[direction+'_prediction'])
        sensitivity[direction]=dict(groups=len(groups),source_switches=sum(len(set(v))>1 for v in groups.values()))
    return dict(histories=len(rows),conflicting_pairs=len(conflicting),
        both_correct=sum(r['older']==r['older_prediction'] and r['newer']==r['newer_prediction'] for r in conflicting),
        both_accuracy=sum(r['older']==r['older_prediction'] and r['newer']==r['newer_prediction'] for r in conflicting)/len(conflicting),
        older_accuracy=sum(r['older']==r['older_prediction'] for r in rows)/len(rows),
        newer_accuracy=sum(r['newer']==r['newer_prediction'] for r in rows)/len(rows),
        query_switches=sum(r['older_prediction']!=r['newer_prediction'] for r in conflicting),
        nll=total_nll/(2*len(rows)),source_sensitivity=sensitivity,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--part',choices=('full','pairs','single','correction','streams','language'),default='full')
    p.add_argument('--control',choices=('intact','blocked','shuffled','pooled','no_refinement','no_feedback'),default='intact')
    p.add_argument('--split',choices=('validation','test'),default='test')
    p.add_argument('--seed',type=int)
    p.add_argument('--start',type=int)
    p.add_argument('--pair-count',type=int)
    p.add_argument('--single-count',type=int)
    p.add_argument('--gaps',type=int,nargs='+')
    p.add_argument('--cue-clock',choices=('round','event'))
    p.add_argument('--reinstatement',choices=('ingress','expert'))
    a=p.parse_args(); torch.set_num_threads(1)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    raw=a.data.read_bytes(); train,validation,test=split_bytes(raw)
    if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']: raise ValueError('Corpus mismatch.')
    cfg=GraphConfig.from_dict(saved['config']); data=TaskStreamConfig.from_dict(saved['data_config'])
    if a.control!='intact' and not cfg.hippocampus.enabled: raise ValueError('Control requires enabled hippocampus.')
    cfg=replace(cfg,hippocampus=replace(cfg.hippocampus,control=a.control))
    intervention={key:value for key,value in dict(cue_clock=a.cue_clock,reinstatement=a.reinstatement).items()
                  if value is not None}
    if intervention:
        if not cfg.hippocampus.enabled: raise ValueError('Path intervention requires enabled hippocampus.')
        cfg=replace(cfg,hippocampus=replace(cfg.hippocampus,**intervention))
    model=RegionLanguageModel(cfg); model.load_state_dict(saved['weights']); model.eval()
    heldout=validation if a.split=='validation' else test
    seed,start=(683,3000) if a.split=='validation' else (1201,80000)
    pair_count=8 if a.split=='validation' else 16
    single_count=8 if a.split=='validation' else 32
    seed=seed if a.seed is None else a.seed
    start=start if a.start is None else a.start
    pair_count=pair_count if a.pair_count is None else a.pair_count
    single_count=single_count if a.single_count is None else a.single_count
    gaps=((0,32) if a.split=='validation' else (0,32,64)) if a.gaps is None else tuple(a.gaps)
    if start<0 or min(pair_count,single_count)<1 or not gaps or min(gaps)<0:
        raise ValueError('Invalid cohort or delay budget.')
    identity=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),step=saved['step'],
        raw_sha256=saved['raw_sha256'],seed=seed,start=start,pair_templates=pair_count,single_templates=single_count,
        gaps=list(gaps),control=a.control,part=a.part,split=a.split,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if intervention: identity['path_intervention']=intervention
    if a.out.exists():
        result=json.loads(a.out.read_text())
        if result['identity']!=identity: raise ValueError('Evaluation cache identity mismatch.')
    else: result=dict(identity=identity,measurements={})
    def record(name,value):
        result['measurements'][name]=value; a.out.parent.mkdir(parents=True,exist_ok=True); write_json(a.out,result)
        print(json.dumps({'measurement':name,'summary':{k:v for k,v in value.items() if k!='rows'}}),flush=True)
    if a.part in ('pairs','full'):
        for gap in gaps:
            name=f'pairs_{gap}'
            if name not in result['measurements']:
                record(name,evaluate_pairs(model,data,heldout,split=a.split,seed=seed,start=start,count=pair_count,gap=gap))
    for mode in ('single','correction'):
        if a.part not in (mode,'full'): continue
        for gap in gaps:
            name=f'{mode}_{gap}'
            if name not in result['measurements']:
                cases=selective_cases(data,split=a.split,seed=seed,start=start,
                    count=single_count if mode=='single' else pair_count,mode=mode,gap=gap,distractor_bytes=heldout)
                record(name,summarize_values(evaluate_values(model,cases,context=32,cases_per_batch=8)))
    if a.part in ('streams','language','full'):
        if 'language' not in result['measurements']:
            record('language',evaluate_language(model,heldout,batch=4,context=32,chunks=8))
        for split in (() if a.part=='language' else ('test','long_delay')):
            if split not in result['measurements']:
                docs=make_documents(data,split=split,seed=seed,count=8,start=start,distractor_bytes=heldout)
                metrics,traces=evaluate_memory(model,data,seed,8,32,split,documents=docs)
                record(split,metrics)
                (a.out.parent/(a.out.stem+'_'+split+'_traces.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in traces))

if __name__=='__main__': main()
