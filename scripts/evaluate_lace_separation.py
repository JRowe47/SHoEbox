"""Locked validation/final inputs and same-weight ablations; no credit capture."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.paired_recall import fork_state
from experiments.text_regions.memory_selectivity import selective_cases,summarize_values
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language
from scripts.lace_cohorts import from_record
from scripts.run_memory_selectivity import write_json


def load_model(path,control='intact'):
    saved=torch.load(path,weights_only=True,map_location='cpu')
    cfg=GraphConfig.from_dict(saved['config'])
    if control=='bias_off':cfg=replace(cfg,lace_separation=replace(cfg.lace_separation,enabled=False))
    if control=='blocked':cfg=replace(cfg,hippocampus=replace(cfg.hippocampus,control='blocked'))
    model=RegionLanguageModel(cfg);model.load_state_dict(saved['weights']);model.eval()
    return model,saved


@torch.no_grad()
def evaluate_pairs(model,records,control='intact'):
    rows=[];nll=0.;experts=positions=0;work={};started=time.perf_counter()
    for record in records:
        batch=from_record(record);prefix=model(batch['history'])
        if control=='cue_permuted':model._lace_cue_rows=torch.arange(len(batch['query']))^1
        try:branch=model(batch['query'],state=fork_state(prefix['state'],batch['rows']))
        finally:model._lace_cue_rows=None
        logits=branch['logits'][:,-1];targets=batch['answers'];pred=logits.argmax(-1)
        nll+=float(F.cross_entropy(logits,targets,reduction='sum'))
        experts+=prefix['expert_executions']+branch['expert_executions']
        positions+=batch['history'].numel()+batch['query'].numel()
        for w in prefix['hippocampus_work']+branch['hippocampus_work']:
            for k,v in w.items():
                if isinstance(v,(int,float)):work[k]=work.get(k,0)+v
        for i,(a,b) in enumerate(targets.reshape(-1,2).tolist()):
            x,y=pred[2*i:2*i+2].tolist()
            rows.append(dict(template=record['template'],older=a,newer=b,older_prediction=x,newer_prediction=y,
                older_margin=float(logits[2*i,a]-logits[2*i,b]),newer_margin=float(logits[2*i+1,b]-logits[2*i+1,a]),
                case_sha256=record['case_sha256'][2*i:2*i+2]))
    distinct=[r for r in rows if r['older']!=r['newer']];same=[r for r in rows if r['older']==r['newer']]
    both=lambda row:row['older']==row['older_prediction'] and row['newer']==row['newer_prediction']
    return dict(histories=len(rows),conflicting_pairs=len(distinct),both_correct=sum(map(both,distinct)),
        both_accuracy=sum(map(both,distinct))/len(distinct),query_switches=sum(r['older_prediction']!=r['newer_prediction'] for r in distinct),
        older_accuracy=sum(r['older']==r['older_prediction'] for r in rows)/len(rows),
        newer_accuracy=sum(r['newer']==r['newer_prediction'] for r in rows)/len(rows),
        equal_answer_histories=len(same),equal_answer_both_correct=sum(map(both,same)),nll=nll/(2*len(rows)),
        seconds=time.perf_counter()-started,executed_positions=positions,expert_executions=experts,
        memory_work=work,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort',choices=('validation','test'),required=True)
    p.add_argument('--part',choices=('pairs','unseen','retention','language'),required=True)
    p.add_argument('--control',choices=('intact','bias_off','cue_permuted','blocked'),default='intact')
    p.add_argument('--cohorts',type=Path,default=ROOT/'artifacts/lace_separation/cohorts.json')
    a=p.parse_args();torch.set_num_threads(1)
    if a.cohort=='test' and not (ROOT/'artifacts/lace_separation/training_complete.json').exists():
        raise ValueError('Final test remains sealed until every fixed endpoint is complete.')
    model,saved=load_model(a.checkpoint,a.control);lock=json.loads(a.cohorts.read_text())
    raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']:raise ValueError('Corpus mismatch.')
    heldout=split_bytes(raw)[1 if a.cohort=='validation' else 2]
    identity=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),cohorts_sha256=hashlib.sha256(a.cohorts.read_bytes()).hexdigest(),
        cohort=a.cohort,control=a.control,part=a.part,step=saved['step'],evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.out.exists():
        result=json.loads(a.out.read_text())
        if result['identity']!=identity:raise ValueError('Evaluation cache mismatch.')
    else:result=dict(identity=identity,measurements={})
    def record(key,value):
        result['measurements'][key]=value;a.out.parent.mkdir(parents=True,exist_ok=True);write_json(a.out,result)
        print(json.dumps(dict(measurement=key)),flush=True)
    with torch.no_grad():
        if a.part in ('pairs','unseen'):
            wording='native' if a.part=='pairs' else 'unseen'
            for gap in (0,32,64):
                key=f'{a.part}_{gap}'
                if key in result['measurements']:continue
                rows=[r for r in lock['cohorts'][a.cohort]['records'] if r['gap']==gap and r['wording']==wording]
                record(key,evaluate_pairs(model,rows,a.control))
        elif a.part=='retention':
            data=TaskStreamConfig.from_dict(saved['data_config']);r=lock['retention'];split=a.cohort
            gaps=r['validation_gaps'] if split=='validation' else [r['test_gap']]
            for mode in ('single','correction'):
                for gap in gaps:
                    key=f'{mode}_{gap}'
                    if key in result['measurements']:continue
                    cases=selective_cases(data,split=split,seed=r[f'{split}_seed'],start=r[f'{split}_start'],
                        count=r[f'{split}_{mode}_templates'],mode=mode,gap=gap,distractor_bytes=heldout)
                    scores=evaluate_values(model,cases,context=32,cases_per_batch=8)
                    record(key,summarize_values(scores)|dict(rows=scores['rows']))
        elif 'language' not in result['measurements']:
            record('language',evaluate_language(model,heldout,batch=4,context=32,chunks=2 if a.cohort=='validation' else 8))

if __name__=='__main__':main()
