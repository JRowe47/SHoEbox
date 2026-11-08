"""Compact paired-fact, retention, language and one-source acceptance set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.memory_selectivity import selective_cases,summarize_values
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language
from experiments.text_regions.paired_recall import fork_state
from scripts.lace_cohorts import render,digest,from_record
from scripts.run_memory_selectivity import write_json


@torch.no_grad()
def pairs(model, record):
    batch=from_record(record);started=time.perf_counter()
    prefix=model(batch['history']);state=fork_state(prefix['state'],batch['rows'])
    result=model(batch['query'],state=state)
    logits=result['logits'][:,-1];targets=batch['answers'];prediction=logits.argmax(-1)
    correct=(prediction==targets).reshape(-1,2);different=targets.reshape(-1,2).diff(dim=-1).squeeze(1).ne(0)
    counts={}
    for work in prefix['hippocampus_work']+result['hippocampus_work']:
        for key,value in work.items():counts[key]=counts.get(key,0)+value
    metrics=dict(histories=len(correct),conflicting=int(different.sum()),both_correct=int(correct[different].all(-1).sum()),
        answer_accuracy=float(correct.float().mean()),nll=float(F.cross_entropy(logits,targets)),
        predictions=prediction.tolist(),targets=targets.tolist(),memory_work=counts,seconds=time.perf_counter()-started)
    if model.memory_interface is not None and record['gap']==0:
        model.memory_interface.blocked_sources=('A',)
        try:blocked=model(batch['query'],state=state)['logits'][:,-1]
        finally:model.memory_interface.blocked_sources=()
        metrics['source_intervention']=dict(source='A',scope='new context only; identical shared history',
            max_logit_change=float((blocked-logits).abs().max()),changed_answers=int((blocked.argmax(-1)!=prediction).sum()),
            blocked_nll=float(F.cross_entropy(blocked,targets)))
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cohorts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--lock',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    saved=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    data=TaskStreamConfig.from_dict(saved['data_config']);raw=a.data.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']:raise ValueError('Corpus mismatch.')
    heldout=split_bytes(raw)[1]
    if a.lock:
        if a.cohorts.exists():raise ValueError('Cohorts already locked.')
        records=[]
        for gap in (0,32,64):
            batch=render(data,heldout,'validation',2789,280000,0,gap)
            records.append(dict(gap=gap,template=0,
                case_sha256=[digest(x,[y]) for x,y in zip(batch['full'].tolist(),batch['answers'].tolist())],
                **{k:batch[k].tolist() for k in ('history','query','answers')}))
        write_json(a.cohorts,dict(schema=1,raw_sha256=saved['raw_sha256'],records=records,
            retention=dict(seed=2789,start=280010,gap=64,single_count=1,correction_count=1),
            language=dict(batch=4,context=32,chunks=1),
            purpose='Compact integration acceptance; not the full prior promotion bank'))
    lock=json.loads(a.cohorts.read_text())
    if lock['raw_sha256']!=saved['raw_sha256']:raise ValueError('Cohort corpus mismatch.')
    model=RegionLanguageModel(GraphConfig.from_dict(saved['config']));model.load_state_dict(saved['weights']);model.eval()
    identity=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        cohorts_sha256=hashlib.sha256(a.cohorts.read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),step=saved['step'])
    result=json.loads(a.out.read_text()) if a.out.exists() else dict(identity=identity,measurements={})
    if result['identity']!=identity:raise ValueError('Evaluation cache mismatch.')
    def record(name, value):
        result['measurements'][name]=value;write_json(a.out,result)
        print(json.dumps(dict(measurement=name)),flush=True)
    for case in lock['records']:
        name=f"pairs_{case['gap']}"
        if name not in result['measurements']:record(name,pairs(model,case))
    r=lock['retention']
    for mode in ('single','correction'):
        name=f'{mode}_64'
        if name in result['measurements']:continue
        cases=selective_cases(data,split='validation',seed=r['seed'],start=r['start'],
            count=r[f'{mode}_count'],mode=mode,gap=r['gap'],distractor_bytes=heldout)
        score=evaluate_values(model,cases,context=32,cases_per_batch=4)
        record(name,summarize_values(score)|dict(rows=score['rows']))
    if 'language' not in result['measurements']:
        record('language',evaluate_language(model,heldout,**lock['language']))


if __name__=='__main__':main()
