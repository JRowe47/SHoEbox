"""Resumable held-out evaluation using the existing integrated controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.text_regions.composition import evaluate as composition_evaluate
from experiments.text_regions.composition_train import source_hashes
from experiments.text_regions.composition_continue import evaluate_progress, gate_failures
from experiments.text_regions.integrated_eval import compact, evaluate_views
from experiments.text_regions.integrated_train import load_integrated
from experiments.text_regions.probe_train import evaluate_probe


def assemble_capabilities(report):
    """Reuse identical scored cohorts instead of executing them a second time."""
    composition=report['composition_progress']
    memory={kind:{**row,'controls':{'carried':row['controls']['carried']}}
            for kind,row in report['memory_controls']['16'].items()}
    views=report['heldout_views']
    supported={name:report['independent_views'][name] for name in ('a_only','b_only','c_only')}
    scores={'composition':composition['metrics']['overall']['accuracy']}
    nats=[composition['nats_per_answer']]
    for kind in memory:
        row=memory[kind]['controls']['carried']
        scores[kind]=row['exact_accuracy'];nats.append(row['answer_nats_per_byte'])
    for kind,row in views['all'].items():
        scores[kind]=row['exact_accuracy'];nats.append(row['nats_per_answer'])
    for control,kinds in (('a_only',('shared','left')),('b_only',('shared',)),('c_only',('shared','right'))):
        for kind in kinds:
            row=supported[control][kind]
            scores[control+'/'+kind]=row['exact_accuracy'];nats.append(row['nats_per_answer'])
    failures=['composition/'+key for key in gate_failures(composition)]
    failures += [key for key,score in scores.items() if not math.isfinite(score) or score<.9]
    return dict(composition=composition,memory=compact(memory),views=compact(views),
                supported_sources=compact(supported),scores=scores,
                minimum_accuracy=min(scores.values()),macro_accuracy=sum(scores.values())/len(scores),
                macro_nats=sum(nats)/len(nats),gate_failures=failures,
                composition_preserved=not gate_failures(composition))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=512);p.add_argument('--resume',action='store_true')
    p.add_argument('--batch',type=int,default=256)
    args=p.parse_args()
    if args.count<64 or args.count%64:p.error('Use complete 64-case factorial cycles.')
    if args.batch<1:p.error('Batch size must be positive.')
    torch.set_num_threads(1)
    model,_,_,_,saved=load_integrated(args.checkpoint);model.eval()
    seed=saved['protocol']['seed']
    manifest={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'count':args.count,'batch':args.batch,'source_hashes':source_hashes(),
              'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.resume:
        report=json.loads(args.out.read_text())
        if report['_manifest'] != manifest:raise ValueError('Evaluation resume requires the same checkpoint, count and source.')
    else:
        if args.out.exists():p.error('Output exists; use --resume.')
        report={'checkpoint_step':saved['step'],'parameters':model.parameter_summary(),
                '_manifest':manifest,'_completed':[],'_seconds':{},
                'memory_controls':{},'independent_views':{}}
    def stage(name, callback, destination, key):
        if name in report['_completed']:return
        tick=time.perf_counter();destination[key]=callback()
        report['_completed'].append(name);report['_seconds'][name]=time.perf_counter()-tick
        args.out.parent.mkdir(parents=True,exist_ok=True)
        temp=args.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out)
        print(json.dumps({'completed':name,'seconds':report['_seconds'][name]}),flush=True)
    stage('composition_progress',lambda:evaluate_progress(model,'test'),report,'composition_progress')
    stage('composition',lambda:composition_evaluate(model,split='test',count=None,controls=True,
                                                    rounds=(1,2,4)),report,'composition_controls')
    for gap in (16,32):
        report['memory_controls'].setdefault(str(gap),{})
        for kind in ('recall','correction'):
            stage(f'memory/{gap}/{kind}',lambda:evaluate_probe(model,kind=kind,gap=gap,count=args.count,
                    batch_size=args.batch,segment=16,seed=seed,split='test'),report['memory_controls'][str(gap)],kind)
    stage('heldout_views',lambda:evaluate_views(model,'test',args.count,batch_size=args.batch,seed=seed),report,'heldout_views')
    for control in ('all','a_only','b_only','c_only','no_private','reset_query'):
        stage('independent/'+control,lambda:evaluate_views(model,'independent',args.count,batch_size=args.batch,seed=seed+104729,
              controls=(control,))[control],report['independent_views'],control)
    stage('capabilities',lambda:assemble_capabilities(report),report,'test_capabilities')
    print(json.dumps({'complete':True,'checkpoint_step':saved['step'],
                      'scores':report['test_capabilities']['scores']}),flush=True)


if __name__=='__main__':main()
