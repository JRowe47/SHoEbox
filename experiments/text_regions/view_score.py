"""Evaluate saved shared/private models on separate generalization/control cohorts."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import torch
from .view_eval import evaluate_views, fit_source_readouts
from .view_train import load_view_checkpoint, tensor_fingerprint


def utility_summary(model):
    utility=model.utility
    entries={}
    for name in ('edge','lace'):
        mass=getattr(utility,name+'_mass');gain=getattr(utility,name+'_gain_sum')
        used=mass>0;values=(gain/mass.clamp_min(1e-8))[used]
        entries[name]={'eligible_associations':int(used.sum()),'total_associations':mass.numel(),
                       'mean_gain':float(values.mean()) if values.numel() else None,
                       'min_gain':float(values.min()) if values.numel() else None,
                       'max_gain':float(values.max()) if values.numel() else None}
    return {'enabled':utility.enabled,'commits':int(utility.feedback_updates),**entries}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--cohorts',nargs='+',choices=('validation','test','independent'),default=('test','independent'))
    parser.add_argument('--count-per-query',type=int,default=192)
    parser.add_argument('--batch',type=int,default=64);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--fit-count',type=int,default=128);parser.add_argument('--skip-readouts',action='store_true')
    parser.add_argument('--predictions',action='store_true')
    parser.add_argument('--evidence-gated-ports',action='store_true',help='Same-weight inference intervention; preserve expert counts.')
    parser.add_argument('--controls',nargs='+',choices=('all','a_only','b_only','c_only','no_private','reset_query'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if min(args.batch,args.threads,args.fit_count,args.count_per_query)<1:parser.error('Counts must be positive.')
    torch.set_num_threads(args.threads)
    model,_,_,saved=load_view_checkpoint(args.checkpoint,restore_rng=False)
    if args.evidence_gated_ports:
        model.model.cfg=replace(model.cfg,evidence_gated_ports=True)
    protocol=saved['protocol'];before=tensor_fingerprint(model)
    args.out.mkdir(parents=True,exist_ok=True)
    kwargs={key:protocol[key] for key in ('seed','gap','query_every','segment')}
    started=time.perf_counter()
    readouts=None if args.skip_readouts else fit_source_readouts(model,count=args.fit_count,batch_size=args.batch,**kwargs)
    fitting_seconds=time.perf_counter()-started
    if readouts is not None:
        torch.save({'source_readout_schema':1,**asdict(readouts)},args.out/'source_readouts.pt')
    report={'checkpoint_step':saved['step'],'model_tensor_sha256':before,'protocol':protocol,
            'source_readout_fit_seconds':fitting_seconds,'source_readout_fit_count':0 if readouts is None else args.fit_count,
            'utility':utility_summary(model),'inference_policy':{'evidence_gated_ports':model.cfg.evidence_gated_ports},
            'evaluations':{}}
    with (args.out/'predictions.jsonl').open('w') as predictions:
        for cohort in dict.fromkeys(args.cohorts):
            started=time.perf_counter()
            result=evaluate_views(model,split=cohort,count_per_query=args.count_per_query,
                                  batch_size=args.batch,source_readouts=readouts,
                                  include_predictions=args.predictions,controls=args.controls,**kwargs)
            result['seconds']=time.perf_counter()-started
            if args.predictions:
                for control,row in result['controls'].items():
                    record={'cohort':cohort,'control':control}
                    for key in ('predictions','answers','episode_keys'):record[key]=row.pop(key)
                    predictions.write(json.dumps(record,separators=(',',':'))+'\n')
                predictions.flush()
            report['evaluations'][cohort]=result
            if tensor_fingerprint(model)!=before:raise RuntimeError('Evaluation changed model parameters or utility buffers.')
            (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'cohort':cohort,'checkpoint_step':saved['step'],'seconds':result['seconds'],
                              'controls':{key:{'accuracy':value['exact_accuracy'],
                                               'queries':{q:v['exact_accuracy'] for q,v in value['by_query'].items()},
                                               'expert_executions':value['expert_executions']}
                                          for key,value in result['controls'].items()}}),flush=True)


if __name__=='__main__':main()
