"""Audit cached features, fitted emission heads, and every saved prediction."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .emission_adapter import (ACTIVATIONS, ADAPTER_SEEDS, load_adapters, same_protocol,
                               score_adapters, sha)
from .readout_compare import expected_cohort, require
from .readout_diagnostic import REPRESENTATIONS, controlled_batch, load_diagnostic, visible
from .view_eval import FACTORS, accuracy_summary

ROOT=Path(__file__).resolve().parents[2]


def audit_rows(report, rows):
    expected=[(split,g) for split,r in report['evaluations'].items() for g in r['groups']]
    require(len(expected)==len(rows),'Prediction group count differs.')
    heads=set(report['fits'])|{'direct','linear/raw','linear/mixed'}
    scores={}
    for (split,group),row in zip(expected,rows):
        require((split,group['control'],group['query'])==(row['split'],row['control'],row['query']),
                'Prediction condition differs.')
        n=group['stop']-group['start']
        require(len(row['episode_ids'])==n and len(set(row['episode_ids']))==n,'Invalid episode inventory.')
        require(len(row['answers'])==n and Counter(row['answers'])=={v:n//4 for v in b'ABCD'},'Target balance differs.')
        require(set(row['predictions'])==heads and set(group['decoders'])==heads,'Common head inventory differs.')
        for name,predicted in row['predictions'].items():
            require(len(predicted)==n and all(type(v) is int and 0<=v<256 for v in predicted),'Invalid predicted byte.')
            metrics=accuracy_summary(predicted,row['answers'])
            require(all(group['decoders'][name].get(k)==v for k,v in metrics.items()),'Prediction score differs.')
            nll=group['decoders'][name]['answer_nats_per_byte']
            require(math.isfinite(nll) and nll>=0,'Invalid answer NLL.')
            scores[(split,group['control'],group['query'],name)]=metrics['exact_accuracy']
    return scores


def summarize(scores, heads):
    result={}
    for name in heads:
        result[name]={
            'heldout_overall':statistics.mean(scores[('test','all',q,name)] for q in FACTORS),
            'heldout_private':statistics.mean(scores[('test','all',q,name)] for q in ('left','right')),
            'heldout_shared':scores[('test','all','shared',name)],
            'b_only_shared':scores[('independent','b_only','shared',name)],
            'missing_information':statistics.mean(v for (s,c,q,h),v in scores.items()
                if s=='independent' and h==name and not visible(c,q))}
    return result


def audit_run(directory, run, rebuilt=None):
    directory=Path(directory)
    report=json.loads((directory/'results.json').read_text())
    rows=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    require(report['complete'] and report['frozen_verified'] and report['baseline_predictions_reproduced'],
            'Incomplete experiment or frozen/reference checks.')
    require(report['checkpoint_sha256']==run['checkpoint_sha256']==sha(ROOT/run['checkpoint']),
            'Frozen checkpoint bytes differ.')
    reference_path=ROOT/run['reference'];prior_path=ROOT/run['prior_results']
    require(report['reference_sha256']==sha(reference_path) and report['prior_results_sha256']==sha(prior_path),
            'Reference artifact bytes differ.')
    reference=load_diagnostic(reference_path)
    prior=json.loads(gzip.decompress(prior_path.read_bytes()))
    old_rows=[json.loads(s) for s in prior_path.with_name('predictions.jsonl').read_text().splitlines()]
    require(report['features_sha256']==sha(directory/'features.pt') and report['adapters_sha256']==sha(directory/'adapters.pt'),
            'Feature or head artifact bytes differ.')
    cache=torch.load(directory/'features.pt',map_location='cpu',weights_only=True)
    fitted=load_adapters(directory/'adapters.pt')
    require(cache['schema_version']==1 and cache['checkpoint_sha256']==report['checkpoint_sha256'],'Invalid feature artifact.')
    require(cache['source_sha256']==report['source_sha256'],'Capture source provenance differs.')
    require(all(sha(ROOT/path)==digest for path,digest in report['source_sha256'].items()),'Measured source has changed.')
    require(set(cache['datasets'])=={'train','test','independent'},'Missing captured cohort.')
    require(report['initialization_seeds']==fitted['initialization_seeds']==list(ADAPTER_SEEDS),'Initialization seed set differs.')
    require(report['fits']=={k:h.fit for k,h in fitted['heads'].items()},'Saved fit metadata differs.')
    require(report['fit_query_counts']==fitted['query_counts']=={'shared':448,'left':256,'right':256},'Query exposure differs.')
    require(same_protocol(fitted['fit_protocol'],reference['fit_protocol']) and
            same_protocol(report['fit_protocol'],fitted['fit_protocol']),'Fit cohort differs from the linear reference.')
    require(report['model_tensor_sha256']==reference['fit_protocol']['model_tensor_sha256'],'Frozen model differs.')
    require(report['training_protocol'].get('source_schedule')==run['source_schedule'] and
            report['training_protocol']['seed']==run['training_seed'],'Training lineage differs.')
    require(report['fit_protocol']['policy']['evidence_gated_ports']==(run['publication_policy']=='gated'),
            'Publication policy differs.')
    require(same_protocol(report['training_protocol'],prior['training_protocol']),'Original graph training protocol changed.')
    rebuilt={} if rebuilt is None else rebuilt
    for split,data in cache['datasets'].items():
        protocol=data['protocol']
        old_protocol=reference['fit_protocol'] if split=='train' else prior['evaluations'][split]['protocol']
        require(same_protocol(protocol,old_protocol),'Captured protocol differs from the reference.')
        require(protocol['frozen_verified'] and protocol['model_tensor_sha256']==report['model_tensor_sha256'],'Frozen graph changed.')
        signature=tuple(protocol[k] for k in ('split','count_per_cell','seed','gap','query_every'))
        if signature not in rebuilt:rebuilt[signature]=expected_cohort(protocol)
        expected=rebuilt[signature]
        require(all(protocol[k]==expected[k] for k in ('inputs_sha256','case_sha256','episodes')),'Input cohort reconstruction failed.')
        require([(g['control'],g['query']) for g in data['groups']]==expected['groups'],'Control inventory differs.')
        count=protocol['episodes'];cfg=protocol['policy']['graph_config']
        calls_per_event=len(cfg['nodes'])*3*cfg['active_groups']*cfg['experts_per_group']
        require(cfg['rounds']==1 and protocol['expert_executions']==count*(17+protocol['gap'])*calls_per_event,
                'Expert budget differs.')
        for representation,width in (('raw',64),('mixed',32)):
            x=data['features']['final/'+representation]
            require(x.shape==(count,width) and torch.isfinite(x).all() and not x.requires_grad,'Invalid frozen features.')
        for group in data['groups']:
            a,b=group['start'],group['stop']
            batch=controlled_batch(split,0,b-a,control=group['control'],query=group['query'],
                **{k:protocol[k] for k in ('seed','gap','query_every')})
            require(data['identities'][a:b]==list(batch.identities) and torch.equal(data['answers'][a:b],batch.answers),
                    'Saved identities or targets differ from the synthetic generator.')
            require(data['queries'][a:b]==[group['query']]*(b-a) and data['controls'][a:b]==[group['control']]*(b-a),
                    'Saved fitting metadata differs from its controls.')
        if split=='train':
            require(fitted['fit_identities']==reference['fit_identities']==expected['fit_identities'],'Training identities differ.')
        else:
            require(not set(data['identities']) & set(fitted['fit_identities']),'Fit/evaluation leakage.')
            rescored,predictions=score_adapters(data,fitted,reference)
            require(same_protocol(rescored,report['evaluations'][split]),'Cached-feature score reproduction failed.')
            saved=[r for r in rows if r['split']==split]
            require(predictions==saved,'Restored head predictions differ from raw rows.')
            for now,old in zip(saved,[r for r in old_rows if r['split']==split],strict=True):
                require(now['episode_ids']==old['episode_ids'] and now['answers']==old['answers'],'Prior diagnostic cases changed.')
                for new,previous in [('direct','direct'),('linear/raw','raw'),('linear/mixed','mixed')]:
                    require(now['predictions'][new]==old['predictions'][previous],'Prior baseline predictions changed.')
    training=cache['datasets']['train']
    weights=torch.tensor([1/fitted['query_counts'][q] for q in training['queries']],dtype=torch.double)
    weights=weights/weights.sum()
    for representation in REPRESENTATIONS:
        x=training['features']['final/'+representation].double()
        mean=(x*weights[:,None]).sum(0)
        scale=((x-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(1e-6)
        for seed in ADAPTER_SEEDS:
            pair=[fitted['heads'][f'{representation}/{a}/{seed}'] for a in ACTIVATIONS]
            require(pair[0].fit['initial_tensor_sha256']==pair[1].fit['initial_tensor_sha256'],'Matched initialization differs.')
            for head in pair:
                fit=head.fit
                require(fit['hidden']==64 and fit['max_iter']==400 and fit['l2']==.001,'Fitting protocol differs.')
                require(fit['parameters']==((x.shape[1]+1)*64+(64+1)*256),'Parameter budget differs.')
                require(torch.equal(head.mean,mean) and torch.equal(head.scale,scale),'Normalization used different training data.')
                logits=head.logits(x);nll=float((F.cross_entropy(logits,training['answers'],reduction='none')*weights).sum())
                objective=nll+.5*fit['l2']*sum(float(p.square().sum()) for p in (head.w1,head.b1,head.w2,head.b2))
                accuracy=float(((logits.argmax(-1)==training['answers']).double()*weights).sum())
                require(math.isclose(nll,fit['weighted_training_nll'],rel_tol=1e-12,abs_tol=1e-12) and
                        math.isclose(objective,fit['objective'],rel_tol=1e-12,abs_tol=1e-12) and
                        accuracy==fit['weighted_training_accuracy'],'Training objective reconstruction failed.')
    scores=audit_rows(report,rows)
    by_head=summarize(scores,rows[0]['predictions'])
    families={name:[name] for name in ('direct','linear/raw','linear/mixed')}
    families.update({f'{r}/{a}':[f'{r}/{a}/{s}' for s in ADAPTER_SEEDS] for r in REPRESENTATIONS for a in ACTIVATIONS})
    means={family:{metric:statistics.mean(by_head[h][metric] for h in names) for metric in by_head['direct']}
           for family,names in families.items()}
    files=[{'name':name,'sha256':sha(directory/name),'size_bytes':(directory/name).stat().st_size}
           for name in ('results.json','predictions.jsonl','adapters.pt','features.pt')]
    return {'name':run['name'],'publication_policy':run['publication_policy'],'source_schedule':run['source_schedule'],
        'training_seed':run['training_seed'],'checkpoint_sha256':report['checkpoint_sha256'],
        'model_tensor_sha256':report['model_tensor_sha256'],'metrics':means,'by_head':by_head,
        'fits':report['fits'],'diagnostic_expert_executions':sum(d['protocol']['expert_executions'] for d in cache['datasets'].values()),
        'seconds':report['seconds'],'files':files}


def compare(source):
    source=Path(source);status=json.loads((source/'status.json').read_text())
    require(status['complete'] and all(r['status']=='complete' and r['returncode']==0 for r in status['runs']),
            'All experiment runs must complete.')
    expected={(p,s,n) for p in ('original','gated') for s in ('dropout','balanced') for n in (20261106,20261107,20261108)}
    require(len(status['runs'])==12 and {(r['publication_policy'],r['source_schedule'],r['training_seed']) for r in status['runs']}==expected,
            'The study requires all twelve frozen models.')
    rebuilt={};runs=[audit_run(source/r['name'],r,rebuilt) for r in status['runs']]
    groups={}
    for policy,schedule in [(p,s) for p in ('original','gated') for s in ('dropout','balanced')]:
        cohort=[r for r in runs if r['publication_policy']==policy and r['source_schedule']==schedule]
        groups[policy+'/'+schedule]={family:{metric:{'mean':statistics.mean(r['metrics'][family][metric] for r in cohort),
            'min_graph':min(r['metrics'][family][metric] for r in cohort),'max_graph':max(r['metrics'][family][metric] for r in cohort)}
            for metric in cohort[0]['metrics'][family]} for family in cohort[0]['metrics']}
    return {'schema_version':1,'verified_models':12,'fitted_heads':144,'adapter_seeds':list(ADAPTER_SEEDS),
        'settings':status['settings'],'groups':groups,'runs':runs,
        'total_diagnostic_expert_executions':sum(r['diagnostic_expert_executions'] for r in runs),
        'integrity':{'predictions_rescored':True,'cached_feature_predictions_reproduced':True,
            'synthetic_cases_reconstructed':True,'prior_baselines_reproduced_exactly':True,
            'training_objectives_reconstructed':True,'paired_initializations_and_parameter_counts_checked':True,
            'no_evaluation_based_selection':True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    result=compare(args.source);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified_models':result['verified_models'],'fitted_heads':result['fitted_heads'],'out':str(args.out)}))


if __name__=='__main__':
    main()
