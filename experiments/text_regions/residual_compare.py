"""Reconstruct residual-decoder results, validation decisions, and frozen baselines."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .emission_adapter import ACTIVATIONS, ADAPTER_SEEDS, sha, same_protocol
from .emission_compare import summarize
from .readout_compare import expected_cohort, require, digest
from .readout_diagnostic import SUBSETS, FACTORS, controlled_batch, visible, load_diagnostic
from .residual_adapter import (STRENGTHS, SELECTION_RULE, balanced_weights, base_fingerprint,
    initialize_residual, load_residuals, score_residuals, select_strength, write_json)
from .view_eval import accuracy_summary

ROOT=Path(__file__).resolve().parents[2]


def expected_validation(protocol):
    require(protocol['split']=='validation','Validation split required.')
    cases=[];identities=[];groups=[];inputs=hashlib.sha256();count=protocol['count_per_cell']
    for control in SUBSETS:
        for query in FACTORS:
            if not visible(control,query):continue
            batch=controlled_batch('validation',0,count,control=control,query=query,
                **{k:protocol[k] for k in ('seed','gap','query_every')})
            require(Counter(batch.answers.tolist())=={v:count//4 for v in b'ABCD'},'Validation targets are unbalanced.')
            groups.append((control,query))
            for i,identity in enumerate(batch.identities):
                cases.append((control,identity));identities.append(identity)
                inputs.update(json.dumps((control,query,identity)).encode())
                for source in 'ABC':
                    inputs.update(bytes(batch.inputs[source][i].tolist()))
                    inputs.update(bytes(batch.availability[source][i].tolist()))
                inputs.update(bytes(batch.reset[i].tolist()))
    return {'case_sha256':digest(cases),'inputs_sha256':inputs.hexdigest(),
        'fit_identities':sorted(set(identities)),'episodes':len(cases),'groups':groups}


def audit_rows(report,rows):
    expected=[(split,g) for split,r in report['evaluations'].items() for g in r['groups']]
    require(len(rows)==len(expected),'Incomplete prediction export.')
    heads={'linear/raw'}|{name+'/'+mode for name in report['fits'] for mode in ('full','selected')}
    scores={}
    for (split,group),row in zip(expected,rows):
        require((split,group['control'],group['query'])==(row['split'],row['control'],row['query']),
                'Prediction condition differs.')
        n=group['stop']-group['start']
        require(len(row['episode_ids'])==len(set(row['episode_ids']))==n,'Invalid evaluation identities.')
        require(Counter(row['answers'])=={v:n//4 for v in b'ABCD'},'Unbalanced evaluation answers.')
        require(set(row['predictions'])==set(group['decoders'])==heads,'Decoder inventory differs.')
        for name,predicted in row['predictions'].items():
            require(len(predicted)==n and all(type(v) is int and 0<=v<256 for v in predicted),'Invalid output bytes.')
            metrics=accuracy_summary(predicted,row['answers']);saved=group['decoders'][name]
            require(all(saved.get(k)==v for k,v in metrics.items()),'Prediction score differs.')
            require(math.isfinite(saved['answer_nats_per_byte']) and saved['answer_nats_per_byte']>=0,'Invalid NLL.')
            scores[(split,group['control'],group['query'],name)]=metrics['exact_accuracy']
    return scores


def audit_cohort(data,rebuilt):
    protocol=data['protocol'];split=protocol['split']
    signature=tuple(protocol[k] for k in ('split','count_per_cell','seed','gap','query_every'))
    if signature not in rebuilt:
        rebuilt[signature]=(expected_validation if split=='validation' else expected_cohort)(protocol)
    expected=rebuilt[signature]
    require(all(protocol[k]==expected[k] for k in ('inputs_sha256','case_sha256','episodes')),'Input reconstruction differs.')
    require([(g['control'],g['query']) for g in data['groups']]==expected['groups'],'Source/query inventory differs.')
    count=protocol['episodes'];x=data['features']['final/raw'];cfg=protocol['policy']['graph_config']
    require(x.shape==(count,64) and torch.isfinite(x).all() and not x.requires_grad,'Invalid frozen features.')
    calls=len(cfg['nodes'])*3*cfg['active_groups']*cfg['experts_per_group']
    require(cfg['rounds']==1 and protocol['external_events']==count*(17+protocol['gap']) and
            protocol['expert_executions']==protocol['external_events']*calls and protocol['frozen_verified'],
            'Graph execution budget differs.')
    for group in data['groups']:
        a,b=group['start'],group['stop']
        batch=controlled_batch(split,0,b-a,control=group['control'],query=group['query'],
            **{k:protocol[k] for k in ('seed','gap','query_every')})
        require(data['identities'][a:b]==list(batch.identities) and torch.equal(data['answers'][a:b],batch.answers),
                'Identities or ground-truth bytes differ from generator.')
        require(data['queries'][a:b]==[group['query']]*(b-a) and data['controls'][a:b]==[group['control']]*(b-a),
                'Query/source metadata differs.')
    return expected


def audit_run(directory,run,rebuilt=None):
    directory=Path(directory);rebuilt={} if rebuilt is None else rebuilt
    report=json.loads((directory/'results.json').read_text())
    require(report['complete'] and report['frozen_verified'],'Incomplete residual experiment.')
    require(report['checkpoint_sha256']==run['checkpoint_sha256']==sha(ROOT/run['checkpoint']),'Checkpoint changed.')
    require(all(sha(ROOT/p)==h for p,h in report['source_sha256'].items()),'Measured source changed.')
    for filename,key in [('validation.pt','validation_sha256'),('residuals.pt','residuals_sha256'),
                         ('selection_lock.json','selection_lock_sha256'),('predictions.jsonl','predictions_sha256')]:
        require(sha(directory/filename)==report[key],'Artifact changed: '+filename)
    prior=ROOT/run['prior_run'];reference_path=ROOT/run['reference']
    require(sha(prior/'features.pt')==report['prior_features_sha256'] and
            sha(prior/'results.json')==report['prior_results_sha256'] and
            sha(reference_path)==report['reference_sha256'],'Prior artifacts changed.')
    previous=json.loads((prior/'results.json').read_text())
    require(same_protocol(previous['training_protocol'],report['prior_training_protocol']),'Graph training protocol changed.')
    require(previous['training_protocol']['seed']==run['training_seed'] and
            previous['training_protocol']['source_schedule']==run['source_schedule'],'Graph lineage differs.')
    reference=load_diagnostic(reference_path);fitted=load_residuals(directory/'residuals.pt')
    cache=torch.load(prior/'features.pt',map_location='cpu',weights_only=True)['datasets']
    validation=torch.load(directory/'validation.pt',map_location='cpu',weights_only=True)
    base=fitted['base'];training=cache['train'];mass=balanced_weights(training)
    require(base_fingerprint(base)==report['base_head_sha256']==base_fingerprint(reference['answer_heads']['raw']),
            'Frozen base head changed.')
    require(fitted['initialization_seeds']==list(ADAPTER_SEEDS),'Initialization inventory differs.')
    require(same_protocol(fitted['fit_protocol'],reference['fit_protocol']) and
            same_protocol(fitted['fit_protocol'],report['fit_protocol']) and
            same_protocol(validation['protocol'],fitted['validation_protocol']) and
            same_protocol(validation['protocol'],report['validation_protocol']),'Fit or validation protocol differs.')
    require(report['fit_protocol']['policy']['evidence_gated_ports']==(run['publication_policy']=='gated'),
            'Publication policy differs.')
    lock=json.loads((directory/'selection_lock.json').read_text())
    require(lock==report['selection'] and lock['selection_inputs']==['training','validation'] and
            lock['selection_rule']==SELECTION_RULE and lock['fit_inputs_sha256']==training['protocol']['inputs_sha256'] and
            lock['validation_inputs_sha256']==validation['protocol']['inputs_sha256'],'Selection lock differs.')
    ids={}
    for split,data in {**cache,'validation':validation}.items():
        expected=audit_cohort(data,rebuilt);ids[split]=set(expected['fit_identities'])
        for key in ('model_tensor_sha256','policy','gap','query_every'):
            require(same_protocol(data['protocol'][key],fitted['fit_protocol'][key]),'Frozen graph or capture protocol differs.')
    require(fitted['fit_identities']==sorted(ids['train']) and fitted['validation_identities']==sorted(ids['validation']),
            'Fitting/selection identity inventory differs.')
    require(all(not a&b for i,a in enumerate(ids.values()) for b in list(ids.values())[i+1:]),'Split identity overlap.')
    x=training['features']['final/raw'].double();mean=(x*mass[:,None]).sum(0)
    scale=((x-mean).square()*mass[:,None]).sum(0).sqrt().clamp_min(1e-6)
    require(torch.equal(mean,base.mean) and torch.equal(scale,base.scale),'Normalization differs from weighted training data.')
    for name,head in fitted['heads'].items():
        fit=head.correction.fit
        require(fit==report['fits'][name] and fit['hidden']==16 and fit['max_iter']==400 and fit['l2']==.001 and
                fit['parameters']==5392 and fit['examples']==960 and fit['base_frozen_verified'],
                'Fit budget or metadata differs.')
        initial=initialize_residual(base,hidden=16,seed=fit['seed'],activation=fit['activation'])
        require(initial.correction.fit['initial_tensor_sha256']==fit['initial_tensor_sha256'] and
                torch.equal(initial.logits(x),base.logits(x)) and fit['initial_correction_exactly_zero'],
                'Initial residual changed baseline.')
        corrected=base.logits(x)+head.correction.logits(x)
        normalized=mass/mass.sum()
        nll=float((F.cross_entropy(corrected,training['answers'],reduction='none')*normalized).sum())
        objective=nll+.5*fit['l2']*sum(float(p.square().sum()) for p in
                                      (head.correction.w1,head.correction.b1,head.correction.w2,head.correction.b2))
        require(math.isclose(nll,fit['weighted_training_nll'],abs_tol=1e-12,rel_tol=1e-12) and
                math.isclose(objective,fit['objective'],abs_tol=1e-12,rel_tol=1e-12),'Fit objective reconstruction differs.')
        chosen=select_strength(head,validation,fit_identities=fitted['fit_identities'])
        require(chosen.strength==head.strength and same_protocol(chosen.selection,head.selection) and
                lock['heads'][name]=={'strength':head.strength,'selection':head.selection},'Validation decision differs.')
        feasible=[c for c in head.selection['candidates'] if c['eligible']]
        require(feasible and feasible[0]['strength']==0 and all(c['strength'] in STRENGTHS for c in feasible),
                'Zero-strength fallback is missing.')
    rows=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    require(report['prediction_groups']==len(rows)==30,'Incomplete prediction groups.')
    prior_rows=[json.loads(line) for line in (prior/'predictions.jsonl').read_text().splitlines()]
    for split in ('test','independent'):
        rescored,predictions=score_residuals(cache[split],fitted)
        require(same_protocol(rescored,report['evaluations'][split]) and predictions==[r for r in rows if r['split']==split],
                'Restored residual predictions or losses differ.')
        for now,old in zip(predictions,[r for r in prior_rows if r['split']==split],strict=True):
            require(now['episode_ids']==old['episode_ids'] and now['answers']==old['answers'] and
                    now['predictions']['linear/raw']==old['predictions']['linear/raw'],'Prior baseline changed.')
    scores=audit_rows(report,rows);by_head=summarize(scores,rows[0]['predictions'])
    families={'linear/raw':['linear/raw']}
    families.update({a+'/'+mode:[f'{a}/{s}/{mode}' for s in ADAPTER_SEEDS] for a in ACTIVATIONS for mode in ('full','selected')})
    means={family:{metric:statistics.mean(by_head[h][metric] for h in names) for metric in by_head['linear/raw']}
           for family,names in families.items()}
    prior_metrics={}
    for metric in by_head['linear/raw']:
        # Preserve the previous unconstrained raw GELU comparison using its published groups.
        old_scores={(s,g['control'],g['query'],h):v['exact_accuracy'] for s,e in previous['evaluations'].items()
                    for g in e['groups'] for h,v in g['decoders'].items() if h.startswith('raw/gelu/')}
        old_summary=summarize(old_scores,sorted({k[-1] for k in old_scores}))
        prior_metrics[metric]=statistics.mean(m[metric] for m in old_summary.values())
    means['previous/gelu']=prior_metrics
    files=[{'name':name,'sha256':sha(directory/name),'size_bytes':(directory/name).stat().st_size}
           for name in ('results.json','predictions.jsonl','validation.pt','residuals.pt','selection_lock.json')]
    return {'name':run['name'],'publication_policy':run['publication_policy'],'source_schedule':run['source_schedule'],
        'training_seed':run['training_seed'],'checkpoint_sha256':report['checkpoint_sha256'],
        'model_tensor_sha256':report['fit_protocol']['model_tensor_sha256'],'metrics':means,'by_head':by_head,
        'fits':report['fits'],'selection':lock,'files':files,'seconds':report['seconds'],
        'validation_expert_executions':validation['protocol']['expert_executions']}


def compare(source):
    source=Path(source);status=json.loads((source/'status.json').read_text())
    require(status['complete'] and all(r['status']=='complete' and r['returncode']==0 for r in status['runs']),
            'All runs must complete.')
    expected={(p,s,n) for p in ('original','gated') for s in ('dropout','balanced') for n in (20261106,20261107,20261108)}
    require(len(status['runs'])==12 and {(r['publication_policy'],r['source_schedule'],r['training_seed']) for r in status['runs']}==expected,
            'All twelve frozen graphs required.')
    rebuilt={};runs=[audit_run(source/r['name'],r,rebuilt) for r in status['runs']]
    groups={}
    for policy,schedule in [(p,s) for p in ('original','gated') for s in ('dropout','balanced')]:
        cohort=[r for r in runs if (r['publication_policy'],r['source_schedule'])==(policy,schedule)]
        groups[policy+'/'+schedule]={family:{metric:{'mean':statistics.mean(r['metrics'][family][metric] for r in cohort),
            'min_graph':min(r['metrics'][family][metric] for r in cohort),'max_graph':max(r['metrics'][family][metric] for r in cohort)}
            for metric in cohort[0]['metrics'][family]} for family in cohort[0]['metrics']}
    strengths={a:dict(sorted(Counter(str(r['selection']['heads'][f'{a}/{s}']['strength']) for r in runs for s in ADAPTER_SEEDS).items()))
               for a in ACTIVATIONS}
    return {'schema_version':1,'verified_models':12,'fitted_heads':72,'adapter_seeds':list(ADAPTER_SEEDS),
        'settings':status['settings'],'groups':groups,'runs':runs,'selected_strengths':strengths,
        'total_validation_expert_executions':sum(r['validation_expert_executions'] for r in runs),
        'integrity':{'predictions_rescored':True,'saved_weight_predictions_and_losses_reproduced':True,
            'synthetic_cases_reconstructed':True,'prior_linear_baselines_reproduced_exactly':True,
            'training_objectives_reconstructed':True,'validation_selection_reproduced':True,
            'zero_initial_correction_verified':True,'training_validation_evaluation_identities_disjoint':True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    result=compare(args.source);args.out.parent.mkdir(parents=True,exist_ok=True);write_json(args.out,result)
    print(json.dumps({k:result[k] for k in ('verified_models','fitted_heads','selected_strengths')}))


if __name__=='__main__':main()
