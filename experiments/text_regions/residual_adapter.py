"""Frozen linear byte head plus a validation-scaled residual correction."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .emission_adapter import EmissionAdapter, ADAPTER_SEEDS, ACTIVATIONS, sha, tensor_hash, same_protocol
from .readout_diagnostic import (ByteDecoder, SUBSETS, capture_readout_features, controlled_batch,
                                 load_diagnostic, visible, _digest)
from .transfer_eval import _policy
from .view_eval import FACTORS, accuracy_summary
from .view_train import load_view_checkpoint, tensor_fingerprint

STRENGTHS=(0.,.125,.25,.5,1.)
SELECTION_RULE='minimum query-balanced validation NLL subject to all-source/shared and B-only/shared accuracy >= frozen baseline; ties prefer smaller strength'


def write_bytes(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    if path.read_bytes()!=data:raise RuntimeError('Artifact writeback verification failed.')


def write_json(path,value):
    write_bytes(path,(json.dumps(value,indent=2)+'\n').encode())


def save_tensors(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def base_fingerprint(base):
    return tensor_hash((base.mean,base.scale,base.weight))


def balanced_weights(data):
    counts={q:data['queries'].count(q) for q in FACTORS}
    n=len(data['answers'])
    if (not all(counts.values()) or sum(counts.values())!=n or len(data['controls'])!=n
            or not all(visible(c,q) for c,q in zip(data['controls'],data['queries']))):
        raise ValueError('Expected every query with supported sources only.')
    mass=torch.tensor([1/counts[q] for q in data['queries']],dtype=torch.double)
    return mass/mass.sum()


@dataclass(frozen=True)
class ResidualHead:
    base: ByteDecoder
    correction: EmissionAdapter
    strength: float
    selection: dict

    def logits(self,features):
        if not math.isfinite(self.strength) or not 0<=self.strength<=1:
            raise ValueError('Residual strength must be in [0,1].')
        base=self.base.logits(features)
        return base if self.strength==0 else base+self.strength*self.correction.logits(features)

    def predict(self,features):return self.logits(features).argmax(-1)


def initialize_residual(base, *, hidden=16, seed=20261121, activation='gelu'):
    if hidden<1 or activation not in ACTIVATIONS:raise ValueError('Invalid residual architecture.')
    d=len(base.mean)
    if (base.weight.shape!=(d+1,256) or base.scale.shape!=(d,) or not (base.scale>0).all()
            or any(not torch.isfinite(t).all() for t in (base.mean,base.scale,base.weight))):
        raise ValueError('Invalid frozen byte head.')
    frozen=ByteDecoder(*(t.detach().clone() for t in (base.mean,base.scale,base.weight)),dict(base.fit))
    generator=torch.Generator().manual_seed(seed)
    tensors=[torch.randn(d,hidden,generator=generator,dtype=torch.double)/math.sqrt(d),
             torch.zeros(hidden,dtype=torch.double),torch.zeros(hidden,256,dtype=torch.double),
             torch.zeros(256,dtype=torch.double)]
    metadata={'width':d,'hidden':hidden,'output_bytes':256,'activation':activation,'seed':seed,
        'parameters':sum(t.numel() for t in tensors),'initial_tensor_sha256':tensor_hash(tensors),
        'initial_correction_exactly_zero':True,'base_head_sha256':base_fingerprint(frozen)}
    correction=EmissionAdapter(frozen.mean.clone(),frozen.scale.clone(),*tensors,activation,metadata)
    return ResidualHead(frozen,correction,1.,{})


def fit_residual(features,answers,base, *, sample_weights=None,hidden=16,seed=20261121,
                 activation='gelu',l2=.001,max_iter=400):
    x,y=features.detach().cpu().double(),answers.detach().cpu().long()
    if (x.ndim!=2 or not len(x) or x.shape[1]!=len(base.mean) or y.shape!=(len(x),)
            or not torch.isfinite(x).all() or ((y<0)|(y>255)).any()
            or not math.isfinite(l2) or l2<=0 or max_iter<1):raise ValueError('Invalid residual fitting inputs.')
    mass=torch.ones(len(x),dtype=torch.double) if sample_weights is None else sample_weights.detach().cpu().double()
    if mass.shape!=y.shape or not torch.isfinite(mass).all() or (mass<=0).any():raise ValueError('Invalid fitting weights.')
    mass=mass/mass.sum()
    head=initialize_residual(base,hidden=hidden,seed=seed,activation=activation)
    before=base_fingerprint(base);baseline=head.base.logits(x).detach()
    z=(x-head.base.mean)/head.base.scale
    parameters=[getattr(head.correction,name).clone().requires_grad_(True) for name in ('w1','b1','w2','b2')]
    w1,b1,w2,b2=parameters
    def logits():
        h=z@w1+b1
        return baseline+(F.gelu(h) if activation=='gelu' else h)@w2+b2
    if not torch.equal(logits().detach(),baseline):raise RuntimeError('Residual initialization changed the baseline.')
    optimizer=torch.optim.LBFGS(parameters,max_iter=max_iter,history_size=30,
        tolerance_grad=1e-7,tolerance_change=1e-11,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=(F.cross_entropy(logits(),y,reduction='none')*mass).sum()+.5*l2*sum(p.square().sum() for p in parameters)
        loss.backward();return loss
    started=time.perf_counter()
    with torch.enable_grad():optimizer.step(closure);objective=float(closure().detach())
    with torch.no_grad():
        result=logits();nll=float((F.cross_entropy(result,y,reduction='none')*mass).sum())
        accuracy=float(((result.argmax(-1)==y).double()*mass).sum())
    if not math.isfinite(objective) or any(not torch.isfinite(p).all() for p in parameters):raise RuntimeError('Nonfinite residual fit.')
    if base_fingerprint(base)!=before or base_fingerprint(head.base)!=before:raise RuntimeError('Frozen linear head changed.')
    metadata={**head.correction.fit,'examples':len(x),'l2':l2,'max_iter':max_iter,
        'iterations':optimizer.state[w1]['n_iter'],'gradient_max':max(float(p.grad.abs().max()) for p in parameters),
        'objective':objective,'weighted_training_nll':nll,'weighted_training_accuracy':accuracy,
        'seconds':time.perf_counter()-started,'optimizer':'full-batch L-BFGS, strong Wolfe',
        'normalization':'frozen linear head training mean/scale','base_frozen_verified':True,
        'regularization':'0.5 * l2 * squared sum of correction weights and biases only'}
    correction=replace(head.correction,**dict(zip(('w1','b1','w2','b2'),(p.detach().clone() for p in parameters))),fit=metadata)
    return replace(head,correction=correction)


def select_strength(head,validation, *, fit_identities):
    if validation['protocol']['split']!='validation':raise ValueError('Select strength using validation only.')
    if set(validation['identities'])&set(fit_identities):raise ValueError('Training/validation identity overlap.')
    mass=balanced_weights(validation);x=validation['features']['final/raw'];y=validation['answers']
    baseline=head.base.logits(x);delta=head.correction.logits(x)
    masks={control:torch.tensor([c==control and q=='shared' for c,q in zip(validation['controls'],validation['queries'])])
           for control in ('all','b_only')}
    if not all(mask.any() for mask in masks.values()):raise ValueError('Missing validation guard condition.')
    baseline_correct={c:int((baseline.argmax(-1)[m]==y[m]).sum()) for c,m in masks.items()}
    candidates=[]
    for strength in STRENGTHS:
        logits=baseline if strength==0 else baseline+strength*delta;predicted=logits.argmax(-1)
        correct={c:int((predicted[m]==y[m]).sum()) for c,m in masks.items()}
        row={'strength':strength,'weighted_nll':float((F.cross_entropy(logits,y,reduction='none')*mass).sum()),
             'weighted_accuracy':float(((predicted==y).double()*mass).sum()),'shared_correct':correct,
             'eligible':all(correct[c]>=baseline_correct[c] for c in masks)}
        if not math.isfinite(row['weighted_nll']):raise ValueError('Nonfinite validation objective.')
        candidates.append(row)
    best=min(r['weighted_nll'] for r in candidates if r['eligible'])
    chosen=min((r for r in candidates if r['eligible'] and r['weighted_nll']<=best+1e-12),key=lambda r:r['strength'])
    selection={'split':'validation','rule':SELECTION_RULE,'strengths':list(STRENGTHS),'candidates':candidates,
        'baseline_shared_correct':baseline_correct,'guard_counts':{c:int(m.sum()) for c,m in masks.items()},
        'selected_strength':chosen['strength'],'selected_weighted_nll':chosen['weighted_nll']}
    return replace(head,strength=chosen['strength'],selection=selection)


def capture_validation(model, *, count=128, seed=20261120, gap=8, query_every=4,batch_size=64,segment=16,progress=None):
    if count<16 or count%16 or min(batch_size,segment,query_every)<1 or gap<0:raise ValueError('Use complete validation tuple cycles.')
    before=tensor_fingerprint(model);policy=_policy(model);rng=torch.get_rng_state().clone()
    features=[];answers=[];identities=[];queries=[];controls=[];groups=[];cursor=0
    inputs=hashlib.sha256()
    for control in SUBSETS:
        for query in FACTORS:
            if not visible(control,query):continue
            calls=events=0
            for start in range(0,count,batch_size):
                batch=controlled_batch('validation',start,min(batch_size,count-start),control=control,query=query,
                    seed=seed,gap=gap,query_every=query_every)
                captured=capture_readout_features(model,batch,segment)
                features.append(captured['features']['final/raw']);answers.append(batch.answers)
                identities.extend(batch.identities);queries.extend(batch.query_types);controls.extend([control]*len(batch.identities))
                calls+=captured['expert_executions'];events+=captured['external_events']
                for i,identity in enumerate(batch.identities):
                    inputs.update(json.dumps((control,query,identity)).encode())
                    for source in 'ABC':
                        inputs.update(bytes(batch.inputs[source][i].tolist()));inputs.update(bytes(batch.availability[source][i].tolist()))
                    inputs.update(bytes(batch.reset[i].tolist()))
            groups.append({'control':control,'query':query,'start':cursor,'stop':cursor+count,
                           'expert_executions':calls,'external_events':events,'information_present':True})
            cursor+=count
            if progress:progress({'validation_cell':len(groups),'control':control,'query':query})
    if tensor_fingerprint(model)!=before or _policy(model)!=policy or not torch.equal(rng,torch.get_rng_state()):
        raise RuntimeError('Validation capture changed the frozen graph or RNG.')
    return {'features':{'final/raw':torch.cat(features)},'answers':torch.cat(answers),'identities':identities,
        'queries':queries,'controls':controls,'groups':groups,'protocol':{'split':'validation','count_per_cell':count,
            'seed':seed,'gap':gap,'query_every':query_every,'batch_size':batch_size,'segment':segment,'episodes':cursor,
            'model_tensor_sha256':before,'policy':policy,'case_sha256':_digest(list(zip(controls,identities))),
            'inputs_sha256':inputs.hexdigest(),'expert_executions':sum(g['expert_executions'] for g in groups),
            'external_events':sum(g['external_events'] for g in groups),'frozen_verified':True}}


def fit_and_select(training,validation,base, *, seeds=ADAPTER_SEEDS,hidden=16,l2=.001,max_iter=400,progress=None):
    if training['protocol']['split']!='train' or validation['protocol']['split']!='validation':raise ValueError('Training and validation splits required.')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Use distinct initialization seeds.')
    fit_ids=sorted(set(training['identities']));valid_ids=sorted(set(validation['identities']))
    if set(fit_ids)&set(valid_ids):raise ValueError('Training and validation overlap.')
    for key in ('model_tensor_sha256','policy','gap','query_every'):
        if not same_protocol(training['protocol'][key],validation['protocol'][key]):raise ValueError('Validation graph or protocol differs.')
    mass=balanced_weights(training);x=training['features']['final/raw'].double()
    mean=(x*mass[:,None]).sum(0);scale=((x-mean).square()*mass[:,None]).sum(0).sqrt().clamp_min(1e-6)
    if not torch.equal(mean,base.mean) or not torch.equal(scale,base.scale):raise ValueError('Training data differs from frozen normalization.')
    heads={}
    for seed in seeds:
        for activation in ACTIVATIONS:
            name=f'{activation}/{seed}'
            head=fit_residual(x,training['answers'],base,sample_weights=mass,hidden=hidden,seed=seed,
                              activation=activation,l2=l2,max_iter=max_iter)
            heads[name]=select_strength(head,validation,fit_identities=fit_ids)
            if progress:progress({'fit':name,'strength':heads[name].strength,**head.correction.fit})
    return {'schema_version':1,'base':base,'heads':heads,'fit_protocol':training['protocol'],
            'validation_protocol':validation['protocol'],'fit_identities':fit_ids,'validation_identities':valid_ids,
            'initialization_seeds':list(seeds),'selection_rule':SELECTION_RULE}


def save_residuals(path,fitted):
    save_tensors(path,{**fitted,'base':asdict(fitted['base']),
        'heads':{n:{'correction':asdict(h.correction),'strength':h.strength,'selection':h.selection} for n,h in fitted['heads'].items()}})


def load_residuals(path):
    p=torch.load(path,map_location='cpu',weights_only=True)
    expected={'schema_version','base','heads','fit_protocol','validation_protocol','fit_identities',
              'validation_identities','initialization_seeds','selection_rule'}
    if (not isinstance(p,dict) or set(p)!=expected or p['schema_version']!=1 or p['fit_protocol']['split']!='train'
            or p['validation_protocol']['split']!='validation' or p['selection_rule']!=SELECTION_RULE):raise ValueError('Invalid residual artifact.')
    seeds=p['initialization_seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):raise ValueError('Invalid seeds.')
    for key in ('fit_identities','validation_identities'):
        ids=p[key]
        if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,str) for i in ids):raise ValueError('Invalid split identities.')
    if set(p['fit_identities'])&set(p['validation_identities']):raise ValueError('Split overlap.')
    if set(p['base'])!=set(ByteDecoder.__dataclass_fields__):raise ValueError('Invalid frozen base fields.')
    base=ByteDecoder(**p['base']);d=len(base.mean)
    if (base.mean.shape!=(d,) or base.scale.shape!=(d,) or base.weight.shape!=(d+1,256) or not (base.scale>0).all()
            or any(not t.is_floating_point() or t.requires_grad or not torch.isfinite(t).all() for t in (base.mean,base.scale,base.weight))):
        raise ValueError('Invalid frozen base tensors.')
    if set(p['heads'])!={f'{a}/{s}' for a in ACTIVATIONS for s in seeds}:raise ValueError('Incomplete residual head inventory.')
    for name,record in p['heads'].items():
        if set(record)!={'correction','strength','selection'} or set(record['correction'])!=set(EmissionAdapter.__dataclass_fields__):
            raise ValueError('Invalid residual record.')
        c=EmissionAdapter(**record['correction']);h=c.fit.get('hidden');a,seed=name.split('/')
        tensors=[getattr(c,k) for k in ('mean','scale','w1','b1','w2','b2')]
        if (type(h) is not int or h<1 or any(not torch.is_tensor(t) or not t.is_floating_point() or t.requires_grad or not torch.isfinite(t).all() for t in tensors)
                or [tuple(t.shape) for t in tensors]!=[(d,),(d,),(d,h),(h,),(h,256),(256,)]
                or not torch.equal(c.mean,base.mean) or not torch.equal(c.scale,base.scale)
                or c.activation!=a or c.fit.get('seed')!=int(seed) or c.fit.get('base_head_sha256')!=base_fingerprint(base)
                or c.fit.get('parameters')!=(d+1)*h+(h+1)*256 or c.fit.get('examples')!=p['fit_protocol']['episodes']):
            raise ValueError('Invalid residual tensors or fitting provenance.')
        selection=record['selection'];strength=record['strength']
        if (strength not in STRENGTHS or selection.get('split')!='validation' or selection.get('rule')!=SELECTION_RULE
                or selection.get('selected_strength')!=strength or selection.get('strengths')!=list(STRENGTHS)):
            raise ValueError('Invalid validation selection.')
        p['heads'][name]=ResidualHead(base,c,strength,selection)
    p['base']=base
    return p


def score_residuals(data,fitted):
    if data['protocol']['split'] not in ('test','independent'):raise ValueError('Test or independent cohort required.')
    if set(data['identities'])&(set(fitted['fit_identities'])|set(fitted['validation_identities'])):raise ValueError('Evaluation split overlap.')
    for key in ('model_tensor_sha256','policy','gap','query_every'):
        if not same_protocol(data['protocol'][key],fitted['fit_protocol'][key]):raise ValueError('Evaluation graph or protocol differs.')
    x=data['features']['final/raw'];y=data['answers'];base=fitted['base'].logits(x)
    logits={'linear/raw':base}
    for name,h in fitted['heads'].items():
        delta=h.correction.logits(x)
        logits[name+'/full']=base+delta
        logits[name+'/selected']=base if h.strength==0 else base+h.strength*delta
    predictions={name:z.argmax(-1) for name,z in logits.items()}
    nll={name:F.cross_entropy(z,y,reduction='none') for name,z in logits.items()}
    groups=[];rows=[]
    for group in data['groups']:
        a,b=group['start'],group['stop'];answers=y[a:b].tolist()
        scores={};row={'split':data['protocol']['split'],'control':group['control'],'query':group['query'],
                      'episode_ids':data['identities'][a:b],'answers':answers,'predictions':{}}
        for name,values in predictions.items():
            p=values[a:b].tolist();row['predictions'][name]=p
            scores[name]={**accuracy_summary(p,answers),'answer_nats_per_byte':float(nll[name][a:b].mean())}
        groups.append({**group,'decoders':scores});rows.append(row)
    return {'protocol':data['protocol'],'groups':groups},rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--prior-run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--validation-count',type=int,default=128)
    args=parser.parse_args()
    if args.out.exists() or args.threads<1:parser.error('Use a new output directory and positive thread count.')
    torch.set_num_threads(args.threads);args.out.mkdir(parents=True);started=time.perf_counter()
    progress=lambda value:print(json.dumps(value),flush=True)
    files=[Path(__file__).with_name(n) for n in ('residual_adapter.py','emission_adapter.py','readout_diagnostic.py',
             'model.py','learning.py','transfer_eval.py','view_data.py','view_eval.py','view_train.py')]
    root=Path(__file__).resolve().parents[2];sources={str(p.relative_to(root)):sha(p) for p in files}
    previous=json.loads((args.prior_run/'results.json').read_text())
    if sha(args.prior_run/'features.pt')!=previous['features_sha256'] or sha(args.reference)!=previous['reference_sha256']:
        raise ValueError('Prior feature or base artifact changed.')
    if sha(args.checkpoint)!=previous['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed.')
    cache=torch.load(args.prior_run/'features.pt',map_location='cpu',weights_only=True)
    reference=load_diagnostic(args.reference);training=cache['datasets']['train']
    if not same_protocol(training['protocol'],reference['fit_protocol']):raise ValueError('Training cache differs from reference.')
    model,_,_,_=load_view_checkpoint(args.checkpoint,restore_rng=False)
    if tensor_fingerprint(model)!=training['protocol']['model_tensor_sha256'] or model.cfg.rounds!=1:
        raise ValueError('Frozen graph differs from training features.')
    options={k:training['protocol'][k] for k in ('seed','gap','query_every','batch_size','segment')}
    validation=capture_validation(model,count=args.validation_count,progress=progress,**options)
    del model
    save_tensors(args.out/'validation.pt',validation)
    fitted=fit_and_select(training,validation,reference['answer_heads']['raw'],progress=progress)
    save_residuals(args.out/'residuals.pt',fitted);restored=load_residuals(args.out/'residuals.pt')
    lock={'schema_version':1,'selection_inputs':['training','validation'],'selection_rule':SELECTION_RULE,
          'fit_inputs_sha256':training['protocol']['inputs_sha256'],'validation_inputs_sha256':validation['protocol']['inputs_sha256'],
          'heads':{name:{'strength':h.strength,'selection':h.selection} for name,h in restored['heads'].items()}}
    write_json(args.out/'selection_lock.json',lock)
    report={'schema_version':1,'complete':False,'source_sha256':sources,'checkpoint_sha256':sha(args.checkpoint),
        'reference_sha256':sha(args.reference),'prior_results_sha256':sha(args.prior_run/'results.json'),
        'prior_features_sha256':sha(args.prior_run/'features.pt'),'base_head_sha256':base_fingerprint(restored['base']),
        'fit_protocol':training['protocol'],'validation_protocol':validation['protocol'],
        'fits':{name:h.correction.fit for name,h in restored['heads'].items()},'selection':lock,
        'selection_lock_sha256':sha(args.out/'selection_lock.json'),'residuals_sha256':sha(args.out/'residuals.pt'),
        'validation_sha256':sha(args.out/'validation.pt'),'frozen_verified':True,'evaluations':{},
        'runtime':{'torch':str(torch.__version__),'threads':args.threads},'prior_training_protocol':previous['training_protocol']}
    rows=[]
    for split in ('test','independent'):
        result,predictions=score_residuals(cache['datasets'][split],restored)
        report['evaluations'][split]=result;rows.extend(predictions)
    write_bytes(args.out/'predictions.jsonl',''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode())
    report['predictions_sha256']=sha(args.out/'predictions.jsonl');report['prediction_groups']=len(rows)
    if sources!={str(p.relative_to(root)):sha(p) for p in files}:raise RuntimeError('Measured source changed during execution.')
    report.update(complete=True,seconds=time.perf_counter()-started)
    write_json(args.out/'results.json',report)
    progress({'complete':True,'seconds':report['seconds'],'out':str(args.out)})


if __name__=='__main__':main()
