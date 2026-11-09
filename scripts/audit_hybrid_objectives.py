"""Reload audit, held-out grounding, health and inference cost of Phase A."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shoebox.checkpoint import load
from shoebox.train_hierarchy import load_data, evaluate
from shoebox.hierarchy_objectives import future_pairs, diversity_loss
from shoebox.train_fineweb import atomic_json, source_digest


@torch.no_grad()
def grounding(model,docs):
    model.eval()
    group=torch.stack([d[:64] for d in docs if d.numel()>=64][:8])
    changed=group.clone(); changed[:,32:]=group.roll(1,0)[:,32:]
    start=time.perf_counter()
    a=model(group,collect_messages=True)
    b=model(changed,collect_messages=True)
    _,_,health=diversity_loss(a['publications'],.1)
    result=dict(message_health=health)
    if model.objective_config.mode=='latent':
        result['fixed_prefix']={}
        for name in ('L2','S2','L3'):
            pa,pb=[future_pairs(out['publications'],name) for out in (a,b)]
            choose=(pa['positions'][None,:].expand_as(pa['mask'])==31)[pa['mask']]
            x,y=pa['input'][choose],pb['input'][choose]
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            target_a=model.regional_objectives.normalized_bundle(pa['target'][choose],pa['sources'])
            target_b=model.regional_objectives.normalized_bundle(pb['target'][choose],pb['sources'])
            predictor=model.regional_objectives.predictors[name](x).reshape_as(target_a)
            persistence=model.regional_objectives.normalized_bundle(pa['persistence'][choose],pa['sources'])
            result['fixed_prefix'][name]=dict(prediction_input_exact=True,examples=x.shape[0],
                        changed_future_target_mse=float(F.mse_loss(target_a,target_b)),
                        original_huber=float(F.smooth_l1_loss(predictor,target_a)),
                        changed_future_huber=float(F.smooth_l1_loss(predictor,target_b)),
                        persistence_huber=float(F.smooth_l1_loss(persistence,target_a)))
    result['diagnostic_seconds']=time.perf_counter()-start
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,state,_,checkpoint=load(args.run/'last.pt')
    _,docs,_=load_data(args.data)
    expected=json.loads((args.run/'results.json').read_text())['evaluation'][-1]
    if not expected['full']:
        raise ValueError('Need a completed full held-out endpoint.')
    before={k:v.clone() for k,v in model.state_dict().items() if 'normalizers' in k}
    scored=evaluate(model,docs,context=128,span_metrics=True)
    fields=('nll','accuracy','targets','span_nll','span_targets','regional','message_std')
    exact=all(scored[k]==expected[k] for k in fields)
    if not exact:
        raise AssertionError({k:(expected[k],scored[k]) for k in fields if expected[k]!=scored[k]})
    for k,v in before.items():
        torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    finite=all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    ages={}
    for group,names in zip(checkpoint['optimizer']['param_groups'],checkpoint['optimizer_names']):
        for key,name in zip(group['params'],names):
            item=checkpoint['optimizer']['state'].get(key)
            if item:
                ages[name]=float(item['step'])
    result=dict(exact_score_reproduction=exact,finite_parameters_and_statistics=finite,
                normalizers_unchanged_by_validation=True,source_matches=source_digest()==checkpoint['metadata']['protocol']['source_sha256'],
                checkpoint_sha256=hashlib.sha256((args.run/'last.pt').read_bytes()).hexdigest(),
                shared_optimizer_ages=sorted(set(v for k,v in ages.items() if not k.startswith(('regional_objectives.','future_heads.')))),
                legacy_head_ages=sorted(set(v for k,v in ages.items() if k.startswith('future_heads.'))),
                latent_head_ages=sorted(set(v for k,v in ages.items() if k.startswith('regional_objectives.'))),
                grounding=grounding(model,docs),score=scored,
                monitoring=evaluate(model,docs,context=128,span_metrics=True,limit=8192))
    atomic_json(args.run/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('score','grounding')}))


if __name__=='__main__':
    main()
