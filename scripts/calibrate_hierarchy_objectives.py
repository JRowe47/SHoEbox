"""Calibrate Phase A on disjoint training-only windows without weight updates."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shoebox.checkpoint import load, save
from shoebox.hierarchy_config import ObjectiveConfig
from shoebox.hierarchy_training import regional_optimizer, restore_named_optimizer
from shoebox.train_hierarchy import load_data, training_document_ids
from shoebox.train_fineweb import atomic_json, source_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--windows', type=int, default=2)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,state,_,record=load(args.checkpoint)
    original={k:v.clone() for k,v in model.state_dict().items()}
    cfg=ObjectiveConfig(mode='latent')
    model.configure_objective(cfg)
    optimizer=regional_optimizer(model,.001)
    migration=restore_named_optimizer(model,optimizer,record)
    train,_,manifest=load_data(args.data)
    docs=training_document_ids(manifest)
    measurements=[]
    state=None
    for i in range(args.windows):
        cursor=i*256
        out=model(train[:,cursor:cursor+256],train[:,cursor+1:cursor+257],state,
                  document_ids=docs[:,cursor:cursor+256])
        row={}
        for name in ('L2','S2','L3'):
            parameter=model.regions[name].publish.weight
            ce,=torch.autograd.grad(out['ce'],parameter,retain_graph=True)
            latent,=torch.autograd.grad(out['regional']['losses'][name],parameter,retain_graph=True)
            row[name]=dict(global_norm=float(ce.norm()),local_norm=float(latent.norm()),
                           loss=float(out['regional']['losses'][name].detach()),
                           pairs=out['regional']['pairs'][name])
        state=out['state'].detach()
        measurements.append(row)
        print(json.dumps(dict(window=i+1,gradients=row)),flush=True)
        del out
    # Aim for 10% of the global-byte gradient at each owning publisher.
    # Bounds prevent a tiny global gradient from eliminating the experiment.
    coefficients={}
    for name in ('L2','S2','L3'):
        global_norm=sum(r[name]['global_norm'] for r in measurements)/len(measurements)
        local_norm=sum(r[name]['local_norm'] for r in measurements)/len(measurements)
        coefficients[name.lower()+'_weight']=max(.005,min(.2,.1*global_norm/max(local_norm,1e-12)))
    cfg=replace(cfg,**coefficients)
    model.configure_objective(cfg)
    for k,v in original.items():
        torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(objective=cfg.to_dict(),measurements=measurements,optimizer_migration=migration,
                source_sha256=source_digest(),
                shared_origin_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                calibration_sha256=manifest['files']['train']['sha256'],
                calibration_observations=args.windows*train.shape[0]*256,
                policy='Mean publisher gradient ratio: 0.1 * global / latent; clip coefficient to [0.005, 0.2]. No weight updates.',
                shared_weights_exact=True)
    atomic_json(args.out/'calibration.json',result)
    atomic_json(args.out/'objective.json',cfg.to_dict())
    # Training data gets fresh runtime state; normalizer statistics and new heads
    # are retained. Lifetime byte exposures count optimization, not calibration.
    metadata=dict(record['metadata'],calibration=result)
    save(args.out/'seed.pt',model,None,optimizer,step=record['step'],metadata=metadata)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
