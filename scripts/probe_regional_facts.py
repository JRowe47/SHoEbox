"""Frozen local-head diagnostics; never used for checkpoint selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.episode_data import make_batch
from experiments.text_regions.fact_supervision import forward_sources, view_targets
from experiments.text_regions.integrated_train import load_integrated
from experiments.text_regions.view_data import make_view_batch, QUERY_TYPES


@torch.no_grad()
def diagnose(model, count=64, seed=20261107):
    model.eval(); records={}
    batches=[('recall', make_batch('recall', 'test', 0, count, seed=seed, gap=16)),
             ('correction', make_batch('correction', 'test', 0, count, seed=seed, gap=16))]
    batches += [('views/'+kind, make_view_batch('independent', 0, count, seed=seed+104729,
                                               gap=8, query_type=kind)) for kind in QUERY_TYPES]
    sources=forward_sources(model.cfg)
    for task,batch in batches:
        state=None; calls=0
        captured={name:[] for name in model.nodes}
        for pos in range(0,batch.length,16):
            chunk=batch.slice(pos,min(pos+16,batch.length))
            result=model(chunk.inputs,state=state,reset=chunk.reset,availability=chunk.availability,
                         rounds=2,feedback=True,return_node_activity=tuple(model.nodes))
            state=result['state'];calls+=result['expert_executions']
            for name in model.nodes:captured[name].append(result['node_activity'][name]['payload'])
        nodes={}
        for name in model.nodes:
            sequence=torch.cat(captured[name],1)
            if isinstance(batch.inputs,dict):
                labels=view_targets(batch,sources[name])[:,-1]
                logits=model.fact_retention(state.ports.latest[name].payload).reshape(count,3,4)
                factors=QUERY_TYPES
                positions=[]
                for factor in factors:
                    observed=[batch.observation_positions[v][factor] for v in sorted(sources[name])]
                    positions.append(torch.stack(observed).max(0).values if observed
                                     else torch.full((count,),-1,dtype=torch.long))
            else:
                labels=batch.retained_targets[:,-1,None]
                logits=model.memory_retention(state.ports.latest[name].payload)[:,None,:]
                factors=('current',)
                position=torch.tensor([batch.length-1-distance for distance in batch.distances])
                positions=[position]
            nodes[name]={}
            for axis,factor in enumerate(factors):
                mask=labels[:,axis].ne(-100) & state.ports.latest[name].valid
                if not bool(mask.any()): continue
                truth=labels[:,axis][mask];pred=logits[:,axis][mask].argmax(-1)+65
                write_features=sequence[torch.arange(count),positions[axis].clamp_min(0)]
                if isinstance(batch.inputs,dict):
                    write_logits=model.fact_retention(write_features).reshape(count,3,4)[:,axis]
                else:
                    write_logits=model.memory_retention(write_features)
                write_pred=write_logits[mask].argmax(-1)+65
                nodes[name][factor]={'count':int(mask.sum()),'correct':int(pred.eq(truth).sum()),
                                    'accuracy':float(pred.eq(truth).float().mean()),
                                    'predictions':pred.tolist(),'targets':truth.tolist(),
                                    'write_correct':int(write_pred.eq(truth).sum()),
                                    'write_accuracy':float(write_pred.eq(truth).float().mean()),
                                    'write_predictions':write_pred.tolist(),
                                    'write_positions':positions[axis][mask].tolist()}
        records[task]={'nodes':nodes,'expert_executions':calls,'identities':list(batch.identities)}
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    model,_,_,_,saved=load_integrated(args.checkpoint)
    result={'checkpoint_step':saved['step'],'count_per_task':64,
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'selection_use':'none; frozen post-selection local-head diagnostic',
            'records':diagnose(model,64,saved['protocol']['seed'])}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({task:{node:{factor:value['accuracy'] for factor,value in fields.items()}
                                for node,fields in row['nodes'].items()} for task,row in result['records'].items()}))


if __name__=='__main__':main()
