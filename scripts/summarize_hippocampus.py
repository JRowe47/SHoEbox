"""Audit completed fixed-budget comparisons and assemble the report data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.state import GraphState
from scripts.run_memory_selectivity import write_json


def tensor_storage(values):
    return sum({(str(x.device),x.untyped_storage().data_ptr()):x.untyped_storage().nbytes() for x in values}.values())


def tensors(value):
    if torch.is_tensor(value): return [value]
    if isinstance(value,dict): return sum((tensors(v) for v in value.values()),[])
    if isinstance(value,(list,tuple)): return sum((tensors(v) for v in value),[])
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT/'artifacts/hippocampus')
    p.add_argument('--out',type=Path,default=ROOT/'experiments/text_regions/results/hippocampus/summary.json')
    a=p.parse_args(); torch.set_num_threads(1); root=a.root
    initial=json.loads((root/'A/initial.json').read_text()); parent_val=initial['cohorts']
    result=dict(parent='artifacts/memory_selectivity/selective/best.pt',arms={},fixed_update_budget=64,
        initial_parent_validation=initial,control_reuse=json.loads((root/'A/control_reuse.json').read_text()),
        structural=json.loads((ROOT/'experiments/text_regions/results/hippocampus/structural.json').read_text()),
        scaling=json.loads((root/'scaling.json').read_text()),cohorts=json.loads((root/'cohorts.json').read_text()))
    trajectories=[]
    for arm in ('parent','A','B','C'):
        checkpoint=root/'parent.pt' if arm=='parent' else root/arm/'last.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=GraphConfig.from_dict(saved['config']); model=RegionLanguageModel(cfg); model.load_state_dict(saved['weights'])
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        row=dict(checkpoint_sha256=digest,unique_parameters=sum(p.numel() for p in model.parameters()),
            unique_weight_bytes=tensor_storage(model.parameters()),optimizer_tensor_bytes=tensor_storage(tensors(saved['optimizer'])),
            carried_state_tensor_bytes=GraphState.from_dict(saved['language_state']).storage_bytes(),
            checkpoint_file_bytes=checkpoint.stat().st_size,measurements={})
        for part in ('pairs','single','correction','streams'):
            evaluated=json.loads((root/'evaluation'/f'{arm}_intact_{part}.json').read_text())
            if evaluated['identity']['checkpoint_sha256']!=digest: raise ValueError('Evaluation checkpoint mismatch.')
            row['measurements'].update(evaluated['measurements'])
        if arm!='parent':
            history=[json.loads(x) for x in (root/arm/'history.jsonl').read_text().splitlines()]
            if saved['step']!=64 or len(history)!=64: raise ValueError('Incomplete training arm.')
            trajectory=[{k:r[k] for k in ('step','next_document','mode','gap','totals')} for r in history]
            trajectories.append(trajectory)
            final=history[-1]['validation']
            selected=torch.load(root/arm/'best.pt',map_location='cpu',weights_only=True)
            row.update(final_validation=final,selected_update=selected['step'],selected_validation_score=selected['best'],
                       totals=saved['totals'],next_document=saved['next_document'],
                       training_seconds=sum(r['seconds'] for r in history),peak_process_mib=max(r['peak_mib'] for r in history),
                       activation_recomputation_updates=[r['step'] for r in history if r.get('activation_checkpointing')])
            gates=dict(paired=all(final['paired'][str(g)]['accuracy']>=.5 for g in (0,32)),
                single=all(final['cohorts'][f'single_{g}']['accuracy']>=parent_val[f'single_{g}']['accuracy']-.02 for g in (0,32)),
                correction=all(final['cohorts'][f'correction_{g}']['accuracy']>=parent_val[f'correction_{g}']['accuracy']-.03 for g in (0,32)),
                language=final['language']['nll']<=initial['language']['nll']+.05)
            row['promotion_gates']=gates; row['promoted']=all(gates.values()) and arm!='A'
        result['arms'][arm]=row
    if trajectories[0]!=trajectories[1] or trajectories[0]!=trajectories[2]: raise ValueError('Source trajectories differ.')
    result['source_trajectories_identical']=True
    result['ablations']={}
    c_digest=result['arms']['C']['checkpoint_sha256']
    for control in ('blocked','shuffled','pooled','no_refinement','no_feedback'):
        data=json.loads((root/'evaluation'/f'C_{control}_pairs.json').read_text())
        if data['identity']['checkpoint_sha256']!=c_digest: raise ValueError('Ablation identity mismatch.')
        result['ablations'][control]=data['measurements']
    result['retain_parent']=not result['arms']['C']['promoted']
    a.out.parent.mkdir(parents=True,exist_ok=True); write_json(a.out,result)
    print(json.dumps({arm:dict(parameters=r['unique_parameters'],both_0=r['measurements']['pairs_0']['both_accuracy'],
         both_32=r['measurements']['pairs_32']['both_accuracy'],language=r['measurements']['language']['nll'],
         promoted=r.get('promoted',False)) for arm,r in result['arms'].items()}))

if __name__=='__main__': main()
