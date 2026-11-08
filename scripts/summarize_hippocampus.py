"""Audit completed fixed-budget comparisons and assemble the report data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
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


def compact_measurements(measurements):
    # Per-history predictions remain in the checkpoint archive's evaluations.
    # Keep the repository report concise and directly inspectable.
    return {name:{key:value for key,value in row.items() if key!='rows'}
            for name,row in measurements.items()}


def training_storage_probe(model,raw):
    """Unique saved storage in a declared cold-start full-model training call."""
    values=torch.tensor(list(raw[-57:]),dtype=torch.long)
    inputs=values[:-1].reshape(2,28); targets=values[1:].reshape(2,28)
    key=lambda x:(str(x.device),x.untyped_storage().data_ptr())
    excluded={key(p) for p in model.parameters()}|{key(inputs),key(targets)}
    saved={}
    def pack(value):
        if key(value) not in excluded:
            saved[key(value)]=value.untyped_storage().nbytes()
        return value
    model.train();model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
        result=model(inputs,targets=targets)
    forward_saved_bytes=sum(saved.values())
    result['loss'].backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    report=dict(batch=2,context=28,external_positions=56,cold_start=True,
        execution='direct; no optimizer update',saved_tensor_bytes_excluding_weights_inputs=forward_saved_bytes,
        gradient_tensor_bytes=tensor_storage(gradients),gradient_parameter_tensors=len(gradients),
        resulting_state_tensor_bytes=result['state'].storage_bytes())
    model.zero_grad(set_to_none=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT/'artifacts/hippocampus')
    p.add_argument('--out',type=Path,default=ROOT/'experiments/text_regions/results/hippocampus/summary.json')
    a=p.parse_args(); torch.set_num_threads(1); root=a.root
    raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes()
    initial=json.loads((root/'A/initial.json').read_text()); parent_val=initial['cohorts']
    result=dict(parent='artifacts/memory_selectivity/selective/best.pt',arms={},fixed_update_budget=64,
        initial_parent_validation=initial,control_reuse=json.loads((root/'A/control_reuse.json').read_text()),
        structural=json.loads((ROOT/'experiments/text_regions/results/hippocampus/structural.json').read_text()),
        scaling=json.loads((root/'scaling.json').read_text()),cohorts=json.loads((root/'cohorts.json').read_text()),
        execution_policy=json.loads((root/'execution_policy.json').read_text()),
        environment=dict(python=platform.python_version(),torch=torch.__version__,machine=platform.machine(),
            execution_device='cpu',torch_threads=torch.get_num_threads(),
            cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None))
    trajectories=[]
    for arm in ('parent','A','B','C'):
        checkpoint=root/'parent.pt' if arm=='parent' else root/arm/'last.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['raw_sha256']!=hashlib.sha256(raw).hexdigest():
            raise ValueError('Corpus mismatch: '+arm)
        for field in ('weights','optimizer','language_state'):
            if not all(bool(torch.isfinite(value).all()) for value in tensors(saved[field])):
                raise ValueError('Nonfinite checkpoint tensors: '+arm+'/'+field)
        cfg=GraphConfig.from_dict(saved['config']); model=RegionLanguageModel(cfg); model.load_state_dict(saved['weights'])
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        row=dict(checkpoint_sha256=digest,unique_parameters=sum(p.numel() for p in model.parameters()),
            unique_weight_bytes=tensor_storage(model.parameters()),optimizer_tensor_bytes=tensor_storage(tensors(saved['optimizer'])),
            carried_state_tensor_bytes=GraphState.from_dict(saved['language_state']).storage_bytes(),
            checkpoint_file_bytes=checkpoint.stat().st_size,finite_checkpoint_tensors=True,measurements={})
        row['training_storage_probe']=training_storage_probe(model,raw)
        for part in ('pairs','single','correction','streams'):
            evaluated=json.loads((root/'evaluation'/f'{arm}_intact_{part}.json').read_text())
            if evaluated['identity']['checkpoint_sha256']!=digest: raise ValueError('Evaluation checkpoint mismatch.')
            row['measurements'].update(compact_measurements(evaluated['measurements']))
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
                       recomputation_wrapper_enabled_updates=[r['step'] for r in history if r.get('activation_checkpointing')])
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
        result['ablations'][control]=compact_measurements(data['measurements'])
    result['retain_parent']=not result['arms']['C']['promoted']
    result['milestones']={
        'H0':dict(status='pass',scope='Contracts, disabled parity and migration'),
        'H1':dict(status='pass',scope='Regional anchors, conditional readers and live gradients'),
        'H2':dict(status='pass',scope='Two-shard shared service and deterministic causal integration'),
        'H3':dict(status='complete',behavioral_promotion_passed=result['arms']['C']['promoted'],
                  scope='Matched 64-update endpoints, locked tests and five inference controls'),
        'H4':dict(status='pass',scope='Bounded 64-512-owner CPU actor conformance smoke'),
        'H5':dict(status='outside_initial_scope',scope='Physical distributed/GPU deployment')}
    a.out.parent.mkdir(parents=True,exist_ok=True); write_json(a.out,result)
    print(json.dumps({arm:dict(parameters=r['unique_parameters'],both_0=r['measurements']['pairs_0']['both_accuracy'],
         both_32=r['measurements']['pairs_32']['both_accuracy'],language=r['measurements']['language']['nll'],
         promoted=r.get('promoted',False)) for arm,r in result['arms'].items()}))

if __name__=='__main__': main()
