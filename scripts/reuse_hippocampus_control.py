"""Verify and reuse the prior fixed-budget control, as allowed by the v1 spec."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_hippocampus import HippocampusRun,write_json
from experiments.text_regions.state import GraphState


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/hippocampus/A')
    a=p.parse_args(); torch.set_num_threads(1)
    old=[json.loads(x) for x in (a.prior/'history.jsonl').read_text().splitlines()]
    new=[json.loads(x) for x in (a.out/'history.jsonl').read_text().splitlines()]
    keys=('step','next_document','mode','gap','language_nll','memory_nll','syntax_nll','train_accuracy','gradient_norm','totals')
    for x,y in zip(old,new):
        for key in keys:
            if x[key]!=y[key]: raise ValueError(f'Trajectory mismatch: {x["step"]} {key}')
    if len(new)<16 or old[-1]['step']!=64: raise ValueError('Insufficient reproduction or incomplete prior control.')
    parent_path=ROOT/'artifacts/hippocampus/parent.pt'; raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes()
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    source=a.prior/'last.pt'; saved=torch.load(source,map_location='cpu',weights_only=True)
    digest=hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if saved['protocol']['parent_sha256']!=digest or saved['raw_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Parent/corpus identity mismatch.')
    run=HippocampusRun(parent,raw,arm='A'); run.protocol['parent_sha256']=digest
    for key in ('lr','steps','gaps','language_weight','data_seed','language_batch','language_context',
                'initial_document','modes','syntax_weight','global_clip'):
        if saved['protocol'][key]!=run.protocol[key]: raise ValueError('Control protocol mismatch: '+key)
    run.model.load_state_dict(saved['weights']); run.optimizer.load_state_dict(saved['optimizer'])
    run.language_state=GraphState.from_dict(saved['language_state']); run.streams.load_state_dict(saved['language_stream'])
    torch.set_rng_state(saved['rng']); run.step=64; run.next_document=saved['next_document']; run.totals=saved['totals']
    validation=run.validate(); run.best=validation['selection_score']
    proof=dict(reproduced_updates=len(new),exact_fields=list(keys),all_values_equal=True,
        prior_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parent_sha256=digest,
        raw_sha256=saved['raw_sha256'],fixed_endpoint=64,
        selection='Control uses the verified prior fixed endpoint; intermediate validation selection is not compared',
        validation_seed=683,validation_start=3000,validation_templates=8)
    write_json(a.out/'control_reuse.json',proof)
    (a.out/'reproduction_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
    run.save(a.out/'last.pt'); run.save(a.out/'best.pt')
    write_json(a.out/'protocol.json',dict(protocol=run.protocol,config=run.cfg.to_dict(),
        data_config=run.data_cfg.to_dict(),raw_sha256=run.raw_sha256,parameters=sum(p.numel() for p in run.model.parameters()),
        new_parameter_tensors=0,control_reuse=proof))
    history=[]
    for i,row in enumerate(old):
        row=deepcopy(new[i] if i<len(new) else row)
        if i>=len(new):
            row.pop('validation',None); row['source_run']='verified prior paired control'
        if row['step']==64: row['validation']=validation
        history.append(row)
    temporary=a.out/'history.tmp'; temporary.write_text(''.join(json.dumps(r)+'\n' for r in history)); temporary.replace(a.out/'history.jsonl')
    print(json.dumps(proof|{'validation':validation}),flush=True)

if __name__=='__main__': main()
