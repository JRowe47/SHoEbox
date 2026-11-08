"""Read-only final mechanistic probes on the locked diagnostic cohort."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_lace_separation import load_model
from scripts.preflight_lace_separation import probe
from scripts.lace_cohorts import from_record
from scripts.run_memory_selectivity import write_json
from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing

def main():
    torch.set_num_threads(1);root=ROOT/'artifacts/lace_separation'
    while not (root/'training_complete.json').exists():time.sleep(3)
    lock=json.loads((root/'cohorts.json').read_text());result={}
    for arm in 'BCD':
        model,saved=load_model(root/arm/'last.pt')
        enable_associative_checkpointing(model,segment=16,min_positions=512)
        result[arm]={}
        for record in lock['cohorts']['diagnostic']['records']:
            if record['template']!=0:continue
            _,metrics=probe(SimpleNamespace(model=model,cfg=model.cfg),from_record(record),True)
            result[arm][str(record['gap'])]=metrics
        result[arm]['head_norms']={n:float(p.detach().norm()) for n,p in model.named_parameters() if '.lace_head.' in n}
        write_json(root/'endpoint_mechanism.json',result)
        print(json.dumps(dict(arm=arm)),flush=True)
if __name__=='__main__':main()
