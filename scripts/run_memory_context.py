"""One ordinary-loss continuation; no additional lace-credit training rules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_paired_recall import PairedRecallRun,write_json
from experiments.text_regions.memory_context_checkpoint import migrate_memory_context
from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing


def source_hashes():
    paths = sorted((ROOT/'experiments').rglob('*.py'))
    paths += [ROOT/'scripts/run_paired_recall.py',ROOT/'scripts/run_memory_selectivity.py',Path(__file__)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class MemoryContextRun(PairedRecallRun):
    def __init__(self,parent,raw,steps=8):
        compatible = dict(parent,kind='paired_recall') if parent.get('kind') == 'memory_context_v1' else parent
        super().__init__(compatible,raw,lr=parent['protocol']['lr'],steps=steps,eval_every=steps,
                         validation_cases=1,gaps=(0,4,8,16,32))
        self.model,self.optimizer,self.missing = migrate_memory_context(parent)
        self.cfg = self.model.cfg
        self.protocol.update(phase='memory_context_v1',source_hashes=source_hashes(),
            selection='Fixed endpoint; compact integration acceptance, no checkpoint promotion',
            checkpoint_segment=8,checkpoint_min_positions=256)
        torch.set_rng_state(parent['rng'])
        enable_associative_checkpointing(self.model,segment=8,min_positions=256)

    def state_dict(self):
        return super().state_dict() | dict(kind='memory_context_v1',
            optimizer_parameter_names=[n for n,_ in self.model.named_parameters()])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    stop=a.steps if a.stop_after is None else a.stop_after
    if not 1 <= stop <= a.steps: p.error('Invalid continuation endpoint.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()): p.error('Use a new output directory.')
    parent=torch.load(a.parent,weights_only=True,map_location='cpu')
    run=MemoryContextRun(parent,a.data.read_bytes(),a.steps)
    run.protocol['parent_sha256']=hashlib.sha256(a.parent.read_bytes()).hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        run.restore(torch.load(a.out/'last.pt',weights_only=True,map_location='cpu'))
    else:
        run.save(a.out/'initial.pt')
        write_json(a.out/'protocol.json',dict(config=run.cfg.to_dict(),protocol=run.protocol,
            new_parameters=run.missing,parameters=run.model.parameter_summary()))
    if stop < run.step: p.error('Endpoint precedes saved state.')
    while run.step < stop:
        row=run.update()
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('step','mode','memory_nll','language_nll','seconds','peak_mib')}),flush=True)


if __name__ == '__main__':main()
