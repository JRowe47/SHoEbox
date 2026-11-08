"""Matched short continuations isolating bounded CA1 reread acceptance."""
import argparse
from dataclasses import replace
import hashlib,json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_hippocampus import HippocampusRun
from scripts.run_paired_recall import PairedRecallRun,write_json
from experiments.text_regions.memory_selectivity import selective_cases
from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing


class CA1Run(HippocampusRun):
    def __init__(self,parent,raw,*,policy='bounded',steps=16):
        PairedRecallRun.__init__(self,parent,raw,steps=steps,eval_every=steps,validation_cases=8)
        if not self.cfg.hippocampus.enabled:raise ValueError('CA1 continuation needs a hippocampus parent.')
        self.cfg=replace(self.cfg,hippocampus=replace(self.cfg.hippocampus,ca1_policy=policy))
        self.model.cfg=self.cfg;self.model.hippocampus.config=self.cfg.hippocampus
        for node in self.model.nodes.values():node.cfg=self.cfg
        self.new_parameters=();self.validation_workers=2
        self.protocol.update(phase='bounded_ca1_continuation',ca1_policy=policy,
            validation_start=3000,test_seed=1699,test_start=160000,
            checkpoint_selection='Fixed update16 endpoint; initial and selected checkpoints preserved',
            hippocampus_source_sha256=hashlib.sha256((ROOT/'experiments/text_regions/hippocampus.py').read_bytes()).hexdigest())
        self.val_cases={f'{mode}_{gap}':selective_cases(self.data_cfg,split='validation',seed=683,
            start=3000,count=8,mode=mode,gap=gap,distractor_bytes=self.validation)
            for mode in ('single','older','newer','correction') for gap in (0,32)}
        torch.set_rng_state(parent['rng'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=ROOT/'artifacts/hippocampus_binding/event_expert/last.pt')
    p.add_argument('--policy',choices=('legacy','bounded'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if not a.resume and a.out.exists() and any(a.out.iterdir()):p.error('Use an empty new output directory.')
    a.out.mkdir(parents=True,exist_ok=True)
    parent=torch.load(a.parent,weights_only=True,map_location='cpu')
    run=CA1Run(parent,(ROOT/'data/tiny_shakespeare.txt').read_bytes(),policy=a.policy)
    run.protocol['parent_sha256']=hashlib.sha256(a.parent.read_bytes()).hexdigest()
    enable_associative_checkpointing(run.model,min_positions=512)
    if a.resume:run.restore(torch.load(a.out/'last.pt',weights_only=True,map_location='cpu'))
    else:
        # Preserve both untrained policies for same-weight causal comparisons.
        run.save(a.out/'initial.pt')
        if a.policy=='legacy':
            history=[json.loads(s) for s in (a.parent.parent/'history.jsonl').read_text().splitlines()]
            if history[-1]['step']!=parent['step']:raise ValueError('Parent history mismatch.')
            initial=history[-1]['validation']|{'exact_parent_validation_reuse':True}
        else:initial=run.validate()
        run.best=initial['selection_score']
        write_json(a.out/'initial.json',initial)
        write_json(a.out/'protocol.json',dict(protocol=run.protocol,config=run.cfg.to_dict(),
            raw_sha256=run.raw_sha256,data_config=run.data_cfg.to_dict(),
            parameters=sum(p.numel() for p in run.model.parameters())))
        run.save(a.out/'best.pt');run.save(a.out/'last.pt')
        print(json.dumps({'stage':'initial','policy':a.policy,'pairs':initial['paired']}),flush=True)
    while run.step<16:
        row=run.update();row.update(activation_checkpointing=True,checkpoint_min_positions=512)
        if run.step==16:
            print(json.dumps({'stage':'validation','step':16}),flush=True)
            row['validation']=run.validate()
            if row['validation']['selection_score']<run.best:
                run.best=row['validation']['selection_score'];run.save(a.out/'best.pt')
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('step','seconds','memory_nll','language_nll','peak_mib')}),flush=True)

if __name__=='__main__':main()
