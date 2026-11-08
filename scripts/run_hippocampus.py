"""Matched A/B/C fixed-budget hippocampus pilot with named optimizer migration."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_paired_recall import PairedRecallRun,write_json
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.hippocampus_config import HippocampusConfig
from experiments.text_regions.hippocampus_checkpoint import load_parent,migrate_optimizer
from experiments.text_regions.memory_selectivity import selective_cases


class HippocampusRun(PairedRecallRun):
    def __init__(self,parent,raw,*,arm='C',steps=64,eval_every=16,validation_cases=8):
        super().__init__(parent,raw,steps=steps,eval_every=eval_every,validation_cases=validation_cases)
        if arm not in ('A','B','C'): raise ValueError('Expected arm A, B or C.')
        if arm!='A':
            old=self.model
            self.cfg=replace(self.cfg,hippocampus=HippocampusConfig(enabled=True,mode='local' if arm=='B' else 'shared'))
            self.model=RegionLanguageModel(self.cfg)
            self.new_parameters=load_parent(self.model,parent)
            self.optimizer=migrate_optimizer(old,parent['optimizer'],self.model,.003)
        else:
            self.new_parameters=()
        self.protocol.update(arm=arm,validation_start=3000,test_seed=1201,test_start=80000,
            checkpoint_selection='Validation candidates reported separately; primary endpoint is update 64',
            hippocampus_source_sha256=hashlib.sha256((ROOT/'experiments/text_regions/hippocampus.py').read_bytes()).hexdigest())
        self.val_cases={f'{mode}_{gap}':selective_cases(self.data_cfg,split='validation',
            seed=683,start=3000,count=validation_cases,mode=mode,gap=gap,distractor_bytes=self.validation)
            for mode in ('single','older','newer','correction') for gap in (0,32)}
        torch.set_rng_state(parent['rng'])
        self.validation_workers=1

    def validate(self):
        if self.validation_workers>1:
            from experiments.text_regions.hippocampus_validation import parallel_validation
            return parallel_validation(self,self.validation_workers)
        return super().validate()

    def state_dict(self):
        return super().state_dict()|dict(kind='hippocampus_v1',optimizer_parameter_names=[n for n,_ in self.model.named_parameters()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'artifacts/hippocampus/parent.pt')
    parser.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    parser.add_argument('--arm',choices=['A','B','C'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--eval-every',type=int,default=16)
    parser.add_argument('--validation-cases',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint-training',action='store_true',help='Recompute activations; does not change the training protocol')
    parser.add_argument('--checkpoint-min-positions',type=int,default=768,help='Batch-byte threshold; zero recomputes every training call')
    parser.add_argument('--validation-workers',type=int,default=1,help='Independent CPU validation cohorts; unchanged measurements')
    parser.add_argument('--initial-reference',type=Path,help='Reuse audited zero-residual arm-A initial validation')
    parser.add_argument('--skip-initial-validation',action='store_true',help='Timing smoke only; do not use for comparisons')
    a=parser.parse_args(); torch.set_num_threads(1)
    if not a.resume and a.out.exists() and any(a.out.iterdir()): parser.error('Use an empty output directory.')
    a.out.mkdir(parents=True,exist_ok=True)
    run=HippocampusRun(torch.load(a.parent,map_location='cpu',weights_only=True),a.data.read_bytes(),
        arm=a.arm,steps=a.steps,eval_every=a.eval_every,validation_cases=a.validation_cases)
    run.protocol['parent_sha256']=hashlib.sha256(a.parent.read_bytes()).hexdigest()
    if not 1<=a.validation_workers<=4: parser.error('Use one through four validation workers.')
    run.validation_workers=a.validation_workers
    if a.checkpoint_training:
        from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing
        enable_associative_checkpointing(run.model,min_positions=a.checkpoint_min_positions)
    if a.resume:
        run.restore(torch.load(a.out/'last.pt',map_location='cpu',weights_only=True))
    else:
        write_json(a.out/'protocol.json',dict(protocol=run.protocol,config=run.cfg.to_dict(),
            data_config=run.data_cfg.to_dict(),raw_sha256=run.raw_sha256,parameters=sum(p.numel() for p in run.model.parameters()),
            new_parameter_tensors=len(run.new_parameters)))
        if not a.skip_initial_validation:
            if a.initial_reference is not None:
                reference=json.loads((a.initial_reference/'protocol.json').read_text())
                for key in ('parent_sha256','validation_start','validation_cases','data_seed','gaps','lr'):
                    if reference['protocol'][key]!=run.protocol[key]: raise ValueError('Initial reference mismatch: '+key)
                if reference['protocol']['arm']!='A' or a.arm=='A': raise ValueError('Require arm A reference for B/C.')
                if (bool(run.model.hippocampus.compose.weight.detach().any()) or
                        bool(run.model.hippocampus.publication_feedback_gain.detach())):
                    raise ValueError('Initial-reference reuse requires exactly zero added output/feedback.')
                initial=json.loads((a.initial_reference/'initial.json').read_text())
                initial['exact_zero_residual_reference']=str(a.initial_reference/'initial.json')
            else:
                initial=run.validate()
            run.best=initial['selection_score']; write_json(a.out/'initial.json',initial)
            print(json.dumps({'stage':'initial','score':run.best}),flush=True)
        run.save(a.out/'best.pt'); run.save(a.out/'last.pt')
    stop=a.steps if a.stop_after is None else a.stop_after
    if not run.step <= stop <= a.steps: parser.error('Invalid stopping point.')
    while run.step<stop:
        row=run.update()
        row['activation_checkpointing']=a.checkpoint_training
        row['checkpoint_min_positions']=a.checkpoint_min_positions if a.checkpoint_training else None
        if run.step%a.eval_every==0 or run.step==a.steps:
            print(json.dumps({'stage':'validation','step':run.step,'workers':run.validation_workers}),flush=True)
            row['validation']=run.validate()
            if row['validation']['selection_score']<run.best:
                run.best=row['validation']['selection_score']; run.save(a.out/'best.pt')
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('step','seconds','memory_nll','language_nll','peak_mib')}),flush=True)

if __name__=='__main__': main()
