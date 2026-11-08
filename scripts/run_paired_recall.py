"""Train both query directions from shared histories, with retention anchors."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_memory_selectivity import SelectivityRun, write_json
from experiments.text_regions.memory_selectivity import selective_cases, summarize_values
from experiments.text_regions.memory_acquisition import pack_cases, prefix_forward
from experiments.text_regions.paired_recall import paired_batch, paired_forward
from experiments.text_regions.task_memory_train import byte_loss
from experiments.text_regions.mixed_gradients import clip_mixed_gradients
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language
from scripts.evaluate_memory_selectivity import paired_queries


MODES = ('paired','paired','single','correction')


class PairedRecallRun(SelectivityRun):
    def __init__(self,parent,raw,*,lr=.003,steps=64,gaps=(0,4,8,16,32),
                 language_weight=.25,eval_every=16,validation_cases=4):
        super().__init__(parent,raw,lr=lr,steps=steps,gaps=gaps,
            language_weight=language_weight,eval_every=eval_every,validation_cases=validation_cases)
        self.protocol.update(modes=list(MODES),validation_start=2000,
            selection='Minimum mean older/newer NLL over gaps 0 and 32; earlier ties win',
            supervision='Both query directions share one live differentiable history per value pair',
            generator_sha256=hashlib.sha256((ROOT/'experiments/text_regions/paired_recall.py').read_bytes()).hexdigest(),
            base_generator_sha256=hashlib.sha256((ROOT/'experiments/text_regions/memory_selectivity.py').read_bytes()).hexdigest())
        self.totals=dict(memory_targets=0,memory_bytes=0,memory_padded_positions=0,
            executed_positions=0,language_bytes=0,by_mode={mode:0 for mode in ('paired','single','correction')})
        self.val_cases={f'{mode}_{gap}':selective_cases(self.data_cfg,split='validation',
            seed=self.protocol['data_seed'],start=2000,count=validation_cases,mode=mode,gap=gap,
            distractor_bytes=self.validation) for mode in ('single','older','newer','correction') for gap in (0,32)}

    def validate(self):
        raw={key:evaluate_values(self.model,cases,context=32,cases_per_batch=8)
             for key,cases in self.val_cases.items()}
        cohorts={key:summarize_values(value) for key,value in raw.items()}
        score=sum(cohorts[f'{mode}_{gap}']['nll'] for mode in ('older','newer') for gap in (0,32))/4
        pairs={str(gap):paired_queries(raw[f'older_{gap}'],raw[f'newer_{gap}'],self.data_cfg.values)
               for gap in (0,32)}
        return dict(selection_score=score,cohorts=cohorts,paired=pairs,
            language=evaluate_language(self.model,self.validation,batch=4,context=32,chunks=2))

    def update(self):
        started=time.perf_counter(); p=self.protocol
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        language=self.streams.next_chunk(p['language_context'])
        result=self.model(language.values,state=self.language_state,
            reset=language.reset,targets=language.targets)
        language_nll=float(result['ce'].detach())
        (p['language_weight']*result['loss']).backward()
        self.language_state=result['state'].detach(); del result
        mode=MODES[self.step%len(MODES)]; gap=p['gaps'][self.step%len(p['gaps'])]
        options=dict(split='train',seed=p['data_seed'],start=self.next_document,
                     count=1,gap=gap,distractor_bytes=self.train)
        if mode == 'paired':
            batch=paired_batch(self.data_cfg,**options)
            result=paired_forward(self.model,batch)
            answers,ordinary=batch['answers'],batch['syntax']
            memory_bytes=batch['full'].numel(); padded=memory_bytes
            executed=result['executed_positions']
        else:
            options['count']=len(self.data_cfg.values) if mode == 'single' else 1
            packed=pack_cases(selective_cases(self.data_cfg,mode=mode,**options))
            result=prefix_forward(self.model,packed)
            ordinary,answers=packed[1:3]
            memory_bytes=int(packed[3].sum()); padded=packed[0].numel(); executed=padded
        logits=result['logits'][:,-1]
        primary=F.cross_entropy(logits,answers)
        syntax=byte_loss(result['logits'],ordinary)
        loss=(primary+p['syntax_weight']*syntax+self.cfg.auxiliary_loss_weight*result['aux']
              +self.cfg.hat_balance_weight*result['hat_aux'])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite memory objective.')
        loss.backward()
        norm,recovered=clip_mixed_gradients(self.model.parameters(),p['global_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite gradient norm.')
        self.optimizer.step(); self.step+=1; self.next_document+=options['count']
        self.totals['memory_targets']+=answers.numel()
        self.totals['memory_bytes']+=memory_bytes
        self.totals['memory_padded_positions']+=padded
        self.totals['executed_positions']+=executed
        self.totals['language_bytes']+=language.values.numel()
        self.totals['by_mode'][mode]+=answers.numel()
        row=dict(step=self.step,next_document=self.next_document,mode=mode,gap=gap,
            language_nll=language_nll,memory_nll=float(primary.detach()),syntax_nll=float(syntax.detach()),
            train_accuracy=float((logits.argmax(-1)==answers).float().mean()),
            gradient_norm=float(norm),fp64_norm_recovered=recovered,
            totals=json.loads(json.dumps(self.totals)),seconds=time.perf_counter()-started,
            peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if mode == 'paired':
            correct=(logits.argmax(-1)==answers).reshape(-1,2)
            different=answers.reshape(-1,2).diff(dim=1).squeeze(1).ne(0)
            row['paired_accuracy']=float(correct[different].all(-1).float().mean())
            row['older_accuracy']=float(correct[:,0].float().mean())
            row['newer_accuracy']=float(correct[:,1].float().mean())
        return row

    def state_dict(self):
        return super().state_dict() | dict(kind='paired_recall')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--gaps',type=int,nargs='+',default=[0,4,8,16,32])
    p.add_argument('--language-weight',type=float,default=.25)
    p.add_argument('--eval-every',type=int,default=16)
    p.add_argument('--validation-cases',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); torch.set_num_threads(1)
    stop=a.steps if a.stop_after is None else a.stop_after
    if not 1<=stop<=a.steps: p.error('Stopping point must lie inside declared budget.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()):
        p.error('New experiments require an empty output directory.')
    run=PairedRecallRun(torch.load(a.parent,map_location='cpu',weights_only=True),a.data.read_bytes(),
        lr=a.lr,steps=a.steps,gaps=a.gaps,language_weight=a.language_weight,
        eval_every=a.eval_every,validation_cases=a.validation_cases)
    run.protocol['parent_sha256']=hashlib.sha256(a.parent.read_bytes()).hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        run.restore(torch.load(a.out/'last.pt',map_location='cpu',weights_only=True))
        if not (a.out/'best.pt').exists(): raise ValueError('Resume requires best.pt.')
    else:
        initial=run.validate(); run.best=initial['selection_score']
        run.save(a.out/'best.pt'); run.save(a.out/'last.pt'); write_json(a.out/'initial.json',initial)
        write_json(a.out/'protocol.json',dict(protocol=run.protocol,config=run.cfg.to_dict(),
            data_config=run.data_cfg.to_dict(),raw_sha256=run.raw_sha256,
            parameters=sum(p.numel() for p in run.model.parameters())))
        print(json.dumps(dict(stage='initial',validation=initial)),flush=True)
    if stop<run.step: raise ValueError('Stopping point precedes saved step.')
    while run.step<stop:
        row=run.update()
        if run.step%a.eval_every==0 or run.step==a.steps:
            row['validation']=run.validate()
            if row['validation']['selection_score']<run.best:
                run.best=row['validation']['selection_score']; run.save(a.out/'best.pt')
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)


if __name__=='__main__': main()
