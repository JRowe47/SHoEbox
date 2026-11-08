"""Four locked, default-off lace-separation continuations of bounded CA1."""
import argparse
from dataclasses import replace
import hashlib,json,resource,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_paired_recall import PairedRecallRun, MODES, write_json
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.hippocampus_checkpoint import migrate_optimizer
from experiments.text_regions.lace_separation import LaceSeparationConfig,LaceCredit
from experiments.text_regions.memory_selectivity import selective_cases
from experiments.text_regions.memory_acquisition import pack_cases,prefix_forward
from experiments.text_regions.paired_recall import paired_batch,paired_forward
from experiments.text_regions.task_memory_train import byte_loss
from experiments.text_regions.mixed_gradients import clip_mixed_gradients
from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing


def source_hashes():
    paths=sorted((ROOT/'experiments').rglob('*.py'))
    paths += [ROOT/'scripts/run_paired_recall.py',ROOT/'scripts/run_memory_selectivity.py',Path(__file__)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class LaceRun(PairedRecallRun):
    def __init__(self,parent,raw,arm='D',steps=32):
        super().__init__(parent,raw,lr=parent['protocol']['lr'],steps=steps,eval_every=steps,validation_cases=8)
        if arm not in 'ABCD' or len(arm)!=1:raise ValueError('Expected A/B/C/D.')
        if (self.cfg.hippocampus.ca1_policy,self.cfg.hippocampus.cue_clock,self.cfg.hippocampus.reinstatement)!=('bounded','event','expert'):
            raise ValueError('Require the bounded CA1 fixed endpoint.')
        old=self.model
        option=LaceSeparationConfig(installed=True,enabled=arm!='A',alpha=.1 if arm in 'CD' else 0.,beta=.05 if arm=='D' else 0.)
        self.cfg=replace(self.cfg,lace_separation=option)
        self.model=RegionLanguageModel(self.cfg)
        missing,unexpected=self.model.load_state_dict(parent['weights'],strict=False)
        if unexpected or any('.lace_head.' not in name for name in missing):raise ValueError('Invalid parent migration.')
        self.optimizer=migrate_optimizer(old,parent['optimizer'],self.model,parent['protocol']['lr'])
        self.protocol.update(phase='lace_separation_v1',arm=arm,validation_start=200000,
            validation_seed=2411,test_seed=2539,test_start=240000,
            selection='Fixed update-32 primary; initial and final validation only',
            source_hashes=source_hashes(),checkpoint_segment=16,checkpoint_min_positions=512,
            starting_public_revision='f762518a30c7cc01cc308a4a362a3be73d62b52e')
        for key in ('language_weight','syntax_weight','global_clip','gaps'):
            self.protocol[key]=parent['protocol'][key]
        torch.set_rng_state(parent['rng'])

    def state_dict(self):
        return super().state_dict()|dict(kind='lace_separation_v1',optimizer_parameter_names=[n for n,_ in self.model.named_parameters()])

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
            credit=LaceCredit(batch['history'].shape[1]+batch['query'].shape[1]-1,self.cfg.lace_separation)
            self.model._lace_capture=credit
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
        if mode == 'paired': credit.supervise(logits,answers)
        loss.backward()
        mechanism=credit.apply() if mode == 'paired' else {}
        self.model._lace_capture=None
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
        work=result.get('hippocampus_work',[])
        memory_work={key:sum(w.get(key,0) for w in work) for key in ('posting_reads','candidates','readbacks','refinements')}
        row=dict(step=self.step,next_document=self.next_document,mode=mode,gap=gap,
            language_nll=language_nll,memory_nll=float(primary.detach()),syntax_nll=float(syntax.detach()),
            train_accuracy=float((logits.argmax(-1)==answers).float().mean()),
            gradient_norm=float(norm),fp64_norm_recovered=recovered,
            mechanism=mechanism, answer_targets=answers.tolist(), memory_work=memory_work,
            expert_executions=result.get('expert_executions', executed*len(self.cfg.nodes)*self.cfg.rounds*3*self.cfg.active_groups*self.cfg.experts_per_group),
            totals=json.loads(json.dumps(self.totals)),seconds=time.perf_counter()-started,
            peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if mode == 'paired':
            correct=(logits.argmax(-1)==answers).reshape(-1,2)
            different=answers.reshape(-1,2).diff(dim=1).squeeze(1).ne(0)
            row['paired_accuracy']=float(correct[different].all(-1).float().mean())
            row['older_accuracy']=float(correct[:,0].float().mean())
            row['newer_accuracy']=float(correct[:,1].float().mean())
        return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=ROOT/'artifacts/hippocampus_binding/ca1/bounded/last.pt')
    p.add_argument('--arm',choices=list('ABCD'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,default=32)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--cohorts',type=Path,default=ROOT/'artifacts/lace_separation/cohorts.json')
    a=p.parse_args();torch.set_num_threads(1)
    if not 1<=a.stop_after<=32:p.error('Endpoint must be in 1..32.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()):p.error('Use a fresh output directory.')
    parent=torch.load(a.parent,weights_only=True,map_location='cpu')
    run=LaceRun(parent,(ROOT/'data/tiny_shakespeare.txt').read_bytes(),a.arm)
    run.protocol.update(parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
        cohorts_sha256=hashlib.sha256(a.cohorts.read_bytes()).hexdigest())
    enable_associative_checkpointing(run.model,segment=16,min_positions=512)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.resume:run.restore(torch.load(a.out/'last.pt',weights_only=True,map_location='cpu'))
    else:
        initial=json.loads(a.initial.read_text())
        if initial['checkpoint_sha256']!=run.protocol['parent_sha256']:raise ValueError('Wrong initial reference.')
        run.best=initial['selection_score']
        write_json(a.out/'initial.json',initial)
        write_json(a.out/'protocol.json',dict(protocol=run.protocol,config=run.cfg.to_dict(),
            parameters=sum(p.numel() for p in run.model.parameters()),raw_sha256=run.raw_sha256))
        run.save(a.out/'initial.pt');run.save(a.out/'last.pt')
    while run.step<a.stop_after:
        try:row=run.update()
        finally:run.model._lace_capture=None
        row.update(activation_checkpointing=True,checkpoint_segment=16,checkpoint_min_positions=512)
        run.save(a.out/'last.pt')
        if run.step%8==0:run.save(a.out/f"update_{run.step:02}.pt")
        with (a.out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('step','seconds','memory_nll','language_nll','peak_mib')}),flush=True)

if __name__=='__main__':main()
