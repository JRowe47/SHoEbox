"""Validate identities/source schedules and summarize the binding experiment."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from scripts.summarize_hippocampus import compact_measurements,tensors
from scripts.run_memory_selectivity import write_json


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def main():
    torch.set_num_threads(1)
    root=ROOT/'artifacts/hippocampus_binding'; evaluation=root/'evaluation'
    prior=ROOT/'artifacts/hippocampus'; initial=read(prior/'A/initial.json')
    checkpoints=dict(parent=prior/'parent.pt',legacy=prior/'C/last.pt',event_expert=root/'event_expert/last.pt')
    result=dict(cohort=dict(split='test',seed=1511,start=120000,pair_templates=16,
                            single_templates=32,paired_gaps=[0,32,64],retention_gap=64),models={})
    trajectories=[]
    for name,path in checkpoints.items():
        saved=torch.load(path,weights_only=True,map_location='cpu')
        for field in ('weights','optimizer','language_state'):
            if any(not bool(torch.isfinite(x).all()) for x in tensors(saved[field])):
                raise ValueError('Nonfinite checkpoint tensors: '+name+'/'+field)
        cfg=GraphConfig.from_dict(saved['config']); model=RegionLanguageModel(cfg)
        model.load_state_dict(saved['weights'])
        row=dict(checkpoint_sha256=digest(path),parameters=sum(p.numel() for p in model.parameters()),measurements={})
        for part in ('pairs','single','correction','language'):
            evaluated=read(evaluation/(name+'_'+part+'.json')); identity=evaluated['identity']
            for key,value in dict(checkpoint_sha256=digest(path),seed=1511,start=120000,
                                  pair_templates=16,single_templates=32,raw_sha256=saved['raw_sha256']).items():
                if identity[key]!=value: raise ValueError(f'Evaluation identity mismatch: {name}/{key}')
            row['measurements'].update(compact_measurements(evaluated['measurements']))
        if name!='parent':
            history=[json.loads(s) for s in (path.parent/'history.jsonl').read_text().splitlines()]
            if len(history)!=64 or saved['step']!=64: raise ValueError('Incomplete training run.')
            trajectories.append([{k:r[k] for k in ('step','next_document','mode','gap','totals')} for r in history])
            final=history[-1]['validation']
            gates=dict(paired=all(final['paired'][str(g)]['accuracy']>=.5 for g in (0,32)),
                single=all(final['cohorts'][f'single_{g}']['accuracy']>=initial['cohorts'][f'single_{g}']['accuracy']-.02 for g in (0,32)),
                correction=all(final['cohorts'][f'correction_{g}']['accuracy']>=initial['cohorts'][f'correction_{g}']['accuracy']-.03 for g in (0,32)),
                language=final['language']['nll']<=initial['language']['nll']+.05)
            selected=torch.load(path.parent/'best.pt',weights_only=True,map_location='cpu')
            row.update(promotion_gates=gates,promoted=all(gates.values()),validation=final,
                selected_update=selected['step'],selected_validation_score=selected['best'],
                training_seconds=sum(r['seconds'] for r in history),peak_mib=max(r['peak_mib'] for r in history),
                totals=saved['totals'],next_document=saved['next_document'],
                validation_curve=[dict(step=r['step'],score=r['validation']['selection_score'],
                     pairs=r['validation']['paired']) for r in history if 'validation' in r])
        result['models'][name]=row
    if trajectories[0]!=trajectories[1]: raise ValueError('Source schedules do not match.')
    result['source_schedules_identical']=True
    result['controls']={}
    for name in ('blocked','round_cue','ingress_read'):
        data=read(evaluation/(name+'.json'))
        if data['identity']['checkpoint_sha256']!=result['models']['event_expert']['checkpoint_sha256']:
            raise ValueError('Control checkpoint mismatch.')
        if (data['identity']['seed'],data['identity']['start'])!=(1511,120000):
            raise ValueError('Control cohort mismatch.')
        result['controls'][name]=compact_measurements(data['measurements'])
    result['diagnostics']={name:read(root/(name+'_audit.json')) for name in ('before','after')}
    result['retain_parent']=not result['models']['event_expert']['promoted']
    result['tests']={}
    for name in ('legacy','binding','runtime'):
        text=(root/(name+'_tests.log')).read_text()
        if not text.rstrip().endswith('OK'):raise ValueError('Test suite did not pass: '+name)
        import re
        result['tests'][name]=int(re.search(r'Ran (\d+) tests?',text).group(1))
    output=ROOT/'experiments/text_regions/results/hippocampus_binding/summary.json'
    output.parent.mkdir(exist_ok=True);write_json(output,result)
    print(json.dumps({n:dict(pairs64=v['measurements']['pairs_64']['both_correct'],
          single64=v['measurements']['single_64']['accuracy'],correction64=v['measurements']['correction_64']['accuracy'],
          promoted=v.get('promoted',False)) for n,v in result['models'].items()}))

if __name__=='__main__':main()
