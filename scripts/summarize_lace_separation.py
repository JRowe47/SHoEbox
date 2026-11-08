"""Audit fixed endpoints and generate the compact, measured pilot result."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_memory_selectivity import write_json


def main():
    root=ROOT/'artifacts/lace_separation';resultdir=ROOT/'experiments/text_regions/results/lace_separation'
    read=lambda p:json.loads(p.read_text())
    completed=read(root/'training_complete.json');assert completed['matched_ledgers']
    cohort_hash=hashlib.sha256((root/'cohorts.json').read_bytes()).hexdigest()
    evaluator_hash=hashlib.sha256((ROOT/'scripts/evaluate_lace_separation.py').read_bytes()).hexdigest()
    checkpoints={a:root/a/'last.pt' for a in 'ABCD'}
    checkpoints['parent']=ROOT/'artifacts/hippocampus/parent.pt'
    checkpoint_hashes={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in checkpoints.items()}
    evaluation_files={}
    def read_eval(name,cohort,part,control='intact'):
        label=part if control=='intact' else control
        path=root/'evaluation'/f'{name}_{cohort}_{label}.json';j=read(path)
        identity=j['identity']
        assert identity['checkpoint_sha256']==checkpoint_hashes[name]
        assert identity['cohorts_sha256']==cohort_hash
        assert identity['evaluator_sha256']==evaluator_hash
        assert (identity['cohort'],identity['part'],identity['control'])==(cohort,part,control)
        expected=3 if part in ('pairs','unseen') else (4 if cohort=='validation' else 2) if part=='retention' else 1
        assert len(j['measurements'])==expected,(name,cohort,part,control)
        if part in ('pairs','unseen'):
            count=96 if cohort=='validation' else 192
            for v in j['measurements'].values():
                assert v['conflicting_pairs']==count and len(v['rows'])==count*4//3
        evaluation_files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return j
    measurements={}
    for name in ('A','B','C','D','parent'):
        measurements[name]={}
        for cohort in ('validation','test'):
            combined={}
            for part in ('pairs','retention','language')+(('unseen',) if cohort=='test' else ()):
                j=read_eval(name,cohort,part)
                combined.update({key:{k:v for k,v in value.items() if k!='rows'} for key,value in j['measurements'].items()})
            measurements[name][cohort]=combined
    ablations={c:{k:{x:y for x,y in v.items() if x!='rows'} for k,v in
        read_eval('D','test','pairs',c)['measurements'].items()} for c in ('bias_off','cue_permuted','blocked')}
    assert len(evaluation_files)==38
    intact=read(root/'evaluation/D_test_pairs.json')['measurements']
    swapped=read(root/'evaluation/D_test_cue_permuted.json')['measurements']
    cue_swap={}
    for key in intact:
        pairs=list(zip(intact[key]['rows'],swapped[key]['rows']))
        assert all(a['case_sha256']==b['case_sha256'] for a,b in pairs)
        cue_swap[key]=dict(answers=2*len(pairs),
            changed_predictions=sum(a[f'{q}_prediction']!=b[f'{q}_prediction'] for a,b in pairs for q in ('older','newer')),
            max_absolute_margin_change=max(abs(a[f'{q}_margin']-b[f'{q}_margin']) for a,b in pairs for q in ('older','newer')))
    def same(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    initial_reference=torch.load(root/'A/initial.pt',weights_only=True,map_location='cpu')
    for path,digest in initial_reference['protocol']['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for arm in 'BCD':
        candidate=torch.load(root/arm/'initial.pt',weights_only=True,map_location='cpu')
        for key in ('weights','optimizer','language_state','language_stream','rng','next_document'):
            assert same(initial_reference[key],candidate[key]),(arm,key)
        assert initial_reference['protocol']['source_hashes']==candidate['protocol']['source_hashes']
    training={};ledgers=[]
    for arm in 'ABCD':
        rows=[json.loads(x) for x in (root/arm/'history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,33))
        ledgers.append([{k:r[k] for k in ('step','next_document','mode','gap','totals','answer_targets')} for r in rows])
        saved=torch.load(root/arm/'last.pt',weights_only=True,map_location='cpu')
        assert saved['step']==32 and saved['next_document']==856
        assert saved['protocol']['source_hashes']==initial_reference['protocol']['source_hashes']
        assert all(bool(torch.isfinite(v).all()) for v in saved['weights'].values() if v.is_floating_point())
        stats=[r['mechanism'] for r in rows if r['mechanism']]
        assert all(s['same_target_updated']==0 for s in stats)
        assert all(s['extra_head_norm']<=.250001*s['ordinary_head_norm']+1e-12 for s in stats)
        sums={k:sum(s[k] for s in stats) for k in ('pair_decisions','eligible_decisions','updated_decisions',
            'opposing_laces','helpful_unique','same_target_decisions','same_target_updated')}
        means={k:sum(s[k] for s in stats)/len(stats) for k in ('hard_overlap','soft_l1','shared_mass','unique_mass',
            'input_rms_difference','bias_rms','bias_saturation','ordinary_head_norm','extra_head_norm','seconds')}
        peak_ratio=max((s['extra_head_norm']/s['ordinary_head_norm'] if s['ordinary_head_norm'] else 0.) for s in stats)
        language_experts=saved['totals']['language_bytes']*7*2*3*2*2
        training[arm]=dict(steps=32,totals=saved['totals'],seconds=sum(r['seconds'] for r in rows),
            peak_mib=max(r['peak_mib'] for r in rows),parameters=sum(p.numel() for p in saved['weights'].values()),
            # state_dict contains aliases for shared banks: use named model count below.
            task_expert_executions=sum(r['expert_executions'] for r in rows),language_expert_executions=language_experts,
            task_memory_work={k:sum(r['memory_work'][k] for r in rows) for k in rows[0]['memory_work']},
            mechanism=dict(sums=sums,means=means,max_extra_to_ordinary=peak_ratio),
            checkpoint_sha256=hashlib.sha256((root/arm/'last.pt').read_bytes()).hexdigest(),
            initial_loss=rows[0]['memory_nll'],initial_language_loss=rows[0]['language_nll'])
        training[arm]['parameters']=read(root/arm/'protocol.json')['parameters']
    assert all(x==ledgers[0] for x in ledgers[1:])
    assert len({v['initial_loss'] for v in training.values()})==1
    assert len({v['initial_language_loss'] for v in training.values()})==1
    reference=measurements['parent']['validation'];gates={}
    for arm in 'ABCD':
        m=measurements[arm]['validation']
        gate=dict(paired=all(m[f'pairs_{g}']['both_accuracy']>=.5 for g in (0,32)),
            single=all(m[f'single_{g}']['accuracy']>=reference[f'single_{g}']['accuracy']-.02 for g in (0,32)),
            correction=all(m[f'correction_{g}']['accuracy']>=reference[f'correction_{g}']['accuracy']-.03 for g in (0,32)),
            language=m['language']['nll']<=reference['language']['nll']+.05)
        gate['pass']=all(gate.values());gates[arm]=gate
    final_mechanism=read(root/'endpoint_mechanism.json');assert set(final_mechanism)==set('BCD')
    bypass=read(root/'bypass_trajectory.json');assert bypass['exact']
    summary=dict(initial_states_identical=True,bypass_trajectory=bypass,protocol=read(root/'lock.json'),cohort_audit=read(root/'cohorts.json')['audit'],
        parent='artifacts/hippocampus/parent.pt',training=training,measurements=measurements,
        ablations=ablations,cue_swap_comparison=cue_swap,preflight=read(root/'preflight.json'),endpoint_mechanism=final_mechanism,gates=gates,
        retained_parent=not any(g['pass'] for g in gates.values()),
        validation_selection='Initial and final only; update-32 endpoints are primary',
        focused_tests=52,additional_final_rerun=10,
        training_source_revision=read(root/'publication.json')['public_training_commit'],
        local_training_source_revision=read(root/'publication.json')['local_training_commit'],
        runtime=read(root/'runtime.json'),
        evaluation_audit=dict(complete=True,sections=38,cohorts_sha256=cohort_hash,
            evaluator_sha256=evaluator_hash,files=evaluation_files))
    assessment=resultdir/'assessment.md'
    if assessment.exists():summary['assessment']=assessment.read_text().strip()
    resultdir.mkdir(parents=True,exist_ok=True);write_json(resultdir/'summary.json',summary)
    for arm in 'ABCD':
        for source,suffix in (('protocol.json','protocol.json'),('history.jsonl','history.jsonl')):
            (resultdir/f'{arm}_{suffix}').write_bytes((root/arm/source).read_bytes())
    write_json(root/'summary.json',summary)
    print(json.dumps(dict(gates=gates,training_seconds={k:v['seconds'] for k,v in training.items()})),flush=True)

if __name__=='__main__':main()
