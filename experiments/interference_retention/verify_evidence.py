"""Verify locked artifacts and apply preregistered gates without fitting."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from .run import ROOT, sources, write
from .tasks import manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace
    read=lambda path:json.loads((w/path).read_text())
    plan=read('a1/plan.json');lock=read('a1/selection.json')
    assert plan['source_hashes']==sources()
    assert plan['folds']==manifest()
    for name,expected in lock['checkpoints'].items():
        assert hashlib.sha256((w/'a1'/name).read_bytes()).hexdigest()==expected,name
    for folder in ('diagnosis_initial','diagnosis_corrected'):
        old=read(('diagnosis' if folder=='diagnosis_initial' else 'diagnosis_corrected')+'/lock.json')
        for name,expected in old['source_hashes'].items():
            assert hashlib.sha256((w/'source_snapshots'/folder/name).read_bytes()).hexdigest()==expected,name
    assert not plan['audit_opened'] and not lock['audit_opened']
    protected=['multiscale','shoe_multiscale.py','shoebox_reference.py']
    protected.extend(str(p.relative_to(ROOT)) for p in (ROOT/'experiments').iterdir() if p.is_dir() and p.name!='interference_retention')
    assert not subprocess.check_output(['git','diff','9e34d8075abbc495f966ab37d18ecfa532bb854d','--',*protected],cwd=ROOT)
    cursors=read('cursors.json');history=read('history.json');speed=read('benchmark.json')
    assert cursors['all_exact'] and cursors['count']==18 and history['exact'] and len(history['rows'])==14 and speed['public_e8_exact']
    results=read('a1/results.json');ext=read('extended/results.json');updates=read('update_metrics.json')
    assert not ext['audit_opened'] and not updates['overwrite_qualification_valid']
    assert updates['shortcuts']['overwrite16']['latest_value']==1
    gates={}
    def threshold(vals,mean,minimum):
        return dict(mean=float(np.mean(vals)),minimum=float(min(vals)),mean_threshold=mean,minimum_threshold=minimum,pass_gate=bool(np.mean(vals)>=mean and min(vals)>=minimum))
    for arm in ('e8','a1'):
        rows=[r['arms'][arm] for r in results['rows']];erows=[r['arms'][arm] for r in ext['rows']];g={}
        for condition in ('pad0','repeat16','diverse16','compete16','overwrite16'):
            g[condition]=threshold([r['development'][condition]['exact'] for r in rows],.95,.9)
            g[condition]['positions']=[threshold([r['development'][condition]['positions'][i] for r in rows],.95,.9) for i in (0,1)]
            g[condition]['both_orders']=threshold([r['both_orders'][condition]['both_query_orders_exact'] for r in erows],.9,.85)
            offsets=np.array([r['offsets'][condition] for r in erows]);mean=float(offsets.mean());worst=float(offsets.mean(axis=0).min())
            g[condition]['offsets']=dict(mean=mean,worst=worst,mean_threshold=.95,worst_threshold=.9,pass_gate=mean>=.95 and worst>=.9)
        g['delays']={c:threshold([r['development'][c]['exact'] for r in rows],.9,.85) for c in ('repeat64','repeat256','diverse64','diverse256')}
        g['padding']={str(d):threshold([r['padding'][str(d)]['exact'] for r in erows],.9,.85) for d in (0,64,128,256)}
        g['no_query_drop']=threshold([r['lesions']['None']-r['lesions']['no_query'] for r in erows],.2,.2)
        g['reverse_hurts']=all(r['lesions']['reverse']<r['lesions']['None'] for r in erows)
        g['throughput']=dict(relative_native=speed['rows'][arm]['relative_native'],threshold=.8,pass_gate=speed['rows'][arm]['gate_pass'])
        g['development_pass']=False
        assert not g['compete16']['pass_gate'] and not g['diverse16']['pass_gate'] and not g['delays']['diverse256']['pass_gate']
        gates[arm]=g
    costs={p.stem:json.loads(p.read_text()) for p in w.glob('*_process.json')}
    for name,cost in costs.items():assert cost['exit_code']==(1 if name=='diagnosis_process' else 0),name
    write(a.out,dict(locked_artifacts_verified=len(lock['checkpoints']),fit_sources_unchanged=True,diagnostic_source_snapshots_verified=True,
        protected_sources_unchanged=True,folds_verified=True,audit_opened=False,cursors_exact=18,historical_trajectories_exact=14,
        gates=gates,overwrite_qualification_valid=False,public_default='qualified E8 unchanged',a2_status='registered, not fitted',
        verification_process_exit_codes={k:v['exit_code'] for k,v in costs.items()}))
    print('All source/data/checkpoint locks verified; A1 fails development; E8 retained; audit closed.')

if __name__=='__main__':main()
