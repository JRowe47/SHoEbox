"""Recoverable execution queue: four matched arms, then sealed endpoint tests."""
import argparse,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_memory_selectivity import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=2);p.add_argument('--eval-workers',type=int,default=6)
    a=p.parse_args();root=ROOT/'artifacts/lace_separation';root.mkdir(exist_ok=True)
    status=dict(stage='preflight',completed=[],failed=[]);write_json(root/'pipeline.json',status)
    def read(path):return json.loads(path.read_text()) if path.exists() else {}
    while not ((root/'preflight.json').exists() and
        len(read(root/'initial_pairs.json').get('measurements',{}))==3 and
        len(read(root/'initial_retention.json').get('measurements',{}))==4 and
        len(read(root/'initial_language.json').get('measurements',{}))==1 and
        len(read(root/'reproduction_pairs.json').get('measurements',{}))==3):
        time.sleep(2)
    reproduced=read(root/'reproduction_pairs.json')['measurements']
    assert all(v['both_correct']==0 and v['conflicting_pairs']==96 for v in reproduced.values())
    if not (root/'lock.json').exists():
        old=read(root/'cohorts.json');oldhash=hashlib.sha256((root/'cohorts.json').read_bytes()).hexdigest()
        (root/'cohorts.json').rename(root/'cohorts_audit_draft.json')
        subprocess.run([sys.executable,str(ROOT/'scripts/lace_cohorts.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        new=read(root/'cohorts.json');assert old['cohorts']==new['cohorts']
        write_json(root/'cohort_audit_refresh.json',dict(original_lock_sha256=oldhash,
            reason='Correct historical seed-1031 start to 30000 and include earlier diagnostic banks',
            all_rendered_cohorts_unchanged=True,new_sha256=hashlib.sha256((root/'cohorts.json').read_bytes()).hexdigest()))
        write_json(root/'lock.json',dict(steps=32,arms=['A','B','C','D'],alpha=.1,beta=.05,
            cohorts_sha256=hashlib.sha256((root/'cohorts.json').read_bytes()).hexdigest(),
            test_sealed=True,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if not (root/'initial.json').exists():
        pairs=read(root/'initial_pairs.json');retention=read(root/'initial_retention.json');language=read(root/'initial_language.json')
        initial=dict(checkpoint_sha256=pairs['identity']['checkpoint_sha256'],
            selection_score=sum(pairs['measurements'][f'pairs_{g}']['nll'] for g in (0,32))/2,
            measurements=pairs['measurements']|retention['measurements']|language['measurements'])
        write_json(root/'initial.json',initial)
    def work(label,command):
        with (root/(label+'.log')).open('a') as f:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        return label,result.returncode
    def queue(jobs,workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(work,*j) for j in jobs]):
                label,code=future.result();status['failed' if code else 'completed'].append(label)
                write_json(root/'pipeline.json',status);print(json.dumps(dict(job=label,exit_code=code)),flush=True)
        if status['failed']:raise RuntimeError('Failed jobs: '+str(status['failed']))
    status['stage']='training';write_json(root/'pipeline.json',status)
    jobs=[]
    for arm in 'ABCD':
        out=root/arm
        command=[str(ROOT/'scripts/run_lace_separation.py'),'--arm',arm,'--out',str(out),'--initial',str(root/'initial.json')]
        if (out/'last.pt').exists():command.append('--resume')
        jobs.append((arm,command))
    queue(jobs,a.workers)
    import torch
    ledgers=[];endpoints={}
    for arm in 'ABCD':
        checkpoint=torch.load(root/arm/'last.pt',weights_only=True,map_location='cpu')
        assert checkpoint['step']==32 and checkpoint['next_document']==856
        rows=[json.loads(s) for s in (root/arm/'history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,33))
        ledgers.append([{k:r[k] for k in ('step','next_document','mode','gap','answer_targets','totals')} for r in rows])
        endpoints[arm]=dict(sha256=hashlib.sha256((root/arm/'last.pt').read_bytes()).hexdigest(),totals=checkpoint['totals'])
    assert all(x==ledgers[0] for x in ledgers[1:])
    write_json(root/'training_complete.json',dict(endpoints=endpoints,matched_ledgers=True))
    status['stage']='evaluation';write_json(root/'pipeline.json',status)
    jobs=[]
    checkpoints={arm:root/arm/'last.pt' for arm in 'ABCD'}
    checkpoints['parent']=ROOT/'artifacts/hippocampus/parent.pt'
    for name,checkpoint in checkpoints.items():
        for cohort in ('validation','test'):
            for part in ('pairs','retention','language')+ (('unseen',) if cohort=='test' else ()):
                label=f'{name}_{cohort}_{part}'
                jobs.append((label,[str(ROOT/'scripts/evaluate_lace_separation.py'),'--checkpoint',str(checkpoint),
                    '--cohort',cohort,'--part',part,'--out',str(root/'evaluation'/f'{label}.json')]))
    for control in ('bias_off','cue_permuted','blocked'):
        label='D_test_'+control
        jobs.append((label,[str(ROOT/'scripts/evaluate_lace_separation.py'),'--checkpoint',str(root/'D/last.pt'),
            '--cohort','test','--part','pairs','--control',control,'--out',str(root/'evaluation'/f'{label}.json')]))
    queue(jobs,a.eval_workers)
    status['stage']='complete';write_json(root/'pipeline.json',status)

if __name__=='__main__':main()
