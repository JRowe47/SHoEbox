"""Recoverable local driver for the locked experiment, evaluations and tests."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/gradient_horizon'


def run(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with logfile.open('a') as log:
        process = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError(f'{command} failed ({process.returncode}); see {logfile}')
    return {'command':command, 'log':str(logfile.relative_to(ROOT)), 'seconds':time.time()-started}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT/'run_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'completed':{}, 'state':'running'}
    def save():
        temp = status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(status_path)
    def task(name, command, log):
        if name in status['completed']: return status['completed'][name]
        return run(command, OUT/log)
    try:
        status['state']='training';save()
        jobs = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for arm in ('truncated','full'):
                command=['-m','experiments.text_regions.horizon_train','--parent',str(OUT/'parent.pt'),
                         '--out',str(OUT/arm)]
                if arm=='truncated':command.append('--detach-between')
                if (OUT/arm/'last.pt').exists():
                    summary=OUT/arm/'summary.json'
                    if summary.exists() and json.loads(summary.read_text())['completed_step']==32:continue
                    command.append('--resume')
                jobs.append((arm,pool.submit(task,'train/'+arm,command,arm+'/train.log')))
            for arm, job in jobs:status['completed']['train/'+arm]=job.result();save()
        status['state']='selection';save()
        status['completed']['selection']=task('selection',['scripts/audit_gradient_horizon.py'],'selection.log');save()
        selection=json.loads((OUT/'selection.json').read_text())['selected']
        status['state']='evaluation';save()
        evaluations=[(arm,OUT/arm/'last.pt',OUT/arm/'evaluation.json') for arm in ('truncated','full')]
        if selection['step'] != 32: evaluations.append(('selected',OUT/'selected.pt',OUT/'selected_evaluation.json'))
        jobs=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for name,checkpoint,path in evaluations:
                command=['scripts/evaluate_regional_facts.py',str(checkpoint),'--out',str(path),'--count','512','--batch','256']
                if path.exists():command.append('--resume')
                jobs.append(('evaluation/'+name,pool.submit(task,'evaluation/'+name,command,name+'_evaluation.log')))
            for arm in ('truncated','full'):
                command=['scripts/probe_regional_facts.py',str(OUT/arm/'last.pt'),
                         '--out',str(OUT/arm/'local_diagnostic.json')]
                jobs.append(('diagnostic/'+arm,pool.submit(task,'diagnostic/'+arm,command,arm+'/diagnostic.log')))
            for name,job in jobs:status['completed'][name]=job.result();save()
        status['state']='audit_and_regression';save()
        status['completed']['evaluation_audit']=task('evaluation_audit',
            ['scripts/audit_gradient_horizon.py','--evaluations'],'evaluation_audit.log');save()
        status['completed']['regression']=task('regression',['scripts/check_all.py'],'regression.log');save()
        status['state']='complete';save()
        print(json.dumps({'complete':True,'status':str(status_path)}),flush=True)
    except BaseException as exc:
        status['state']='failed';status['error']=repr(exc);save()
        raise


if __name__=='__main__':main()
