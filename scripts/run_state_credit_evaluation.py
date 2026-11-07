"""Evaluate completed arms concurrently with separate caches, then merge."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/state_credit')
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    a=p.parse_args();root=a.root.resolve()
    evaluation=root/'evaluation.json'
    existing=json.loads(evaluation.read_text()) if evaluation.exists() else {}
    if 'parent' not in existing:
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_state_credit.py'),
            '--root',str(root),'--data',str(a.data.resolve()),'--checkpoints','parent'],check=True)
        existing=json.loads(evaluation.read_text())
    processes=[]
    for mode in ('state','publication'):
        folder=root/('eval_'+mode);folder.mkdir(exist_ok=True)
        for arm in ('state','publication'):
            link=folder/arm
            if link.is_symlink():
                if link.resolve()!=root/arm:raise ValueError('Evaluation link points to another study.')
            elif link.exists():
                raise ValueError('Evaluation staging path is not a symlink.')
            else:
                link.symlink_to(root/arm,target_is_directory=True)
        cache=folder/'evaluation_cache.json'
        if not cache.exists():shutil.copy2(root/'evaluation_cache.json',cache)
        handle=(root/('eval_'+mode+'.log')).open('a')
        command=[sys.executable,str(ROOT/'scripts/evaluate_state_credit.py'),'--root',str(folder),
            '--data',str(a.data.resolve()),'--checkpoints',mode+'_best',mode+'_last']
        processes.append((folder,subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT),handle))
    for folder,process,handle in processes:
        if process.wait():raise RuntimeError('Evaluation failed: '+folder.name)
        handle.close()
    cache=json.loads((root/'evaluation_cache.json').read_text())
    for folder,_,_ in processes:
        existing.update(json.loads((folder/'evaluation.json').read_text()))
        cache.update(json.loads((folder/'evaluation_cache.json').read_text()))
        for path in folder.glob('*_traces.jsonl'):
            shutil.copy2(path,root/path.name)
    for name,data in [('evaluation.json',existing),('evaluation_cache.json',cache)]:
        path=root/name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
    print('Parent, selected and final checkpoints evaluated.',flush=True)


if __name__=='__main__':
    main()
