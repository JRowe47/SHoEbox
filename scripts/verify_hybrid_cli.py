"""Compare actual CLI pause/resume against uninterrupted Phase A updates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_hierarchy_data import prepare


def equal(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/hybrid_cli_verification')
    root=parser.parse_args().out.resolve();root.mkdir(parents=True,exist_ok=False)
    rows=[dict(row_idx=i,row=dict(id=str(i),text=('doc '+str(i)+' one two three. ')*80)) for i in range(12)]
    prepare(rows,root/'data',train_targets=384,batch=2,validation_targets=128,document_cap=64)
    base=[sys.executable,'-m','shoebox.train_hierarchy','--data',str(root/'data'),'--updates','3',
          '--context','64','--eval-context','32','--objective','latent','--validate-every','3','--log-every','1']
    for name,extra in [('whole',[]),('split',['--max-updates','1']),('split',['--resume',str(root/'split/last.pt')])]:
        with (root/(name+('_resume' if '--resume' in extra else '')+'.log')).open('w') as f:
            subprocess.run(base+['--out',str(root/name)]+extra,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    a,b=[torch.load(root/n/'last.pt',weights_only=True) for n in ('whole','split')]
    result={k:equal(a[k],b[k]) for k in ('model','optimizer','optimizer_names','state','rng','objective_config')}
    for key,value in result.items():
        if not value:
            raise AssertionError(key)
    # Source, document metadata and objective changes must reject exact resume.
    reject={}
    for name,args in [('objective',['--objective','legacy']),('source',[]),('data',[])]:
        original=None
        if name=='source':
            # Alter the saved protocol, not the running repository.
            checkpoint=torch.load(root/'split/last.pt',weights_only=True)
            original=(root/'split/last.pt').read_bytes()
            checkpoint['metadata']['protocol']['source_sha256']='changed'
            torch.save(checkpoint,root/'split/last.pt')
        if name=='data':
            path=root/'data/manifest.json';original=path.read_bytes()
            m=json.loads(original);m['preprocessing']+=' changed';path.write_text(json.dumps(m))
        p=subprocess.run(base+['--out',str(root/'split'),'--resume',str(root/'split/last.pt')]+args,
                         cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        reject[name]=p.returncode!=0 and 'same model, corpus, source and training protocol' in p.stdout
        if original is not None:
            (root/'split/last.pt' if name=='source' else root/'data/manifest.json').write_bytes(original)
        if not reject[name]:
            raise AssertionError((name,p.stdout))
    (root/'verification.json').write_text(json.dumps(dict(exact=result,changed_protocol_rejected=reject),indent=2)+'\n')
    print(json.dumps(dict(exact=result,changed_protocol_rejected=reject)))


if __name__=='__main__':
    main()
