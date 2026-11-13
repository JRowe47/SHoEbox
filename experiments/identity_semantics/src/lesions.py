import json,os,sys,time,argparse
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
parser=argparse.ArgumentParser(description='Evaluate causal regional contributions on the complete held-out split.')
parser.add_argument('--core',type=int,default=min(os.sched_getaffinity(0)))
args=parser.parse_args()
os.sched_setaffinity(0,{args.core})
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model
data=np.fromfile(ROOT/'data/heldout.bin',np.uint8);rows=[]
for kernel,folder in [('native','native_seed1'),('composition','delivery_composition_seed1')]:
    model=Model.load(ROOT/'results'/folder/'last.npz')
    for condition,region in [('intact',None),('mean_Y',None),('remove_L3',6),('remove_E4',10),('remove_G5',14)]:
        m=model.frozen_copy()
        if condition=='mean_Y':m.lib.utility_ablate(m.handle,1)
        if region is not None:m.ablate(10+region)
        loss=0.;n=len(data)-1
        for i in range(0,n,8192):
            count=min(8192,n-i);loss+=count*m.run(data[i:i+count],data[i+1:i+count+1])
        row=dict(kernel=kernel,condition=condition,nll=loss/n,targets=n);rows.append(row);print(row,flush=True);m.close()
    model.close()
(ROOT/'results/lesions.json').write_text(json.dumps(rows,indent=2)+'\n')
