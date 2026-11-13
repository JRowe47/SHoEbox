"""Matched fresh-start byte-language shakedown, one CPU core per process."""
import argparse,hashlib,json,math,os,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import fresh_model,Model

def write(path,value):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
def evaluate(model,data):
    m=model.frozen_copy();before=m.composition_state(fitted=True)
    loss=0.;n=len(data)-1;start=time.perf_counter()
    for i in range(0,n,8192):
        count=min(8192,n-i);loss+=count*m.run(data[i:i+count],data[i+1:i+count+1])
    if before.size:np.testing.assert_array_equal(before,m.composition_state(fitted=True))
    metrics=m.composition_metrics();elapsed=time.perf_counter()-start;m.close();return loss/n,elapsed,metrics
def main():
    p=argparse.ArgumentParser();p.add_argument('--kernel',choices=['native','composition'],required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--core',type=int,default=0);p.add_argument('--tokens',type=int,default=1000000);p.add_argument('--tag',default='final');a=p.parse_args()
    os.sched_setaffinity(0,{a.core})
    train=np.fromfile(ROOT/'data/train_segment.txt',np.uint8);held=np.fromfile(ROOT/'data/heldout.bin',np.uint8)
    stream=train[np.arange(a.tokens+1)%len(train)]
    folder=ROOT/'results'/f'{a.tag}_{a.kernel}_seed{a.seed}';folder.mkdir(exist_ok=True)
    model=fresh_model(a.seed,kernel=a.kernel)
    result=dict(kernel=a.kernel,seed=a.seed,tokens=0,parameters=model.parameter_count,extra_runtime_bytes=int(model.composition_state().size),train_seconds=0.,validation=[],training=[],status='running',python=platform.python_version(),core=a.core,train_sha256=hashlib.sha256(train.tobytes()).hexdigest(),heldout_sha256=hashlib.sha256(held.tobytes()).hexdigest(),policy=model.composition)
    result['binary_sha256']=hashlib.sha256(Path(model.lib._name).read_bytes()).hexdigest()
    loss,seconds,_=evaluate(model,held);result['validation'].append(dict(tokens=0,nll=loss,seconds=seconds))
    for pos in range(0,a.tokens,5000):
        count=min(5000,a.tokens-pos);rate=.008+.036*(1+math.cos(math.pi*min(1,(pos+count/2)/1000000)))
        start=time.perf_counter();loss=model.run(stream[pos:pos+count],stream[pos+1:pos+count+1],mode='periodic',lr=rate,lp=rate/3,route=.002);seconds=time.perf_counter()-start
        result['tokens']=pos+count;result['train_seconds']+=seconds
        result['training'].append(dict(tokens=pos+count,nll=float(loss),seconds=seconds,lr=rate))
        if (pos+count)%100000==0 or pos+count==a.tokens:
            val,elapsed,metrics=evaluate(model,held);row=dict(tokens=pos+count,nll=val,seconds=elapsed);result['validation'].append(row)
            result['train_tok_s']=(pos+count)/result['train_seconds'];result['metrics']=model.composition_metrics();result['eval_metrics']=metrics
            write(folder/'training.json',result);print(a.kernel,a.seed,pos+count,round(val,6),round(result['train_tok_s']),flush=True)
    model.save(folder/'last.npz');result['status']='complete';write(folder/'training.json',result);model.close()
if __name__=='__main__':main()
