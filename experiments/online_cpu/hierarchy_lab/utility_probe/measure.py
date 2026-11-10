import argparse,ctypes as ct,hashlib,json,os,time
import numpy as np
from model import ROOT,Model,core,evaluate
from experiment import DATA,SIZE,atomic
p=argparse.ArgumentParser();p.add_argument('profile');p.add_argument('--seed',type=int,default=1);p.add_argument('--benchmark',action='store_true');a=p.parse_args()
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','8'))})
def source(name):return ROOT/'results'/SIZE/f'seed{a.seed}'/name/'last.npz'
def benchmark():
    names=a.profile.split(',');samples={n:[] for n in names};data=np.frombuffer((DATA/'fresh.bin').read_bytes(),np.uint8)[2000000:]
    for rep in range(5):
        for n in names[::1 if rep%2==0 else -1]:
            m=Model.load(source(n));m.run(data[:2048],data[1:2049],mode='periodic',lr=.008,lp=.008/3,route=.002)
            t=time.perf_counter();m.run(data[2048:18432],data[2049:18433],mode='periodic',lr=.008,lp=.008/3,route=.002);samples[n].append(16384/(time.perf_counter()-t));m.close()
    out={k:dict(samples=v,median=float(np.median(v))) for k,v in samples.items()}
    atomic(ROOT/'results/evaluation'/f'benchmark_{SIZE}_s{a.seed}.json',out);print(json.dumps(out),flush=True)
def measure():
    assert (ROOT/'selection.json').exists(),'Held-out test sealed until validation choice is saved'
    m=Model.load(source(a.profile));before=m.fingerprint()
    data=np.frombuffer((DATA/'test.bin').read_bytes(),np.uint8)
    manifest=json.loads((DATA/'manifest.json').read_text());docs=manifest['heldout']['test']['documents']
    result=dict(size=SIZE,seed=a.seed,profile=a.profile,parameters=m.parameter_count,checkpoint_sha256=hashlib.sha256(source(a.profile).read_bytes()).hexdigest(),test=evaluate(m,data,documents=docs))
    # A common calibration segment from training; no test-fitted mean.
    train=np.frombuffer((DATA/'fresh.bin').read_bytes(),np.uint8)[1000000-65536:1000001]
    f=m.frozen_copy();f.lib.diagnostic_enable(f.handle,1);f.run(train[:-1],train[1:])
    stat=np.zeros(18+18*4*64+18*4*64*64+18*8,np.float64);f.lib.diagnostic_export(f.handle,stat.ctypes.data_as(ct.POINTER(ct.c_double)))
    mean=(stat[18:18+18*4*64].reshape(18,4,64)[17,3]/max(1,stat[17])).astype(np.float32)
    result['diagnostics']=f.diagnostics();f.close()
    f=m.frozen_copy()
    if m.utility['reserved1']<=0:f.params['rbias']+=f.params['readout'][:,64:]@(mean-m.utility['center_y']*m.ymean())
    f.params['readout'][:,64:]=0
    result['mean_Y_test']=evaluate(f,data,documents=docs);f.close()
    result['dynamic_Y_delta']=result['mean_Y_test']['nll']-result['test']['nll']
    f=m.frozen_copy();f.lib.utility_observe(f.handle,1)
    f.run(train[:-1],train[1:],mode='periodic',lr=.008,lp=.008/3,route=.002)
    result['teaching_diagnostics']=f.utility_metrics();result['lace_diagnostics']=f.lace_metrics();f.close()
    assert before==m.fingerprint();assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all()
    result['trainer_preserved']=True;result['finite']=True;m.close()
    atomic(ROOT/'results/evaluation'/f'{SIZE}_s{a.seed}_{a.profile}.json',result)
    print(json.dumps({k:result[k] for k in ['size','seed','profile','dynamic_Y_delta']}),flush=True)
benchmark() if a.benchmark else measure()
