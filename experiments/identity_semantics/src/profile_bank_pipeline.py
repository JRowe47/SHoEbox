"""Sample exclusive serial-stage costs in an isolated runtime copy; prove parity."""
import argparse,ctypes as ct,hashlib,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def instrument(runtime):
    native=runtime/'hierarchy_lab/throughput_probe'
    p=native/'composition.cpp';s=p.read_text().replace('#include <immintrin.h>','#include <immintrin.h>\n#include "profiler.inc"')
    s=s.replace('  for(int t=0;t<length;t++){','  for(int t=0;t<length;t++){\n    profile_sample(s.tick+1);ProfileScope phase(P_BYTE);')
    s=s.replace('    float features[2*D],logits','    phase.stage(P_HEAD);\n    float features[2*D],logits')
    s=s.replace('    s.correct+=guess==y[t];','    phase.stage(P_TASK);\n    s.correct+=guess==y[t];')
    s=s.replace('      float norms[2];','      phase.stage(P_READOUT);\n      float norms[2];')
    s=s.replace('    if(m.event.training)event_finish_step(m);',
                '    phase.stage(P_CREDIT);\n    if(m.event.training)event_finish_step(m);')
    p.write_text(s)
    functions={'constituent_union_flush':'P_CREDIT','content_expert_derivatives':'P_MATERIALIZE',
               'content_expert_derivative':'P_MATERIALIZE','neural_collect':'P_MEMORY','neural_archive_slot':'P_PUBLICATION',
               'pipeline_capture':'P_STATE_LEARN','pipeline_finish_step':'P_FLUSH',
               'content_expert_encode':'P_FORWARD','context_forward':'P_FORWARD','content_expert_flush':'P_EXPERT_UPDATE'}
    found={name:0 for name in functions}
    for p in [native/'composition.cpp',*native.glob('*.inc')]:
        s=p.read_text()
        for name,category in functions.items():
            pattern=r'(inline [^\n;]*\b'+name+r'\([^;{}]*\)\{)'
            s,n=re.subn(pattern,lambda m:m[0]+f'\n  ProfileScope measured_scope({category});',s);found[name]+=n
        p.write_text(s)
    for name,count in found.items():
        if count!=1:raise AssertionError((name,count))

def worker(a):
    import numpy as np
    sys.path.insert(0,str(a.runtime));from runtime import fresh_model
    os.sched_setaffinity(0,{a.cpu});m=fresh_model(seed=7,profile='pipeline')
    raw=a.corpus.read_bytes()[:5121];x=np.frombuffer(raw,np.uint8)
    m.run(x[:1024],x[1:1025],mode='periodic',lr=.04,lp=.04/3)
    if hasattr(m.lib,'profile_reset'):m.lib.profile_reset()
    loss,logits=m.run(x[1024:-1],x[1025:],mode='periodic',lr=.04,lp=.04/3,logits=True)
    result=dict(nll=loss,fingerprint=m.fingerprint(),logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest())
    if hasattr(m.lib,'profile_export'):
        values=np.zeros(46,np.uint64);m.lib.profile_export.argtypes=[ct.c_void_p];m.lib.profile_export(values.ctypes.data)
        names=['other_forward','input','record_collection','pipeline_boundary','local_teacher','state_input','routing',
               'content_context_forward','archive_slot','prototype','predictor','decoder_head','task_other','readout_update',
               'bank_flush','reverse_and_event','factor_capture','route_update','bottom_route','feature_mask','publish_state','content_vjp']
        total=sum(map(int,values[2:24]));result.update(sampled_bytes=int(values[1]),total_bytes=int(values[0]),
            categories={name:dict(ns=int(values[2+i]),calls=int(values[24+i]),percent=100*int(values[2+i])/total) for i,name in enumerate(names)})
    m.close();a.output.write_text(json.dumps(result,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path);p.add_argument('--worker',action='store_true')
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));a=p.parse_args()
    if a.worker:return worker(a)
    rows={}
    with tempfile.TemporaryDirectory(prefix='shoebox-stage6-profile-') as folder:
        runtime=Path(folder)/'runtime';shutil.copytree(ROOT/'runtime',runtime,ignore=shutil.ignore_patterns('*.so','*.pyc','__pycache__','.build*'))
        instrument(runtime)
        for name,path in [('control',ROOT/'runtime'),('profile',runtime)]:
            out=Path(folder)/(name+'.json')
            subprocess.run([sys.executable,__file__,'--worker','--runtime',str(path),'--corpus',str(a.corpus),'--output',str(out),'--cpu',str(a.cpu)],check=True)
            rows[name]=json.loads(out.read_text())
        for key in rows['control']:assert rows['control'][key]==rows['profile'][key],key
    a.output.write_text(json.dumps(dict(bitwise_parity=True,**rows),indent=2)+'\n')

if __name__=='__main__':main()
