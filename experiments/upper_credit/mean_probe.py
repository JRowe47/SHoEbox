"""Frozen test: replace upper signals with means calibrated on training inputs."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.plateau_100m import common
from experiments.upper_credit.local_head import replace
from experiments.scale_100m.train import atomic,file_hash
from shoe_multiscale import Model

GLOBALS='''
int mean_levels=0,mean_kind=0;bool mean_record=false;
std::array<std::array<double,144>,4> mean_sums{};
std::array<std::array<float,144>,4> mean_values{};
std::array<std::array<uint64_t,2>,4> mean_counts{};
'''
API='''
extern "C" int mean_start(){return guard([&]{shoe::mean_sums={};shoe::mean_counts={};shoe::mean_record=true;shoe::mean_levels=0;});}
extern "C" int mean_observe(void* p,const uint8_t* x,size_t n){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);if(m.training)throw std::logic_error("frozen only");for(size_t i=0;i<n;++i)m.observe(x[i]);});}
extern "C" int mean_finish(double* counts){return guard([&]{using namespace shoe;mean_record=false;for(int l=0;l<4;++l){for(int j=0;j<144;++j)mean_values[l][j]=mean_sums[l][j]/std::max<uint64_t>(1,mean_counts[l][j<128?0:1]);for(int k=0;k<2;++k)counts[l*2+k]=mean_counts[l][k];}});}
extern "C" int mean_set(int levels,int kind){return guard([&]{shoe::mean_levels=levels;shoe::mean_kind=kind;});}
'''
def build(directory):
    common.build(directory);p=Path(directory)/'native.cpp';s=p.read_text()
    s=s.replace('namespace shoe {','namespace shoe {\n'+GLOBALS,1)
    needle='cached_gain[i]+=s.gain[j];cached_bias[i]+=s.bias[j];'
    s=replace(s,needle,'if(!training&&(mean_levels&(1<<l))&&(mean_kind&1)&&s.executions){cached_gain[i]+=mean_values[l][j];cached_bias[i]+=mean_values[l][64+j];}else{'+needle+'}')
    needle='if(int(l)==shuffle)std::rotate(level.read_now.value.begin(),level.read_now.value.begin()+11,level.read_now.value.end());'
    s=replace(s,needle,needle+'''
        if(!training&&mean_record){auto& sums=mean_sums[l];
            if(level.s.executions){++mean_counts[l][0];for(int i=0;i<D;++i){sums[i]+=level.s.gain[i];sums[64+i]+=level.s.bias[i];}}
            if(level.retained.count){++mean_counts[l][1];for(int i=0;i<RECALL_K;++i)sums[128+i]+=level.read_now.value[i];}}
        if(!training&&(mean_levels&(1<<l))&&(mean_kind&2)&&level.retained.count)
            for(int i=0;i<RECALL_K;++i)level.read_now.value[i]=mean_values[l][128+i];
''')
    p.write_text(s+API)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(p),'-o',str(Path(directory)/'diagnostic.so')],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--seed',type=int,default=7);p.add_argument('--cpu',type=int,default=4);p.add_argument('--build',action='store_true');a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    if a.build:build(a.work/'runtime');return
    os.sched_setaffinity(0,{a.cpu});_,lib=common.install(a.work/'runtime')
    lib.mean_start.argtypes=[];lib.mean_start.restype=C.c_int
    lib.mean_observe.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];lib.mean_observe.restype=C.c_int
    lib.mean_finish.argtypes=[C.c_void_p];lib.mean_finish.restype=C.c_int
    lib.mean_set.argtypes=[C.c_int,C.c_int];lib.mean_set.restype=C.c_int
    meta=json.loads((a.data/'manifest.json').read_text());train=np.fromfile(a.data/'train.bin',np.uint8);dev=np.fromfile(a.data/'development.bin',np.uint8)
    train_docs=common.take_documents(meta['files']['train']['documents'],262144);docs=meta['files']['development']['documents']
    counts=np.zeros((4,2),np.float64)
    with Model.load(a.checkpoint) as m:
        with m.clone() as retired:
            retired.finish()
            with retired.inference_copy(cold=True,clear_memory=True) as frozen:
                frozen._check(lib.mean_start())
                for d in train_docs:
                    frozen.reset_context(clear_memory=True);x=train[d['start']:d['stop']-1]
                    frozen._check(lib.mean_observe(frozen._alive(),x.ctypes.data,len(x)))
                frozen._check(lib.mean_finish(counts.ctypes.data))
        full=common.score(m,dev,docs);r={'seed':a.seed,'checkpoint_sha256':file_hash(a.checkpoint),'calibration_targets':262144,'calibration':'training inputs only; no targets or gradient updates','counts':counts.tolist(),'full_nll':full['nll'],'means':{}}
        for levels,label in [(4,'L2'),(8,'L3'),(12,'L2+L3')]:
            for kind,name in [(1,'modulation'),(2,'learned_recall'),(3,'both')]:
                m._check(lib.mean_set(levels,kind));score=common.score(m,dev,docs)
                r['means'][label+'_'+name]={'nll':score['nll'],'delta':score['nll']-full['nll']}
                print(label,name,r['means'][label+'_'+name],flush=True)
        m._check(lib.mean_set(0,0));atomic(a.work/f'means_s{a.seed}.json',r)
if __name__=='__main__':main()
