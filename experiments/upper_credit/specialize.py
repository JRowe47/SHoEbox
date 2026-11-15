"""Diagnostic extra recall-language credit to expert matrices/biases only."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.upper_credit import screen
from experiments.plateau_100m import common
GAINS={'e4_16':(4,16),'e16_64':(16,64),'C':(1,1),'r1':(1,1)}
for name in GAINS:
    screen.ARMS[name]=(1,1)
    if name!='C':screen.RECALL[name]=(1,1)
def build(directory):
    common.build(directory);p=Path(directory)/'native.cpp';s=p.read_text()
    s=s.replace('int diagnostic_balance_update=1;', 'int diagnostic_balance_update=1;\nfloat expert_recall_gain[4]{1,1,1,1};')
    old='l.recall_content_gradient.clear();l.recall_credit_state=RecallCreditState{};'
    assert s.count(old)==1;s=s.replace(old,'l.recall_credit_state=RecallCreditState{};')
    old='l.constituents.clear();l.routing.constituents.clear();'
    assert s.count(old)==1;s=s.replace(old,'l.recall_content_gradient.clear();'+old)
    old='Vec gated=pre;for(auto& x:gated)x*=c.gate;outer(eg[k].w,gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);'
    assert s.count(old)==1
    new='''Vec gated=pre;for(auto& x:gated)x*=c.gate;
                Vec expert_gated=gated;
                if(recall_credit[scale]>0&&expert_recall_gain[scale]!=1){
                    float weight=(expert_recall_gain[scale]-1)*recall_credit[scale]/std::max(1,epoch_bytes);
                    for(int i=0;i<D;++i){float extra=weight*l.recall_content_gradient[ci-1][i]*(1-c.content[i]*c.content[i]);
                        eg[k].bias[i]+=extra;expert_gated[i]+=c.gate*extra;}}
                outer(eg[k].w,expert_gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);'''
    s=s.replace(old,new)
    s+='\nextern "C" int upper_expert_gain(float a,float b){shoe::expert_recall_gain[2]=a;shoe::expert_recall_gain[3]=b;return 0;}\n';p.write_text(s)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(p),'-o',str(Path(directory)/'diagnostic.so')],check=True)
def install(directory,arm):
    base,lib=common.install(directory);lib.upper_expert_gain.argtypes=[C.c_float,C.c_float];lib.upper_expert_gain.restype=C.c_int
    lib.upper_expert_gain(*GAINS[arm]);return base,lib
if __name__=='__main__':
    screen.build=build;screen.install=install;screen.main()
