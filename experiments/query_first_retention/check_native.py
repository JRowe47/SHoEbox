"""A2 adversarial routing, numerical/lifecycle and ASan/UBSan checks."""
import argparse,os,subprocess,tempfile
from pathlib import Path
from .backend import ROOT,source,header
from experiments.joint_cold_gate.check_native import CPP as NUMERIC
from experiments.joint_cold_gate.check_request import CPP as REQUEST
from experiments.interference_retention.check_native import A1

SELECT=r'''
#include <iostream>
int main(){using namespace shoe;int n=0;auto ck=[&](bool ok){++n;if(!ok)throw std::runtime_error("A2 selection check "+std::to_string(n));};
 QueryAdapter a(500000,7,true,4,true);auto& s=a.model->levels[1].retained;s.count=4;s.clock=4;
 for(int k=0;k<4;++k){s.units[k].stable[k]=.125f;s.units[k].born=k+1;}
 for(auto& row:a.head.gate)row[D]=-2;
 a.head.gate[0][D]=1;a.head.gate[0][0]=10;a.head.gate[0][1]=8;
 for(int r=0;r<QRANK;++r){auto& row=a.head.key[r];row={};row[0]=std::atanh(-.5f);row[1]=std::atanh(-.2f);row[2]=std::atanh(.5f);row[3]=std::atanh(.25f);}
 auto get=[&](){QueryCache c;c.query.fill(.5f);a2_select(a.head,s,c,0);return c;};
 auto c=get();ck(c.slots[0]==2&&c.slots[1]==3);ck(c.phases[0]==0&&c.phases[1]==0);
 for(int r=0;r<QRANK;++r)a.head.key[r][3]=a.head.key[r][2];
 c=get();ck(c.slots[0]==3&&c.slots[1]==2);s.units[3].born=s.units[2].born;
 c=get();ck(c.slots[0]==2&&c.slots[1]==3);
 // A lone recognized source is followed by historical presence fallback,
 // skipping itself. There is no threshold other than the registered >0.
 a.head.gate[0]={};a.head.gate[0][D]=-1;a.head.gate[0][2]=3;a.head.gate[0][0]=.9f;
 c=get();ck(c.slots[0]==2&&c.slots[1]==0);
 a.head.gate[0][2]=0;c=get();ck(c.slots[0]==0&&c.slots[1]==1);
 // Stable slot then phase order breaks exact distance AND occurrence ties.
 a.head.gate[0]={};a.head.gate[1]={};a.head.gate[0][D]=a.head.gate[1][D]=1;
 for(auto& row:a.head.key)row={};for(auto& u:s.units)u.born=1;
 c=get();ck(c.slots[0]==0&&c.phases[0]==0);ck(c.slots[1]==0&&c.phases[1]==1);
 // Paired value decoding and downstream sigmoid mixing remain exact.
 for(auto& row:a.head.value)row={};
 for(int r=0;r<VRANK;++r){a.head.value[r][D]=.2f;a.head.value[VRANK+r][D]=-.3f;}
 c=query_forward(a.head,*a.model,0,true);ck(c.attention==.5f);
 for(int r=0;r<VRANK;++r){ck(c.values[0][r]==std::tanh(.2f));ck(c.values[1][r]==std::tanh(-.3f));ck(c.h[r]==.5f*c.values[0][r]+.5f*c.values[1][r]);}
 s.count=0;c=get();ck(c.slots[0]==-1&&c.slots[1]==-1);
 auto blob=query_save(a);auto b=query_load(blob.data(),blob.size());ck(query_save(*b)==blob);
 // Nonfinite checkpoints reject without silently repairing their meaning.
 a.head.key[0][0]=std::numeric_limits<float>::quiet_NaN();blob=query_save(a);
 bool rejected=false;try{auto bad=query_load(blob.data(),blob.size());}catch(...){rejected=true;}ck(rejected);
 std::cout<<n<<" A2 selection checks passed\n";
}
'''
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection-only',action='store_true');args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='exp0305a2-native-') as folder:
        folder=Path(folder);cpp=folder/'test.cpp';exe=folder/'test'
        (folder/'readout.hpp').write_text(header('a2'))
        for body in ((SELECT,) if args.selection_only else (NUMERIC,REQUEST,A1,SELECT)):
            cpp.write_text(source('a2')+body)
            for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
                subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(folder),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
                subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()
