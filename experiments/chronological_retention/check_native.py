"""A3 exact chronological routing, numerical/lifecycle and ASan/UBSan checks."""
import argparse,os,subprocess,tempfile
from pathlib import Path
from .backend import ROOT,source,header
from experiments.joint_cold_gate.check_native import CPP as NUMERIC
from experiments.joint_cold_gate.check_request import CPP as REQUEST
from experiments.interference_retention.check_native import A1

SELECT=r'''
#include <iostream>
int main(){using namespace shoe;int n=0;auto ck=[&](bool ok){++n;if(!ok)throw std::runtime_error("A3 selection check "+std::to_string(n));};
 QueryAdapter a(500000,7,true,4,true);auto& s=a.model->levels[1].retained;
 auto clear=[&](int count){s={};s.count=count;s.clock=count;a.head={};
  for(auto& row:a.head.gate)row[D]=-1;
  for(int k=0;k<count;++k){s.units[k].stable[k]=.125f;s.units[k].born=k+1;}};
 auto select=[&](bool full=false){QueryCache c;c.query.fill(.5f);a3_select(a.head,s,c,0,full);return c;};
 auto same=[&](const QueryCache& x,const QueryCache& y){ck(x.slots==y.slots);ck(x.phases==y.phases);ck(x.keys==y.keys);ck(x.raw==y.raw);ck(x.scores==y.scores);ck(x.attention==y.attention);};
 // No recognized records: exact historical presence fallback/distance attention.
 clear(3);a.head.gate[0][0]=.9f;a.head.gate[1][1]=.8f;
 auto c=select();ck(c.slots[0]==0&&c.phases[0]==0);ck(c.slots[1]==1&&c.phases[1]==1);ck(c.attention==.5f);
 // One recognized candidate gets all mass, with no fallback candidate.
 a.head.gate[0][0]=2;c=select();ck(c.slots[0]==0&&c.slots[1]==-1);ck(c.attention==1);
 // Old and new exact matches; a later competing key must not displace either.
 clear(3);a.head.gate[0][D]=1;
 for(int r=0;r<QRANK;++r){a.head.key[r][0]=a.head.key[r][1]=std::atanh(.5f);a.head.key[r][2]=std::atanh(-.5f);}
 c=select();ck(c.slots[0]==1&&c.slots[1]==0);ck(c.attention>.999f);same(c,select(true));
 // Reversing occurrence order changes the selected version without changing keys.
 std::swap(s.units[0].born,s.units[1].born);c=select();ck(c.slots[0]==0&&c.slots[1]==1);same(c,select(true));
 s.units[0].born=s.units[1].born;c=select();ck(c.slots[0]==0); // equal born: slot order
 // Equal-born phases use stable phase order and pair their own value projections.
 clear(1);a.head.gate[0][D]=a.head.gate[1][D]=1;
 for(int r=0;r<2*QRANK;++r)a.head.key[r][D]=std::atanh(.5f);
 c=select();ck(c.slots[0]==0&&c.phases[0]==0);ck(c.phases[1]==1);same(c,select(true));
 for(int r=0;r<VRANK;++r){a.head.value[r][D]=.2f;a.head.value[VRANK+r][D]=-.3f;}
 c=query_forward(a.head,*a.model,0,true);
 for(int r=0;r<VRANK;++r){ck(c.values[0][r]==std::tanh(.2f));ck(c.values[1][r]==std::tanh(-.3f));ck(c.h[r]==c.attention*c.values[0][r]+(1-c.attention)*c.values[1][r]);}
 // Stable log expressions do not overflow, including fully saturated projections.
 for(double x:{-1e300,-1000.,-31.36,0.,31.36,1000.,1e300})ck(std::isfinite(a3_softplus(x)));
 clear(16);for(auto& row:a.head.gate)row[D]=1;
 for(auto& row:a.head.key)row[D]=1e30f;
 c=select();same(c,select(true));ck(std::isfinite(c.attention));
 // Many older exact matches: optimization must actually skip projections.
 QueryCache x,y;x.query.fill(1);y.query=x.query;int fast=0,full=0;
 a3_select(a.head,s,x,0,false,&fast);a3_select(a.head,s,y,0,true,&full);same(x,y);ck(fast<full);ck(full==64);
 // Broad random adversarial states, empty/single/multiple recognition, tied
 // occurrences, all lesion source modes; optimized selection exactly exhaustive.
 std::mt19937 rng(937);std::uniform_real_distribution<float> u(-1,1);
 for(int trial=0;trial<1000;++trial){clear(trial%17);
  for(auto& row:a.head.gate)for(auto& z:row)z=u(rng);
  for(auto& row:a.head.key)for(auto& z:row)z=u(rng);
  for(int k=0;k<s.count;++k){s.units[k].born=rng()%7;for(auto& z:s.units[k].stable)z=u(rng)*.125f;}
  QueryCache f,g;for(auto& z:f.query)z=u(rng);g.query=f.query;
  a3_select(a.head,s,f,trial%4);a3_select(a.head,s,g,trial%4,true);same(f,g);ck(std::isfinite(f.attention));
 }
 clear(0);c=select();ck(c.slots[0]==-1&&c.slots[1]==-1);ck(c.attention==.5f);
 auto blob=query_save(a);auto b=query_load(blob.data(),blob.size());ck(query_save(*b)==blob);
 a.head.key[0][0]=std::numeric_limits<float>::quiet_NaN();blob=query_save(a);
 bool rejected=false;try{auto bad=query_load(blob.data(),blob.size());}catch(...){rejected=true;}ck(rejected);
 std::cout<<n<<" A3 selection checks passed\n";
 std::cout<<"A3 scratch struct bytes "<<sizeof(A3Scratch)<<"; occurrence "<<sizeof(A3Occurrence)<<"; match "<<sizeof(A3Match)<<"; unchanged QueryCache "<<sizeof(QueryCache)<<"\n";
}
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection-only',action='store_true');args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='exp0305a3-native-') as folder:
        folder=Path(folder);cpp=folder/'test.cpp';exe=folder/'test'
        (folder/'readout.hpp').write_text(header('a3'))
        for body in ((SELECT,) if args.selection_only else (NUMERIC,REQUEST,A1,SELECT)):
            cpp.write_text(source('a3')+body)
            for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
                subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(folder),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
                subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()
