"""A4 exact chronological routing, numerical/lifecycle and ASan/UBSan checks."""
import argparse,os,subprocess,tempfile
from pathlib import Path
from .backend import ROOT,source,header
from experiments.joint_cold_gate.check_native import CPP as NUMERIC
from experiments.joint_cold_gate.check_request import CPP as REQUEST
from experiments.interference_retention.check_native import A1

SELECT=r'''
#include <iostream>

// Independent exhaustive reference: gathers all phases, projects in a separate
// pass, computes every weight, globally sorts; no optimized helper is called.
shoe::QueryCache independent(const shoe::QueryHead& p,const shoe::RecallStream& s,
                            shoe::LowRank query,int lesion){
 using namespace shoe;
 struct Ref {int slot,phase;uint64_t born;float score,distance=0;Vec raw{};LowRank key{};double logw=0;};
 std::vector<Ref> all,yes;QueryCache c;c.query=query;
 for(int k=0;k<s.count;++k)for(int ph=0;ph<4;++ph){
  Ref v{k,ph,s.units[k].born,0};v.raw=source_vector(s.units[k].stable);if(lesion==3)v.raw={};
  v.score=affine(p.gate[ph],v.raw);all.push_back(v);if(v.score>0)yes.push_back(v);
 }
 auto project=[&](Ref& v){for(int r=0;r<QRANK;++r){v.key[r]=std::tanh(affine(p.key[v.phase*QRANK+r],v.raw));float e=v.key[r]-query[r];v.distance+=e*e;}};
 for(auto& v:yes)project(v);
 auto stable=[](const Ref& x,const Ref& y){return x.slot<y.slot||(x.slot==y.slot&&x.phase<y.phase);};
 if(!yes.empty()){
  float minimum=yes[0].distance;for(const auto& v:yes)minimum=std::min(minimum,v.distance);
  std::sort(yes.begin(),yes.end(),[&](const Ref& x,const Ref& y){return x.born!=y.born?x.born>y.born:stable(x,y);});
  double remaining=0;
  auto softplus=[](double x){return std::max(x,0.)+std::log1p(std::exp(-std::abs(x)));};
  for(auto& v:yes){double z=4.*(double(minimum)+1.96-double(v.distance));v.logw=remaining-softplus(-z);remaining-=softplus(z);}
  std::sort(yes.begin(),yes.end(),[&](const Ref& x,const Ref& y){return x.logw!=y.logw?x.logw>y.logw:stable(x,y);});
  c.attention=yes.size()==1?1:float(std::exp(-softplus(yes[1].logw-yes[0].logw)));
 }else{
  std::stable_sort(all.begin(),all.end(),[](const Ref& x,const Ref& y){return x.score>y.score;});
  for(size_t i=0;i<std::min(size_t(2),all.size());++i){project(all[i]);yes.push_back(all[i]);}
 }
 for(size_t i=0;i<std::min(size_t(2),yes.size());++i){const auto& v=yes[i];c.slots[i]=v.slot;c.phases[i]=v.phase;c.scores[i]=v.score;c.raw[i]=v.raw;c.keys[i]=v.key;}
 bool recognized=false;for(const auto& v:all)recognized|=v.score>0;
 if(!recognized){
  for(int i=int(yes.size());i<2;++i)for(int r=0;r<QRANK;++r)c.keys[i][r]=std::tanh(affine(p.key[r],c.raw[i]));
  float delta=0;for(int r=0;r<QRANK;++r){float x=query[r]-c.keys[0][r],y=query[r]-c.keys[1][r];delta+=y*y-x*x;}c.attention=sigmoid(4*delta);
 }
 return c;
}

int main(){using namespace shoe;int n=0;auto ck=[&](bool ok){++n;if(!ok)throw std::runtime_error("A4 selection check "+std::to_string(n));};
 QueryAdapter a(500000,7,true,4,true);auto& s=a.model->levels[1].retained;
 auto clear=[&](int count){s={};s.count=count;s.clock=count;a.head={};
  for(auto& row:a.head.gate)row[D]=-1;
  for(int k=0;k<count;++k){s.units[k].stable[k]=.125f;s.units[k].born=k+1;}};
 auto select=[&](bool full=false){QueryCache c;c.query.fill(.5f);a4_select(a.head,s,c,0,full);return c;};
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
 for(double x:{-1e300,-1000.,-31.36,0.,31.36,1000.,1e300})ck(std::isfinite(a4_softplus(x)));
 clear(16);for(auto& row:a.head.gate)row[D]=1;
 for(auto& row:a.head.key)row[D]=1e30f;
 c=select();same(c,select(true));ck(std::isfinite(c.attention));
 // Minimum-distance pass MUST project every recognized candidate.
 QueryCache x,y;x.query.fill(1);y.query=x.query;int fast=0,full=0;
 a4_select(a.head,s,x,0,false,&fast);a4_select(a.head,s,y,0,true,&full);same(x,y);ck(fast==full);ck(full==64);
 // Broad random adversarial states, empty/single/multiple recognition, tied
 // occurrences, all lesion source modes; optimized selection exactly exhaustive.
 std::mt19937 rng(937);std::uniform_real_distribution<float> u(-1,1);
 for(int trial=0;trial<1000;++trial){clear(trial%17);
  for(auto& row:a.head.gate)for(auto& z:row)z=u(rng);
  for(auto& row:a.head.key)for(auto& z:row)z=u(rng);
  for(int k=0;k<s.count;++k){s.units[k].born=rng()%7;for(auto& z:s.units[k].stable)z=u(rng)*.125f;}
  QueryCache f,g;for(auto& z:f.query)z=u(rng);g.query=f.query;
  a4_select(a.head,s,f,trial%4);a4_select(a.head,s,g,trial%4,true);same(f,g);same(f,independent(a.head,s,f.query,trial%4));ck(std::isfinite(f.attention));
 }
 // Exact equal log weights prefer stable slot/phase regardless of occurrence.
 A4Match ta,tb;ta.log_weight=tb.log_weight=-1000;ta.occurrence={2,1,1,99};tb.occurrence={2,2,1,999};
 ck(a4_weight_before(ta,tb));ck(!a4_weight_before(tb,ta));tb.occurrence.slot=1;ck(a4_weight_before(tb,ta));
 // The oldest candidate can determine the relative band; no age projection stop.
 clear(16);a.head.gate[0][D]=1;for(int r=0;r<QRANK;++r)for(int k=1;k<16;++k)a.head.key[r][k]=std::atanh(-.7f);
 c=select();same(c,independent(a.head,s,c.query,0));ck(c.slots[0]==0);
 clear(0);c=select();ck(c.slots[0]==-1&&c.slots[1]==-1);ck(c.attention==.5f);
 auto blob=query_save(a);auto b=query_load(blob.data(),blob.size());ck(query_save(*b)==blob);
 a.head.key[0][0]=std::numeric_limits<float>::quiet_NaN();blob=query_save(a);
 bool rejected=false;try{auto bad=query_load(blob.data(),blob.size());}catch(...){rejected=true;}ck(rejected);
 std::cout<<n<<" A4 selection checks passed\n";
 std::cout<<"A4 scratch struct bytes "<<sizeof(A4Scratch)<<"; occurrence "<<sizeof(A4Occurrence)<<"; match "<<sizeof(A4Match)<<"; unchanged QueryCache "<<sizeof(QueryCache)<<"\n";
}
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection-only',action='store_true');args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='exp0305a4-native-') as folder:
        folder=Path(folder);cpp=folder/'test.cpp';exe=folder/'test'
        (folder/'readout.hpp').write_text(header('a4'))
        for body in ((SELECT,) if args.selection_only else (NUMERIC,REQUEST,A1,SELECT)):
            cpp.write_text(source('a4')+body)
            for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
                subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(folder),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
                subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()
