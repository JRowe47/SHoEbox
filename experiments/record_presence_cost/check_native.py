"""Independent logistic/step references; hybrid coupling and exhaustive routing."""
import os,subprocess,tempfile
from pathlib import Path
from .backend import ROOT,source,header
from experiments.discriminative_keys.backend import header as old_header
from experiments.discriminative_keys.check_native import OBJECTIVE as A5_OBJECTIVE
from experiments.relative_chronological_retention.check_native import SELECT,NUMERIC,REQUEST,A1

PRESENCE=r'''
#include <iostream>
long double loss(long double z,bool y){long double v=y?-z:z;return 4*(std::max(v,0.L)+std::log1p(std::exp(-std::abs(v))));}
long double derivative(long double z,bool y){return y?-4/(1+std::exp(z)):4/(1+std::exp(-z));}
// Independent scalar float32 reference: no production affine/energy/update calls.
shoe::Affine expected(shoe::Affine row,const shoe::Vec& x,bool y,bool candidate){
 float z=0,energy=0;for(int j=0;j<64;++j){z+=row[j]*x[j];energy+=x[j]*x[j];}z+=row[64];
 float p=1.f/(1.f+std::exp(-z));float e=(p-y)*(candidate?4.f:(y?4.f:1.f));float rate=.16f*1.f;float lr=rate/(1.f+energy);
 for(int j=0;j<64;++j)row[j]-=lr*e*x[j];row[64]-=lr*e;return row;
}
int main(){using namespace shoe;int n=0;auto ck=[&](bool ok){++n;if(!ok)throw std::runtime_error("A6 presence check "+std::to_string(n));};
 double worst=0;int fd=0,local_calls=0,positive=0,negative=0;
 for(int i=-200;i<=200;++i)for(int y=0;y<2;++y){float z=i*.25f,p=sigmoid(z),old=(p-y)*(y?4.f:1.f),e=(p-y)*4.f;
  ck(std::isfinite(e)&&std::isfinite(double(loss(z,y))));
  if(y)ck(std::memcmp(&e,&old,sizeof(float))==0);else ck(e==4.f*old);
  double err=std::abs(double(e)-double(derivative(z,y)));worst=std::max(worst,err);ck(err<5e-7);
  long double h=1e-4L;ck(std::abs((loss(z+h,y)-loss(z-h,y))/(2*h)-derivative(z,y))<1e-9L);++fd;
 }
 for(float z:{-1e6f,-100.f,0.f,100.f,1e6f})for(int y=0;y<2;++y){float e=(sigmoid(z)-y)*4.f;ck(std::isfinite(e)&&std::isfinite(double(loss(z,y))));}
 std::mt19937 rng(6147);std::uniform_real_distribution<float> u(-.2f,.2f);
 for(int bytes:{1,2,7,8,12})for(int mask=0;mask<16;++mask)for(int query=0;query<2;++query)for(int request=0;request<2;++request){
  QueryAdapter a(500000,7,true,4,true);a.rate=.16f;a.auxiliary=1.f;a.model->bytes=bytes;
  for(auto& v:a.model->levels[0].retained.last)v=u(rng);for(auto& v:a.model->levels[1].retained.last)v=u(rng);
  for(auto& row:a.head.gate)for(auto& v:row)v=u(rng);
  QueryAdapter b(a);auto gates=a.head.gate;Vec x=source_vector(a.model->levels[1].retained.last);Labels labels;labels.fill(-1);labels[5]=request;
  if(query)labels[0]=255;for(int ph=0;ph<4;++ph)if(mask&(1<<ph))labels[ph+1]=(31+ph)*256+73;
  a.local(labels);b.a5_local_reference(labels);local_calls+=2;
  ck(a.head.key==b.head.key);ck(a.head.value==b.head.value);ck(a.applicability==b.applicability);ck(a.head.gate[4]==b.head.gate[4]);ck(save(*a.model)==save(*b.model));
  ck(a.local_updates==b.local_updates&&a.presence_steps==b.presence_steps&&a.projection_steps==b.projection_steps&&a.gate_targets==b.gate_targets);
  for(int ph=0;ph<4;++ph){bool yes=labels[ph+1]>=0;
   if(bytes>=8&&bytes%4==0){ck(a.head.gate[ph]==expected(gates[ph],x,yes,true));ck(b.head.gate[ph]==expected(gates[ph],x,yes,false));
    if(yes){ck(a.head.gate[ph]==b.head.gate[ph]);++positive;}else ++negative;
   }else ck(a.head.gate[ph]==gates[ph]&&a.head.gate[ph]==b.head.gate[ph]);
  }
  // Identical externally supplied responsibility/error keeps native credit exact.
  a.responsibility=b.responsibility=.375f;std::array<RecallValue,4> av{},bv{};a.backward(av);b.backward(bv);
  ck(save(*a.model)==save(*b.model));ck(a.applicability==b.applicability);
 }
 std::cout<<n<<" A6 presence checks passed; max gradient error "<<worst<<"; finite differences "<<fd<<"; local calls "<<local_calls<<"; candidate positive/negative record rows "<<positive<<"/"<<negative<<"\n";
}
'''

def main():
 # This additional method exists only in test binaries, never in the adapter.
 old=old_header('a5');start=old.index('    void local(');end=old.index('    // See et al.',start)
 reference=old[start:end].replace('void local(', 'void a5_local_reference(')
 test_header=header('a6').replace('    void local(',reference+'    void local(')
 # The inherited A5 reference still checks unchanged values, query gate, keys,
 # core and coupling; only its four expected record-negative multipliers change.
 objective=A5_OBJECTIVE.replace('affine_step(a.head.gate[ph],x,(sigmoid(affine(a.head.gate[ph],x))-yes)*(yes?4.f:1.f),lr);','affine_step(a.head.gate[ph],x,(sigmoid(affine(a.head.gate[ph],x))-yes)*4.f,lr);')
 with tempfile.TemporaryDirectory(prefix='exp0305a6-native-') as folder:
  folder=Path(folder);cpp=folder/'test.cpp';exe=folder/'test';(folder/'readout.hpp').write_text(test_header)
  for body in (NUMERIC,REQUEST,A1,SELECT,objective,PRESENCE):
   cpp.write_text(source('a6')+body)
   for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
    subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(folder),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()
