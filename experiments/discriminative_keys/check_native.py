"""Independent objective/reference updates plus unchanged exhaustive A4 routing."""
import os,subprocess,tempfile
from pathlib import Path
from .backend import ROOT,source,header
from experiments.relative_chronological_retention.check_native import SELECT,NUMERIC,REQUEST,A1
OBJECTIVE=r'''
#include <iostream>
// Separate scalar long-double reference: direct quotient/chain-rule form,
// not the production simplified gradient or its softmax helper.
struct Reference {long double loss=0;std::array<long double,16> gradient{};};
Reference reference(const std::array<long double,16>& z,int label){
 Reference o;long double weights[256],denom=0,maxs=-1e300L;
 for(int b=0;b<256;++b){long double d=0;for(int r=0;r<16;++r){long double delta=z[r]-shoe::byte_code(b,r);d+=delta*delta;}weights[b]=-4*d;maxs=std::max(maxs,weights[b]);}
 for(auto& w:weights){w=std::exp(w-maxs);denom+=w;}
 o.loss=-std::log(weights[label]/denom)/16;
 for(int r=0;r<16;++r){long double delta=z[r]-shoe::byte_code(label,r);o.loss+=.25L*delta*delta;
  long double dlogsum=0;for(int b=0;b<256;++b)dlogsum+=(weights[b]/denom)*(-8*(z[r]-shoe::byte_code(b,r)));
  o.gradient[r]=(.5L*delta+(8*delta+dlogsum)/16)*(1-z[r]*z[r]);}
 return o;
}
void original_local(shoe::QueryAdapter& a,const shoe::Labels& labels){using namespace shoe;
 const float lr=a.rate*a.auxiliary;
 if(a.model->bytes>=2){Vec x=source_vector(a.model->levels[0].retained.last);bool yes=labels[0]>=0;
  bool request=labels[5]>0;affine_step(a.applicability,x,(sigmoid(affine(a.applicability,x))-request)*(request?4.f:1.f),lr);
  affine_step(a.head.gate[4],x,(sigmoid(affine(a.head.gate[4],x))-yes)*(yes?4.f:1.f),lr);
  if(yes)for(int r=0;r<QRANK;++r)reconstruct(a.head.key[4*QRANK+r],x,byte_code(labels[0],r),lr*4);
 }
 if(a.model->bytes>=8&&a.model->bytes%4==0){Vec x=source_vector(a.model->levels[1].retained.last);
  for(int ph=0;ph<4;++ph){bool yes=labels[ph+1]>=0;
   affine_step(a.head.gate[ph],x,(sigmoid(affine(a.head.gate[ph],x))-yes)*(yes?4.f:1.f),lr);
   if(yes){int key=labels[ph+1]/256,value=labels[ph+1]%256;
    for(int r=0;r<QRANK;++r)reconstruct(a.head.key[ph*QRANK+r],x,byte_code(key,r),lr*4);
    for(int r=0;r<VRANK;++r)reconstruct(a.head.value[ph*VRANK+r],x,value_code(value,r),lr*4);}
  }
 }
}
int main(){using namespace shoe;int n=0;auto ck=[&](bool ok){++n;if(!ok)throw std::runtime_error("A5 objective check "+std::to_string(n));};
 std::mt19937 rng(5117);std::uniform_real_distribution<float> u(-.95,.95);
 int collisions=0;for(int a=0;a<256;++a)for(int b=0;b<a;++b)if(a5_codebook()[a]==a5_codebook()[b]){++collisions;std::cout<<"code collision "<<b<<" "<<a<<"\n";}
 double worst=0;
 for(int label=0;label<256;++label){LowRank z;std::array<long double,16> v;
  for(int r=0;r<16;++r){z[r]=label==0?1.f:label==255?-1.f:u(rng);v[r]=z[r];ck(a5_codebook()[label][r]==byte_code(label,r));}
  auto ref=reference(v,label);auto o=a5_objective(z,label);ck(std::isfinite(o.attraction+o.discrimination));ck(std::abs(o.attraction+o.discrimination-double(ref.loss))<1e-12);
  double sum=0;for(auto p:o.probabilities){ck(std::isfinite(p)&&p>0);sum+=p;}ck(std::abs(sum-1)<1e-14);
  for(int r=0;r<16;++r){ck(std::abs(o.gradient[r]-double(ref.gradient[r]))<6e-8);worst=std::max(worst,std::abs(o.gradient[r]-double(ref.gradient[r])));
   if(label%16==1){auto plus=v,minus=v;long double h=1e-5L,a=std::atanh(v[r]);plus[r]=std::tanh(a+h);minus[r]=std::tanh(a-h);
    ck(std::abs((reference(plus,label).loss-reference(minus,label).loss)/(2*h)-ref.gradient[r])<1e-8L);}
  }
 }
 // Independent pre-update coupled step, unchanged normalized affine arithmetic.
 for(int label=0;label<256;++label){std::array<Affine,16> rows{},expected;Vec x{};
  for(auto& v:x)v=u(rng)*.1f;for(auto& row:rows)for(auto& v:row)v=u(rng)*.1f;
  expected=rows;LowRank z;for(int r=0;r<16;++r)z[r]=std::tanh(affine(rows[r],x));auto o=a5_objective(z,label);
  for(int r=0;r<16;++r)affine_step(expected[r],x,o.gradient[r],.16f*1.f*4);
  a5_reconstruct(rows.data(),x,label,.16f*1.f*4);ck(rows==expected);
  LowRank after;for(int r=0;r<16;++r)after[r]=std::tanh(affine(rows[r],x));auto b=a5_objective(after,label);ck(b.attraction+b.discrimination<o.attraction+o.discrimination);
 }
 // Test every phase, all negative/positive labels; non-key updates exactly old.
 for(int mode=0;mode<6;++mode){QueryAdapter a(500000,7,true,4,true);a.model->bytes=8;
  for(auto& v:a.model->levels[0].retained.last)v=u(rng)*.1f;
  for(auto& v:a.model->levels[1].retained.last)v=u(rng)*.1f;
  QueryAdapter b(a);Labels labels;labels.fill(-1);labels[5]=mode%2;
  if(mode==1)labels[0]=193;if(mode>=2)labels[mode-1]=255*256+73;
  a.local(labels);original_local(b,labels);
  ck(a.head.gate==b.head.gate);ck(a.head.value==b.head.value);ck(a.applicability==b.applicability);ck(save(*a.model)==save(*b.model));
  if(mode==0)ck(a.head.key==b.head.key);
  auto snap=query_save(a);auto loaded=query_load(snap.data(),snap.size());ck(query_save(*loaded)==snap);
 }
 std::cout<<n<<" A5 objective checks passed; worst gradient error "<<worst<<"; collision pairs "<<collisions<<"\n";
 std::cout<<"A5 fixed read-only training codebook bytes "<<sizeof(A5Codebook)<<"; objective scratch "<<sizeof(A5Objective)<<"; z "<<sizeof(LowRank)<<"; no extra persistent model state\n";
}
'''
def main():
 with tempfile.TemporaryDirectory(prefix='exp0305a5-native-') as folder:
  folder=Path(folder);cpp=folder/'test.cpp';exe=folder/'test';(folder/'readout.hpp').write_text(header('a5'))
  for body in (NUMERIC,REQUEST,A1,SELECT,OBJECTIVE):
   cpp.write_text(source('a5')+body)
   for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
    subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(folder),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()
