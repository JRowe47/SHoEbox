"""Gate finite differences and joint gradient contracts under ASan/UBSan."""
import os,subprocess,tempfile
from pathlib import Path
from .adapter import source,HERE,ROOT
CPP=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto check=[&](bool x){++checks;if(!x)throw std::runtime_error("EXP01-E native check");};
 QueryAdapter a(500000,7,true,3,true);std::mt19937 rng(3);std::normal_distribution<float> dist(0,.1);
 for(auto& x:a.applicability)x=dist(rng);for(auto& x:a.head.output)x=dist(rng);for(auto& x:a.head.bias)x=dist(rng);
 for(int i=0;i<19;++i)a.model->observe(uint8_t(40+i));a.prepare();
 auto loss=[&](){a.prepare();float z=a.logit(4);return softplus(z)-z;};
 float base=loss(),g=a.gate_value,r=a.residual(4),error=sigmoid(g*r)-1;
 auto raw=a.gate_raw;auto old=a.applicability;auto head=a.head;
 for(int i=0;i<=D;++i){float epsilon=.001f;a.applicability[i]=old[i]+epsilon;float plus=loss();a.applicability[i]=old[i]-epsilon;float minus=loss();a.applicability[i]=old[i];
  float expected=error*r*g*(1-g)*(i==D?1:raw[i]);check(std::abs((plus-minus)/(2*epsilon)-expected)<.0002f);}
 a.prepare();a.errors[4]=error;std::array<RecallValue,4> unused{};a.backward(unused);
 for(int i=0;i<=D;++i){float expected=old[i]-a.rate/(1+energy(raw))*error*r*g*(1-g)*(i==D?1:raw[i]);check(std::abs(a.applicability[i]-expected)<1e-7);}
 for(int i=0;i<VRANK;++i){float expected=head.output[4*VRANK+i]-a.rate/(1+dot(a.cache.h.data(),a.cache.h.data(),VRANK))*error*g*a.cache.h[i];check(std::abs(a.head.output[4*VRANK+i]-expected)<1e-7);}
 check(a.targets==1&&a.gate_targets==1);auto saved=query_save(a);auto b=query_load(saved.data(),saved.size());check(saved==query_save(*b));
 for(int i=0;i<64;++i){QueryScope scope(a);a.model->observe(uint8_t(i));a.model->teach(uint8_t(i+1));}
 a.model->finish();saved=query_save(a);b=query_load(saved.data(),saved.size());check(saved==query_save(*b));
 std::cout<<checks<<" EXP01-E native checks passed\n";
}
'''
def main():
 with tempfile.TemporaryDirectory(prefix='joint-native-') as folder:
  cpp=Path(folder)/'test.cpp';exe=Path(folder)/'test';cpp.write_text(source()+CPP)
  for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
   subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
   subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
