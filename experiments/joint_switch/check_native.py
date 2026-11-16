"""E2 switch, gate and core-credit derivatives in optimized/sanitized builds."""
import os,subprocess,tempfile
from pathlib import Path
from .adapter import source,HERE,ROOT
CPP=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto check=[&](bool x){++checks;if(!x)throw std::runtime_error("E2 check");};
 QueryAdapter a(500000,7,true,3,true);for(int i=0;i<19;++i)a.model->observe(uint8_t(i+40));a.applicability[D]=.3f;a.head.bias[4]=1.1f;
 QueryScope scope(a);auto loss=[&](){a.prepare();float z=query_combine(a.model.get(),4,2.4f);return softplus(z)-z;};
 float base=loss(),g=a.gate_value,e=sigmoid((1-g)*2.4f+g*1.1f)-1;
 check(std::abs(query_native_credit(a.model.get())-(1-g))<1e-7);
 auto old=a.applicability;auto raw=a.gate_raw;
 for(int i=0;i<=D;++i){float step=.001f;a.applicability[i]=old[i]+step;float p=loss();a.applicability[i]=old[i]-step;float n=loss();a.applicability[i]=old[i];
  float analytic=e*(1.1f-2.4f)*g*(1-g)*(i==D?1:raw[i]);check(std::abs((p-n)/(2*step)-analytic)<.00025f);}
 a.prepare();query_combine(a.model.get(),4,2.4f);a.errors[4]=e;std::array<RecallValue,4> unused{};a.backward(unused);
 for(int i=0;i<=D;++i){float expected=old[i]-a.rate/(1+energy(raw))*e*(1.1f-2.4f)*g*(1-g)*(i==D?1:raw[i]);check(std::abs(a.applicability[i]-expected)<1e-7);}
 auto blob=query_save(a);auto b=query_load(blob.data(),blob.size());check(blob==query_save(*b));
 for(int i=0;i<64;++i){a.model->observe(uint8_t(i));a.model->teach(uint8_t(i+1));}a.model->finish();
 blob=query_save(a);b=query_load(blob.data(),blob.size());check(blob==query_save(*b));
 std::cout<<checks<<" E2 checks passed\n";
}
'''
def main():
 with tempfile.TemporaryDirectory(prefix='switch-native-') as folder:
  cpp=Path(folder)/'test.cpp';exe=Path(folder)/'test';cpp.write_text(source()+CPP)
  for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
   subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
   subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
