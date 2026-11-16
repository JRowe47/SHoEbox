"""E5 all-byte calibration, mixture-gate derivatives, optimized and sanitized."""
import os,subprocess,tempfile
from pathlib import Path
from .adapter import source,HERE,ROOT
CPP=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto check=[&](bool x){++checks;if(!x)throw std::runtime_error("E5 native check");};
 QueryAdapter a(500000,7,true,3,true);for(int i=0;i<19;++i)a.model->observe(uint8_t(i+40));a.applicability[D+3]=.3f;for(int i=0;i<5;++i)a.head.gate[i][D]=(i-2)*.4f;QueryScope scope(a);
 auto loss=[&](){a.prepare();query_target(a.model.get(),97);return a.mixture_loss;};loss();auto raw=a.gate_raw;auto extra=a.gate_context;auto old=a.applicability;float g=a.gate_value,r=a.responsibility;
 for(int i=0;i<D+4;++i){float eps=.002f;a.applicability[i]=old[i]+eps;float p=loss();a.applicability[i]=old[i]-eps;float n=loss();a.applicability[i]=old[i];check(std::abs((p-n)/(2*eps)-(g-r)*(i<D?raw[i]:(i<D+3?extra[i-D]:1.f)))<.0004f);}
 for(int byte=0;byte<256;++byte){for(int bit=0;bit<8;++bit){float p=((byte>>bit)&1)?1.f-1e-6f:1e-6f;a.byte_logits[bit]=std::log(p)-std::log1p(-p);}
  std::array<float,511> tree{};for(int n=0;n<255;++n){float z=a.residual(n);tree[2*n+1]=tree[n]-softplus(z);tree[2*n+2]=tree[n]-softplus(-z);}
  check(int(std::max_element(tree.begin()+255,tree.end())-(tree.begin()+255))==byte);double sum=0;for(int i=255;i<511;++i)sum+=std::exp(tree[i]);check(std::abs(sum-1)<1e-5);
 }
 auto blob=query_save(a);auto b=query_load(blob.data(),blob.size());check(blob==query_save(*b));
 for(int i=0;i<64;++i){a.model->observe(uint8_t(i));a.model->teach(uint8_t(i+1));}a.model->finish();blob=query_save(a);b=query_load(blob.data(),blob.size());check(blob==query_save(*b));
 check(a.model->parameter_count()+QPARAMS+D+4<=500000);check(QPARAMS+D+4==7673);
 std::cout<<checks<<" E5 native checks passed\n";
}
'''
def main():
 with tempfile.TemporaryDirectory(prefix='calibrated-native-') as folder:
  cpp=Path(folder)/'test.cpp';exe=Path(folder)/'test';cpp.write_text(source()+CPP)
  for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
   subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
   subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
