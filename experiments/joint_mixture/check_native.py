"""Exact probability-mixture derivatives, normalized output and sanitizer checks."""
import os,subprocess,tempfile
from pathlib import Path
from .adapter import source,HERE,ROOT
CPP=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto check=[&](bool x){++checks;if(!x)throw std::runtime_error("E3 native check");};
 QueryAdapter a(500000,7,true,3,true);for(int i=0;i<19;++i)a.model->observe(uint8_t(i+40));a.applicability[D]=.3f;
 std::mt19937 rng(3);std::normal_distribution<float> random(0,.2);for(auto& v:a.head.bias)v=random(rng);
 QueryScope scope(a);uint8_t target=97;
 auto loss=[&](){a.prepare();query_target(a.model.get(),target);return a.mixture_loss;};
 loss();float g=a.gate_value,r=a.responsibility;auto raw=a.gate_raw;auto old=a.applicability;
 for(int i=0;i<=D;++i){float eps=.002f;a.applicability[i]=old[i]+eps;float plus=loss();a.applicability[i]=old[i]-eps;float minus=loss();a.applicability[i]=old[i];
  check(std::abs((plus-minus)/(2*eps)-(g-r)*(i==D?1:raw[i]))<.0003f);}
 int node=0;
 for(int d=0;d<8;++d){int bit=(target>>(7-d))&1;float saved=a.head.bias[node],eps=.002f;
  a.head.bias[node]=saved+eps;float p=loss();a.head.bias[node]=saved-eps;float n=loss();a.head.bias[node]=saved;
  check(std::abs((p-n)/(2*eps)-r*(sigmoid(a.residual(node))-bit))<.0004f);
  float zn=a.model->decoder_bias[node]+dot(a.model->decoder[node].data(),a.model->features.data());for(auto& l:a.model->levels)zn+=dot(l.retrieval.read.data()+node*RECALL_V,l.read_now.value.data(),RECALL_V);
  saved=a.model->decoder_bias[node];a.model->decoder_bias[node]=saved+eps;p=loss();a.model->decoder_bias[node]=saved-eps;n=loss();a.model->decoder_bias[node]=saved;
  check(std::abs((p-n)/(2*eps)-(1-r)*(sigmoid(zn)-bit))<.0004f);node=2*node+1+bit;
 }
 loss();a.backward(*new std::array<RecallValue,4>{});
 for(int i=0;i<=D;++i)check(std::abs(a.applicability[i]-(old[i]-a.rate/(1+energy(raw))*(g-r)*(i==D?1:raw[i])))<1e-7);
 float lp[256];a.model->log_probs(lp);double sum=0;for(auto p:lp)sum+=std::exp(p);check(std::abs(sum-1)<1e-5);
 auto blob=query_save(a);auto b=query_load(blob.data(),blob.size());check(blob==query_save(*b));
 for(int i=0;i<64;++i){a.model->observe(uint8_t(i));a.model->teach(uint8_t(i+1));}a.model->finish();blob=query_save(a);b=query_load(blob.data(),blob.size());check(blob==query_save(*b));
 std::cout<<checks<<" E3 native checks passed\n";
}
'''
CPP=CPP.replace('a.backward(*new std::array<RecallValue,4>{});','std::array<RecallValue,4> scratch{};a.backward(scratch);')
def main():
 with tempfile.TemporaryDirectory(prefix='mixture-native-') as folder:
  cpp=Path(folder)/'test.cpp';exe=Path(folder)/'test';cpp.write_text(source()+CPP)
  for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
   subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
   subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
