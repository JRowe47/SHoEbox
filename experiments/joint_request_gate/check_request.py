"""Native local-request update/freeze/continuation checks, including sanitizers."""
import os,subprocess,tempfile
from pathlib import Path
from .adapter import source,HERE,ROOT
CPP=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto ck=[&](bool x){++checks;if(!x)throw std::runtime_error("request-gate check");};
 QueryAdapter a(500000,7,true,4,true);QueryScope scope(a);
 for(int i=0;i<70;++i){a.model->observe(uint8_t(10+i));Labels label{{-1,-1,-1,-1,-1,i%4==0}};a.local(label);a.model->teach(uint8_t(i+1));}
 ck(a.gate_targets==69);auto gate=a.applicability;auto core=save(*a.model);
 for(int i=0;i<70;++i){a.model->observe(uint8_t(30+i));a.model->teach(uint8_t(i+1));ck(a.applicability==gate);}
 ck(core!=save(*a.model));auto bytes=query_save(a);auto b=query_load(bytes.data(),bytes.size());ck(bytes==query_save(*b));ck(b->kind==4);
 for(int i=0;i<35;++i){uint8_t x=uint8_t(i+81),y=uint8_t(i+82);a.model->observe(x);a.model->teach(y);{QueryScope nested(*b);b->model->observe(x);b->model->teach(y);}ck(query_save(a)==query_save(*b));}
 a.training=false;a.model->inference(false);gate=a.applicability;Labels label{{-1,-1,-1,-1,-1,1}};a.local(label);ck(gate==a.applicability);
 std::cout<<checks<<" E7 request checks passed\n";
}
'''
def main():
 with tempfile.TemporaryDirectory(prefix='request-native-') as folder:
  cpp=Path(folder)/'test.cpp';exe=Path(folder)/'test';cpp.write_text(source()+CPP)
  for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
   subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
   subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
