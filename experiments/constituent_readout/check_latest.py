"""D2 query policy decisions and paired-source invariance, optimized and sanitized."""
import os,subprocess,tempfile
from pathlib import Path
from .adapter import HERE,ROOT,source
CPP=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto check=[&](bool x){++checks;if(!x)throw std::runtime_error("D2 policy check");};
 QueryAdapter a(500000,1,true,2,false);auto& q=a.model->levels[0].retained;
 q.count=2;q.clock=2;q.units[0].born=1;q.units[1].born=2;q.units[0].stable[0]=1;q.units[1].stable[0]=.1f;a.head.gate[4][0]=1;
 auto old=query_forward(a.head,*a.model,0,false),latest=query_forward(a.head,*a.model,0,true);
 check(old.query_slot==0);check(latest.query_slot==1);
 q.units[1].stable[0]=-.1f;latest=query_forward(a.head,*a.model,0,true);check(latest.query_slot==0);
 q.units[0].stable[0]=-1;old=query_forward(a.head,*a.model,0,false);latest=query_forward(a.head,*a.model,0,true);check(old.query_slot==latest.query_slot);
 auto& r=a.model->levels[1].retained;r.count=2;r.clock=2;r.units[0].born=1;r.units[1].born=2;r.units[0].stable[0]=.3f;r.units[1].stable[0]=-.3f;r.units[0].stable[1]=.2f;r.units[1].stable[1]=-.1f;
 a.head.gate[0][D]=5;a.head.key[0][0]=1;a.head.value[0][1]=1;
 auto before=query_forward(a.head,*a.model,0,true);std::swap(r.units[0],r.units[1]);auto after=query_forward(a.head,*a.model,0,true);
 for(int i=0;i<VRANK;++i)check(std::abs(before.h[i]-after.h[i])<1e-6);
 check(a.model->parameter_count()+QPARAMS<=500000);auto raw=query_save(a);auto b=query_load(raw.data(),raw.size());check(b->kind==2);check(raw==query_save(*b));
 std::cout<<checks<<" D2 routing checks passed\n";
}
'''
def main():
 with tempfile.TemporaryDirectory(prefix='paired-d2-tests-') as folder:
  cpp=Path(folder)/'check.cpp';exe=Path(folder)/'check';cpp.write_text(source()+CPP)
  for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
   subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
   subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
