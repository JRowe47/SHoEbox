"""Public E8 C ABI lifecycle and qualified numerical checks under sanitizers."""
import os
from pathlib import Path
import subprocess
import tempfile
from multiscale.e8.backend import source,HERE,ROOT
from experiments.joint_cold_gate.check_native import CPP as NUMERICAL
from experiments.joint_cold_gate.check_request import CPP as REQUEST

PUBLIC=r'''
#include <iostream>
int main(){using namespace shoe;int checks=0;auto ck=[&](bool v){++checks;if(!v)throw std::runtime_error("public E8 check");};
 auto* a=static_cast<QueryAdapter*>(qr_new(500000,7,1,4,1));ck(a);ck(!qr_configure(a,.16f,1.f,.1f));
 auto* b=static_cast<QueryAdapter*>(pe_clone(a,0));ck(b);
 uint8_t x[71],y[71],mask[71];int32_t lab[71*6];
 for(int i=0;i<71;++i){x[i]=uint8_t(13+i);y[i]=uint8_t(14+i);mask[i]=i%3!=0;
  for(int j=0;j<6;++j)lab[6*i+j]=j==5?i%2:-1;}
 double l,r;size_t nt,nr;
 ck(!pe_run(a,x,y,mask,lab,71,&l,&nt));ck(!qr_train(b,x,y,mask,lab,71,&r,&nr));
 ck(l==r&&nt==nr);ck(query_save(*a)==query_save(*b));
 auto saved=query_save(*a);auto* loaded=static_cast<QueryAdapter*>(pe_load(saved.data(),saved.size()));ck(loaded);
 ck(query_save(*loaded)==saved);ck(pe_core(loaded)==loaded->model.get());
 lab[0]=256;ck(pe_run(a,x,y,mask,lab,71,&l,&nt)!=0);ck(query_save(*a)==saved);lab[0]=-1;
 float out[256];ck(pe_observe(a,19,out)!=0);ck(query_save(*a)==saved);
 auto* f=static_cast<QueryAdapter*>(pe_clone(a,1));ck(f&&!f->training&&!f->model->training);
 auto gate=f->applicability;auto head=f->head;
 ck(!pe_observe(f,19,out));double mass=0;for(float v:out)mass+=std::exp(v);ck(std::abs(mass-1)<1e-5);
 ck(!pe_run(f,x,y,nullptr,nullptr,71,&l,&nt));ck(nt==71);ck(f->applicability==gate);ck(std::memcmp(&head,&f->head,sizeof(head))==0);
 ck(pe_run(f,x,y,nullptr,lab,71,&l,&nt)!=0);
 auto* fc=static_cast<QueryAdapter*>(pe_clone(f,0));ck(!fc->training);ck(query_save(*f)==query_save(*fc));
 for(size_t n: {size_t(0),size_t(8),saved.size()-1})ck(pe_load(saved.data(),n)==nullptr);
 auto corrupt=saved;corrupt[0]^=1;ck(pe_load(corrupt.data(),corrupt.size())==nullptr);
 for(int kind: {0,1,2,3}){QueryAdapter control(500000,7,true,kind,true);auto raw=query_save(control);ck(pe_load(raw.data(),raw.size())==nullptr);}
 for(auto* p:{a,b,loaded,f,fc})qr_free(p);
 std::cout<<checks<<" public E8 lifecycle checks passed\n";
}
'''

def main():
 with tempfile.TemporaryDirectory(prefix='public-e8-native-') as folder:
  cpp=Path(folder)/'test.cpp';exe=Path(folder)/'test'
  # Run original numerical suites on the public implementation as well.
  for name,body in [('numerical',NUMERICAL),('request',REQUEST),('public',PUBLIC)]:
   cpp.write_text(source()+body)
   for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
    subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
if __name__=='__main__':main()
