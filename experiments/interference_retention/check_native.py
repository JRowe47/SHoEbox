"""Optimized and ASan/UBSan A1 replacement, frozen instrumentation and E8 numerics."""
import os,subprocess,tempfile
from pathlib import Path
from .backend import ROOT,source,header
from experiments.joint_cold_gate.check_native import CPP as NUMERIC

A1=r'''
#include <iostream>
int main(){using namespace shoe;int n=0;auto ck=[&](bool x){++n;if(!x)throw std::runtime_error("A1 admission check");};
 QueryAdapter a(500000,7,true,4,true);QueryScope scope(a);auto& s=a.model->levels[1].retained;
 s.count=16;s.clock=16;s.have_previous=true;
 for(int k=0;k<16;++k){s.units[k].stable[0]=k==0?1:-1;s.units[k].stable[k+1]=.1f;s.units[k].born=k+1;}
 a.head.gate[0][0]=1;Vec current{},content{};current[2]=1;auto protected_unit=s.units[0];
 int slot=ir_admit(a.model.get(),1,current,content);ck(slot==1);ck(s.units[0].stable==protected_unit.stable);ck(s.units[0].born==1);ck(s.clock==17);
 ck(s.previous==current);ck(s.have_previous);auto frontier=s.last;
 // Every resident unit is now positively recognized: neutral arrivals cannot evict.
 for(auto& u:s.units){u.stable={};u.stable[0]=1;}
 s.previous={};s.last={};auto before=s.units;auto clock=s.clock;current={};current[3]=1;
 ck(ir_admit(a.model.get(),1,current,content)==-1);ck(s.clock==clock);ck(s.previous==current);ck(s.last!=Vec{});
 for(int k=0;k<16;++k)ck(s.units[k].stable==before[k].stable&&s.units[k].born==before[k].born);
 // With zero recognizers, replacement is exactly historical oldest-first.
 a.head.gate={};RecallStream reference=s;current={};current[7]=1;
 int expected=reference.admit(current,content);int actual=ir_admit(a.model.get(),1,current,content);ck(actual==expected);ck(s.clock==reference.clock);
 for(int k=0;k<16;++k){ck(s.units[k].stable==reference.units[k].stable);ck(s.units[k].value==reference.units[k].value);ck(s.units[k].born==reference.units[k].born);}
 auto raw=query_save(a);auto b=query_load(raw.data(),raw.size());ck(query_save(*b)==raw);
 double state[1106],decoded[324];ck(!ir_state(&a,state));ck(!ir_decode(&a,decoded));ck(query_save(a)==raw);
 ck(ir_frozen_span(&a,(const uint8_t*)"abc",3)!=0);
 auto* f=static_cast<QueryAdapter*>(qr_clone(&a,1,0));ck(f);ck(!ir_frozen_span(f,(const uint8_t*)"abc",3));qr_free(f);
 std::cout<<n<<" A1 replacement/lifecycle checks passed\n";
}
'''
def main():
    with tempfile.TemporaryDirectory(prefix='exp0305-native-') as folder:
        folder=Path(folder);cpp=folder/'test.cpp';exe=folder/'test'
        for name,body in [('a1',NUMERIC),('a1',A1)]:
            (folder/'readout.hpp').write_text(header(name));cpp.write_text(source(name)+body)
            for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
                subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(folder),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
                subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()
