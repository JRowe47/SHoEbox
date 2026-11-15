"""Isolate main-language content VJPs and amplify only selected experts."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.upper_credit import local_head,screen
from experiments.plateau_100m import common
from shoe_multiscale import Model
ARMS={'s4_16':(4,16),'s16_64':(16,64),'C':(1,1)}
ACTIVE='C'
for arm in ARMS:screen.ARMS[arm]=(1,1)
def build(directory):
    common.build(directory);directory=Path(directory);p=directory/'native.hpp';s=p.read_text()
    s=local_head.replace(s,'struct Level {','struct Level {\n    std::vector<Vec> direct_language_credit;')
    s=local_head.replace(s,'std::array<float,4> recall_credit{{0,0,0,0}};',
        'std::array<float,4> recall_credit{{0,0,0,0}};\n    std::array<float,4> state_expert_gain{{1,1,1,1}};')
    s=local_head.replace(s,'uint32_t checkpoint_version()const{','uint32_t checkpoint_version()const{\n        return 15;')
    p.write_text(s);p=directory/'native.cpp';s=p.read_text()
    s=s.replace('namespace shoe {','namespace shoe {\nstd::vector<float> diagnostic_content_gradient(const Model&,int,int);',1)
    needle='if(recall_credit[scale]>0)l.recall_content_gradient.emplace_back();'
    s=local_head.replace(s,needle,needle+'\n        if(scale>=2)l.direct_language_credit.emplace_back();')
    needle='Shared g{};AddressParameters ag{};std::array<Expert,BLOCK> eg{};'
    s=local_head.replace(s,needle,'''if(scale>=2&&state_expert_gain[scale]!=1){
        auto language=diagnostic_content_gradient(*this,scale,0);
        for(int t=0;t<s.count;++t){auto& dst=l.direct_language_credit.at(s.events[t].content_id);
            for(int j=0;j<D;++j)dst[j]+=language[t*D+j];}}
    '''+needle)
    needle='Vec gated=pre;for(auto& x:gated)x*=c.gate;outer(eg[k].w,gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);'
    s=local_head.replace(s,needle,'''Vec gated=pre;for(auto& x:gated)x*=c.gate;Vec expert_gated=gated;
                if(scale>=2&&state_expert_gain[scale]!=1){for(int i=0;i<D;++i){
                    float extra=(state_expert_gain[scale]-1)*l.direct_language_credit[ci-1][i]*(1-c.content[i]*c.content[i]);
                    eg[k].bias[i]+=extra;expert_gated[i]+=c.gate*extra;}}
                outer(eg[k].w,expert_gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);''')
    needle='l.constituents.clear();l.routing.constituents.clear();'
    s=local_head.replace(s,needle,'l.direct_language_credit.clear();'+needle)
    needle='for(auto& l:levels){l.recall_credit_state=RecallCreditState{};l.recall_content_gradient.clear();'
    s=local_head.replace(s,needle,needle+'l.direct_language_credit.clear();')
    s=local_head.replace(s,'version>12','version>15||version==13||version==14')
    s=local_head.replace(s,'recall_extension=version>=12;','recall_extension=version>=12,state_extension=version==15;')
    s=local_head.replace(s,'return w.out;',
        'w.pod(m.state_expert_gain);for(const auto& l:m.levels)w.vector(l.direct_language_credit);\n    return w.out;')
    s=local_head.replace(s,'if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");',
        'if(state_extension){r.pod(m.state_expert_gain);for(size_t i=0;i<m.levels.size();++i){auto& l=m.levels[i];l.direct_language_credit.resize(m.training&&i>=2?l.constituents.size():0);r.vector(l.direct_language_credit);}}\n    if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");')
    s+='''\nextern "C" int state_credit(void* p,float a,float b){return guard([&]{
        auto& m=*static_cast<shoe::Model*>(p);m.state_expert_gain={{1,1,a,b}};
    });}\n'''
    p.write_text(s)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(p),'-o',str(directory/'diagnostic.so')],check=True)
    source=(ROOT/'experiments/upper_credit/accessors.cpp').read_text().replace('../../multiscale/native.hpp','native.hpp')
    (directory/'accessors.cpp').write_text(source)
    subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared',str(directory/'accessors.cpp'),'-o',str(directory.parent/'accessors.so')],check=True)
class StateModel(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._check(self._lib.state_credit(self._alive(),*ARMS[ACTIVE]))
def install(directory,arm):
    global ACTIVE
    ACTIVE=arm;base,lib=common.install(directory)
    lib.state_credit.argtypes=[C.c_void_p,C.c_float,C.c_float];lib.state_credit.restype=C.c_int
    Model.loads=classmethod(local_head.loads)
    return base,lib
if __name__=='__main__':
    screen.Model=StateModel;screen.build=build;screen.install=install;screen.main()
