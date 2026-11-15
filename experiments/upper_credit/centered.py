"""Private corrected-C fork: language readouts consume changing slow state."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.upper_credit import local_head,screen
from experiments.plateau_100m import common
from shoe_multiscale import Model
ARMS={'cmean':1,'cnorm':2};ACTIVE='cmean'
for arm in ARMS:screen.ARMS[arm]=(1,1)
HEADER='''
struct CenterTrace {Vec value{};float inverse=1;};
struct CenterState {Vec mean{};uint64_t seen=0;std::array<CenterTrace,BLOCK> events{};};
inline CenterTrace centered_readout(const Vec& state,const Vec& mean,int mode){
    CenterTrace t;for(int i=0;i<D;++i)t.value[i]=state[i]-mean[i];
    if(mode==2){t.inverse=1/std::sqrt(energy(t.value)/D+.01f);for(auto& v:t.value)v*=t.inverse;}
    return t;
}
inline Vec centered_vjp(const CenterTrace& t,const Vec& gradient,int mode){
    Vec out=gradient;if(mode==2){float radial=dot(t.value.data(),gradient.data())/D;
        for(int i=0;i<D;++i)out[i]=t.inverse*(gradient[i]-t.value[i]*radial);}
    return out;
}
'''
def build(directory):
    common.build(directory);directory=Path(directory);p=directory/'native.hpp';s=p.read_text()
    s=local_head.replace(s,'struct Level {',HEADER+'\nstruct Level {\n    CenterState center{};')
    s=local_head.replace(s,'std::array<float,4> recall_credit{{0,0,0,0}};',
        'std::array<float,4> recall_credit{{0,0,0,0}};\n    int center_mode=0;')
    s=local_head.replace(s,'uint32_t checkpoint_version()const{','uint32_t checkpoint_version()const{\n        return 16;')
    p.write_text(s);p=directory/'native.cpp';s=p.read_text()
    needle='if(scale>0){mv(l.p.gain,s.state,s.gain);mv(l.p.bias,s.state,s.bias);}'
    s=local_head.replace(s,needle,'''if(scale>0){Vec projected=s.state;
        if(scale>=2&&center_mode){auto& t=l.center.events[slot];t=centered_readout(s.state,l.center.mean,center_mode);projected=t.value;
            if(training){float alpha=1.f/std::min<uint64_t>(1024,++l.center.seen);
                for(int i=0;i<D;++i)l.center.mean[i]+=alpha*(s.state[i]-l.center.mean[i]);}}
        mv(l.p.gain,projected,s.gain);mv(l.p.bias,projected,s.bias);}''')
    needle='vjp(l.p.gain,dgain,task);vjp(l.p.bias,dbias,task);outer(g.gain,dgain,e.state);outer(g.bias,dbias,e.state);'
    s=local_head.replace(s,needle,'''Vec projected_grad{};vjp(l.p.gain,dgain,projected_grad);vjp(l.p.bias,dbias,projected_grad);
            const Vec& readout=scale>=2&&center_mode?l.center.events[t].value:e.state;
            if(scale>=2&&center_mode)projected_grad=centered_vjp(l.center.events[t],projected_grad,center_mode);
            for(int i=0;i<D;++i)task[i]+=projected_grad[i];
            outer(g.gain,dgain,readout);outer(g.bias,dbias,readout);''')
    # Keep offline objective-specific gradient inspection consistent.
    needle='if(scale){vjp(l.p.gain,dgain,task);vjp(l.p.bias,dbias,task);vjp(l.p.predict,dp,dh);}'
    s=local_head.replace(s,needle,'''if(scale){Vec projected_grad{};vjp(l.p.gain,dgain,projected_grad);vjp(l.p.bias,dbias,projected_grad);
            if(scale>=2&&m.center_mode)projected_grad=centered_vjp(l.center.events[t],projected_grad,m.center_mode);
            for(int i=0;i<D;++i)task[i]+=projected_grad[i];vjp(l.p.predict,dp,dh);}''')
    s=local_head.replace(s,'version>12','version>16||(version>12&&version<16)')
    s=local_head.replace(s,'recall_extension=version>=12;','recall_extension=version>=12,center_extension=version==16;')
    s=local_head.replace(s,'return w.out;','w.pod(m.center_mode);for(const auto& l:m.levels)w.pod(l.center);\n    return w.out;')
    s=local_head.replace(s,'if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");',
        '''if(center_extension){r.pod(m.center_mode);if(m.center_mode<0||m.center_mode>2)throw std::runtime_error("invalid center mode");
        for(auto& l:m.levels){r.pod(l.center);for(float v:l.center.mean)if(!std::isfinite(v))throw std::runtime_error("invalid center mean");}}
    if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");''')
    s+='\nextern "C" int center_set(void* p,int mode){return guard([&]{static_cast<shoe::Model*>(p)->center_mode=mode;});}\n'
    p.write_text(s)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(p),'-o',str(directory/'diagnostic.so')],check=True)
    source=(ROOT/'experiments/upper_credit/accessors.cpp').read_text().replace('../../multiscale/native.hpp','native.hpp')
    (directory/'accessors.cpp').write_text(source)
    subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared',str(directory/'accessors.cpp'),'-o',str(directory.parent/'accessors.so')],check=True)
class CenterModel(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._check(self._lib.center_set(self._alive(),ARMS[ACTIVE]))
def install(directory,arm):
    global ACTIVE
    ACTIVE=arm;base,lib=common.install(directory)
    lib.center_set.argtypes=[C.c_void_p,C.c_int];lib.center_set.restype=C.c_int
    Model.loads=classmethod(local_head.loads)
    return base,lib
if __name__=='__main__':
    screen.Model=CenterModel;screen.build=build;screen.install=install;screen.main()
