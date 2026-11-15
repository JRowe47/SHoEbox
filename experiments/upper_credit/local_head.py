"""Frozen diagnostic fork: direct next-byte language heads at L2/L3."""
import ctypes as C
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.upper_credit import screen
from experiments.plateau_100m import common
from multiscale import model as runtime
from shoe_multiscale import Model
ARMS={'lm01':(.1,.1),'lm1':(1,1),'C':(0,0)}
ACTIVE='C'
for arm in ARMS:screen.ARMS[arm]=(1,1)

def replace(s,old,new,count=1):
    assert s.count(old)==count,(old,s.count(old))
    return s.replace(old,new)

def build(directory):
    common.build(directory);directory=Path(directory);p=directory/'native.hpp';s=p.read_text()
    s=replace(s,'struct Level {','''struct LocalLanguageTrace {
    std::array<float,255> scores{},counts{},ones{};
};
struct LocalLanguageHead {
    std::array<Vec,255> weights{};std::array<float,255> bias{};
    std::array<LocalLanguageTrace,BLOCK> events{};
};
struct LocalLanguageGradient {std::array<Vec,255> weights{};std::array<float,255> bias{};};
struct Level {
    LocalLanguageHead local_language{};''')
    s=replace(s,'std::array<float,4> recall_credit{{0,0,0,0}};',
        'std::array<float,4> recall_credit{{0,0,0,0}};\n    std::array<float,4> local_language_weight{{0,0,0,0}};')
    s=replace(s,'uint32_t checkpoint_version()const{',
        'uint32_t checkpoint_version()const{\n        for(float v:local_language_weight)if(v>0)return 13;')
    p.write_text(s);p=directory/'native.cpp';s=p.read_text()
    s=replace(s,'auto& l=levels[scale];\n        int span=1;',
        'auto& l=levels[scale];l.local_language.weights=decoder;\n        int span=1;')
    s=replace(s,'if(routing_arm==3)n+=(levels.size()-1)*D*D;',
        'if(routing_arm==3)n+=(levels.size()-1)*D*D;\n    for(float v:local_language_weight)if(v>0)n+=255*(D+1);')
    needle='if(training){\n        e.content_id=int(l.constituents.size());'
    s=replace(s,needle,'''if(training&&local_language_weight[scale]>0){
        auto& t=l.local_language.events[slot];t=LocalLanguageTrace{};
        for(int n=0;n<255;++n)t.scores[n]=l.local_language.bias[n]+dot(l.local_language.weights[n].data(),e.content.data());
    }
    '''+needle)
    s=replace(s,'Shared g{};AddressParameters ag{};std::array<Expert,BLOCK> eg{};',
        'std::unique_ptr<LocalLanguageGradient> local_gradient;\n    if(local_language_weight[scale]>0)local_gradient=std::make_unique<LocalLanguageGradient>();\n    Shared g{};AddressParameters ag{};std::array<Expert,BLOCK> eg{};')
    needle='auto& c=l.constituents.at(e.content_id);for(int i=0;i<D;++i)c.grad[i]+=dc[i];'
    s=replace(s,needle,'''if(local_gradient){const auto& trace=l.local_language.events[t];
            for(int node=0;node<255;++node)if(trace.counts[node]){
                float error=(sigmoid(trace.scores[node])*trace.counts[node]-trace.ones[node])/std::max(1,uses);
                local_gradient->bias[node]+=error;
                for(int j=0;j<D;++j){dc[j]+=local_language_weight[scale]*error*l.local_language.weights[node][j];
                    local_gradient->weights[node][j]+=error*e.content[j];}}}
        '''+needle)
    s=replace(s,'float lr=scale?slow_rate:learning_rate;',
        '''if(local_gradient){for(int node=0;node<255;++node)step(l.local_language.weights[node],local_gradient->weights[node],head_rate);
        step(l.local_language.bias,local_gradient->bias,head_rate);}
    float lr=scale?slow_rate:learning_rate;''')
    needle='if(training){\n        if(semantic_laces)for(size_t scale=1;scale<levels.size();++scale)'
    s=replace(s,needle,'''if(training){
        for(size_t scale=0;scale<levels.size();++scale)if(local_language_weight[scale]>0&&levels[scale].s.count){
            auto& trace=levels[scale].local_language.events[levels[scale].s.count-1];int current=0;
            for(int depth=0;depth<8;++depth){int bit=(target>>(7-depth))&1;
                trace.counts[current]+=1;trace.ones[current]+=bit;current=2*current+1+bit;}}
        if(semantic_laces)for(size_t scale=1;scale<levels.size();++scale)''')
    needle='int count=int(m.levels.size()),target=int(m.parameter_count())-(m.routing_arm==3?int(m.levels.size()-1)*D*D:0);'
    s=replace(s,needle,needle+'for(float v:m.local_language_weight)if(v>0)target-=255*(D+1);')
    s=replace(s,'return w.out;','if(version>=13){w.pod(m.local_language_weight);for(const auto& l:m.levels)w.pod(l.local_language);}\n    return w.out;')
    s=replace(s,'version>12','version>13')
    s=replace(s,'const bool credit_extension=version>=11,recall_extension=version>=12;',
        'const bool credit_extension=version>=11,recall_extension=version>=12,local_extension=version>=13;')
    s=replace(s,'if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");',
        '''if(local_extension){r.pod(m.local_language_weight);for(auto& l:m.levels)r.pod(l.local_language);}
    if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");''')
    s+='''\nextern "C" int local_credit(void* p,float a,float b){return guard([&]{
        auto& m=*static_cast<shoe::Model*>(p);m.local_language_weight={{0,0,a,b}};
    });}\n'''
    p.write_text(s)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(p),'-o',str(directory/'diagnostic.so')],check=True)
    # Accessors must use this fork's Level layout.
    source=(ROOT/'experiments/upper_credit/accessors.cpp').read_text().replace('../../multiscale/native.hpp','native.hpp')
    (directory/'accessors.cpp').write_text(source)
    subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared',str(directory/'accessors.cpp'),'-o',str(directory.parent/'accessors.so')],check=True)

ORIGINAL_LOADS=Model.loads.__func__
def loads(cls,data):
    if len(data)>56 and struct.unpack_from('=I',data,48)[0] in (13,14,15,16):
        if data[:8]!=b'SHOEMS01' or hashlib.sha256(data[40:]).digest()!=data[8:40]:raise ValueError('corrupted checkpoint')
        return cls._wrap(runtime._library().ms_load(data[40:],len(data)-40),'addressed')
    return ORIGINAL_LOADS(cls,data)

class LocalModel(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._check(self._lib.local_credit(self._alive(),*ARMS[ACTIVE]))

def install(directory,arm):
    global ACTIVE
    ACTIVE=arm;base,lib=common.install(directory)
    lib.local_credit.argtypes=[C.c_void_p,C.c_float,C.c_float];lib.local_credit.restype=C.c_int
    Model.loads=classmethod(loads)
    return base,lib

if __name__=='__main__':
    screen.Model=LocalModel;screen.build=build;screen.install=install;screen.main()
