"""Local byte teaching restricted to the selected experts' compute parameters."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.upper_credit import local_head,screen
from experiments.plateau_100m import common
ARMS={'lme1':(1,1),'lme4':(4,4)}
local_head.ARMS.update(ARMS)
for key in ARMS:screen.ARMS[key]=(1,1)
def build(directory):
    local_head.build(directory);directory=Path(directory);p=directory/'native.hpp';s=p.read_text()
    s=local_head.replace(s,'LocalLanguageHead local_language{};',
        'LocalLanguageHead local_language{};\n    std::vector<Vec> local_expert_gradient;')
    s=local_head.replace(s,'if(v>0)return 13;','if(v>0)return 14;');p.write_text(s)
    p=directory/'native.cpp';s=p.read_text()
    needle='if(recall_credit[scale]>0)l.recall_content_gradient.emplace_back();'
    s=local_head.replace(s,needle,needle+'\n        if(local_language_weight[scale]>0)l.local_expert_gradient.emplace_back();')
    s=local_head.replace(s,'dc[j]+=local_language_weight[scale]*error*l.local_language.weights[node][j];',
        'l.local_expert_gradient[e.content_id][j]+=local_language_weight[scale]*error*l.local_language.weights[node][j];')
    needle='Vec gated=pre;for(auto& x:gated)x*=c.gate;outer(eg[k].w,gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);'
    s=local_head.replace(s,needle,'''Vec gated=pre;for(auto& x:gated)x*=c.gate;Vec expert_gated=gated;
                if(local_language_weight[scale]>0){for(int i=0;i<D;++i){
                    float extra=l.local_expert_gradient[ci-1][i]*(1-c.content[i]*c.content[i]);
                    eg[k].bias[i]+=extra;expert_gated[i]+=c.gate*extra;}}
                outer(eg[k].w,expert_gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);''')
    needle='l.constituents.clear();l.routing.constituents.clear();'
    s=local_head.replace(s,needle,'l.local_expert_gradient.clear();'+needle)
    needle='for(auto& l:levels){l.recall_credit_state=RecallCreditState{};l.recall_content_gradient.clear();'
    s=local_head.replace(s,needle,needle+'l.local_expert_gradient.clear();')
    s=local_head.replace(s,'version>13','version>14')
    s=local_head.replace(s,'local_extension=version>=13;',
        'local_extension=version>=13,expert_extension=version>=14;')
    s=local_head.replace(s,'return w.out;',
        'if(version>=14)for(const auto& l:m.levels)w.vector(l.local_expert_gradient);\n    return w.out;')
    s=local_head.replace(s,'if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");',
        'if(expert_extension)for(size_t i=0;i<m.levels.size();++i){auto& l=m.levels[i];l.local_expert_gradient.resize(m.training&&m.local_language_weight[i]>0?l.constituents.size():0);r.vector(l.local_expert_gradient);}\n    if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");')
    p.write_text(s)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(p),'-o',str(directory/'diagnostic.so')],check=True)
    subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared',str(directory/'accessors.cpp'),'-o',str(directory.parent/'accessors.so')],check=True)
def install(directory,arm):
    return local_head.install(directory,arm)
if __name__=='__main__':
    screen.Model=local_head.LocalModel;screen.build=build;screen.install=install;screen.main()
