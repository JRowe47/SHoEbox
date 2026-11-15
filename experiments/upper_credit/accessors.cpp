#include "../../multiscale/native.hpp"
// Offline frozen interventions; never called in the training byte loop.
extern "C" int upper_rotate(void* p,int scale,const int* ids,int n,int shift){
    auto& m=*static_cast<shoe::Model*>(p);if(m.training||scale<0||scale>=int(m.levels.size())||n<2)return -1;
    auto& l=m.levels[scale];auto source=l.experts;
    for(int i=0;i<n;++i)if(ids[i]<0||ids[i]>=int(source.size()))return -1;
    for(int i=0;i<n;++i){auto& dst=l.experts[ids[i]];const auto& src=source[ids[(i+shift)%n]];dst.w=src.w;dst.bias=src.bias;}
    return 0;
}
extern "C" int upper_ablate(void* p,int scale,int kind){
    auto& m=*static_cast<shoe::Model*>(p);if(m.training||scale<0||scale>=int(m.levels.size()))return -1;
    auto& l=m.levels[scale];
    if(kind==0){l.p.gain={};l.p.bias={};}
    if(kind==1)l.retrieval.read={};
    if(kind==2||kind==3)for(int n=0;n<255;++n)for(int j=0;j<shoe::RECALL_K;++j)
        l.retrieval.read[n*shoe::RECALL_V+j+(kind==3?shoe::RECALL_K:0)]=0;
    return 0;
}
