#pragma once
#include "native.hpp"

namespace shoe {
constexpr int ACCESS_SOURCES=4*(RECALL_SLOTS+1+GROUP-1);
struct AccessSources {
    std::array<Vec,ACCESS_SOURCES> x{};
    std::array<int,ACCESS_SOURCES> partial{};
    int count=0;
    void add(const Vec& value,int level,float age,int type,int role){
        if(count>=ACCESS_SOURCES)throw std::logic_error("source bound");
        auto& out=x[count];
        for(int j=0;j<60;++j)out[j]=8*value[j];
        out[60]=float(level)/3;out[61]=std::min(age,16.f)/16;
        out[62]=float(type)/2;out[63]=float(role)/3;
        partial[count++]=type==2;
    }
};

// Reuse only already published neural state; never parse bytes or infer record
// indices. Miller et al. (2016), https://aclanthology.org/D16-1147/, sec. 3:
// addressing and returned content can have separate learned representations.
// Dai et al. (2019), https://aclanthology.org/P19-1285/, sec. 3.2:
// completed context remains accessible across segment boundaries. We reuse
// SHoEbox's bounded records; we do not create a Transformer-XL cache.
inline AccessSources access_sources(const Model& m,int lesion=0){
    AccessSources out;
    for(int l=0;l<4;++l){const auto& level=m.levels[l];const auto& s=level.s;
        if(lesion!=6){
            for(int k=0;k<level.retained.count;++k){const auto& u=level.retained.units[k];
                out.add(u.stable,l,float(level.retained.clock-u.born),0,0);}
            if(s.executions)out.add(s.stable,l,0,1,0);
        }
        if(lesion!=7&&l>0)for(int k=0;k<s.children_count;++k)
            out.add(s.stable_children[k],l-1,0,2,k);
    }
    return out;
}

// Query-time provisional composition: valid child prefixes only. A completed
// lower constituent is already in its parent, so append a provisional child
// only when bytes % child_span != 0. In particular, children_count==0 does NOT
// authorize reading the previous group's stale child array. RetNet's split
// between intra-chunk and recurrent contributions motivates this comparison:
// Sun et al. (2023), https://arxiv.org/abs/2307.08621, sec. 2.1.
inline Vec provisional(const Model& m){
    Vec live=m.levels[0].s.content;uint64_t span=1;
    for(int l=1;l<4;++l){const auto& level=m.levels[l];const auto& s=level.s;
        const bool append=m.bytes%span!=0;const int n=s.children_count+int(append);
        if(!n){live=s.content;span*=GROUP;continue;}
        std::array<Vec,GROUP> children{};std::array<float,GROUP> lengths{};
        for(int k=0;k<s.children_count;++k){children[k]=s.children[k];lengths[k]=s.lengths[k];}
        if(append){children[s.children_count]=live;lengths[s.children_count]=float(m.bytes%span);}
        float total=0;for(float x:lengths)total+=x;
        for(int i=0;i<D;++i){float z=level.p.offset[i];
            for(int k=0;k<n;++k){int j=(i*(2*k+1)+13*k)%D;
                z+=level.p.role[k*D+i]*children[k][j];
                z+=level.p.length[i]*(k+1)*lengths[k]/std::max(1.f,total)*.25f;}
            live[i]=std::tanh(z);
        }
        span*=GROUP;
    }
    return live;
}
} // namespace shoe
