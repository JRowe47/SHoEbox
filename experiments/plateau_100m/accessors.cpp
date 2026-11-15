#include "../../multiscale/native.hpp"
// Offline field access for frozen snapshot controls only.
extern "C" int inspect_means(void* p,float* out){
    auto& m=*static_cast<shoe::Model*>(p);
    for(size_t l=0;l<m.levels.size();++l)std::copy(m.levels[l].routing.mean.begin(),m.levels[l].routing.mean.end(),out+l*shoe::D);
    return 0;
}
extern "C" int replace_means(void* p,const float* values){
    auto& m=*static_cast<shoe::Model*>(p);if(m.training)return -1;
    for(size_t l=0;l<m.levels.size();++l)std::copy(values+l*shoe::D,values+(l+1)*shoe::D,m.levels[l].routing.mean.begin());
    return 0;
}
extern "C" int scale_bias(void* p,int level,float factor){
    auto& m=*static_cast<shoe::Model*>(p);if(m.training)return -1;
    for(size_t l=0;l<m.levels.size();++l)if(level<0||int(l)==level)for(auto& x:m.levels[l].routing.bias)x*=factor;
    return 0;
}
