// Research helpers only. The public runtime and checkpoint ABI are unchanged.
#include "../../multiscale/native.cpp"

extern "C" {
int mad_train(void* p,const uint8_t* x,const uint8_t* y,const uint8_t* mask,
              size_t n,double* loss,size_t* targets){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(!m.training||!n)throw std::logic_error("nonempty training stream required");
    *loss=0;*targets=0;
    for(size_t t=0;t<n;++t){m.observe(x[t]);if(mask[t]){*loss+=m.teach(y[t]);++*targets;}}
});}
int mad_observe(void* p,uint8_t byte){return guard([&]{
    static_cast<shoe::Model*>(p)->observe(byte);
});}
int mad_features(void* p,float* out){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(m.levels.size()!=4||!m.bytes||m.bytes%64)
        throw std::logic_error("compression requires a completed L3 constituent");
    std::copy(m.levels[3].s.state.begin(),m.levels[3].s.state.end(),out);
});}
int mad_credit(void* p,const float* dx){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(!m.training||m.levels.size()!=4||!m.bytes||m.bytes%64||m.language_credit[3]!=1)
        throw std::logic_error("compression credit requires a live, unit-gain L3 boundary");
    auto& s=m.levels[3].s;
    if(!s.count)throw std::logic_error("compression event already retired");
    for(int i=0;i<shoe::D;++i)if(!std::isfinite(dx[i]))throw std::invalid_argument("nonfinite state credit");
    // flush() normalizes native task credit over the epoch's byte uses.
    // Undo that denominator for this separately mean-normalized task loss.
    // Finish immediately: no future byte can change this denominator.
    int uses=0;for(int t=0;t<s.count;++t)uses+=s.events[t].uses;
    for(int i=0;i<shoe::D;++i)s.events[s.count-1].task[i]+=std::max(1,uses)*dx[i];
    m.finish();
});}
int mad_score(void* p,const uint8_t* data,const uint8_t* mask,size_t n,
              int low,int classes,int rollout,double* out){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||n<2||low<0||classes<2||low+classes>256||rollout<0||rollout>2)
        throw std::invalid_argument("invalid frozen scoring call");
    for(size_t t=1;t<n;++t)if(mask[t]&&(data[t]<low||data[t]>=low+classes))
        throw std::invalid_argument("target outside declared alphabet");
    std::fill_n(out,7,0.);out[4]=out[5]=1;
    m.observe(data[0]);
    for(size_t t=1;t<n;++t){
        int observed=data[t];
        if(mask[t]){
            float lp[256];m.log_probs(lp);
            int full=int(std::max_element(lp,lp+256)-lp);
            int restricted=int(std::max_element(lp+low,lp+low+classes)-lp);
            float mx=lp[restricted];double mass=0;
            for(int k=low;k<low+classes;++k)mass+=std::exp(double(lp[k]-mx));
            ++out[0];out[1]-=lp[data[t]];out[2]-=lp[data[t]]-mx-std::log(mass);
            out[3]+=restricted==data[t];out[4]*=restricted==data[t];out[5]*=full==data[t];out[6]+=full==data[t];
            if(rollout)observed=rollout==1?restricted:full;
        }
        if(t+1<n)m.observe(uint8_t(observed));
    }
});}
int mad_permute(void* p,int scale,uint32_t seed,int* count){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||scale<1||scale>=int(m.levels.size()))throw std::invalid_argument("frozen upper scale required");
    auto& l=m.levels[scale];std::vector<int> ids;
    for(size_t k=0;k<l.experts.size();++k)if(l.routing.selections[k])ids.push_back(int(k));
    *count=ids.size();if(ids.size()<2)return;
    std::mt19937 rng(seed);std::shuffle(ids.begin(),ids.end(),rng);
    auto saved=l.experts;
    for(size_t j=0;j<ids.size();++j){auto& dst=l.experts[ids[j]];const auto& src=saved[ids[(j+1)%ids.size()]];dst.w=src.w;dst.bias=src.bias;}
});}
int mad_inventory(void* p,double* out){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);std::fill_n(out,10,0.);
    out[0]=m.parameter_count();out[1]=m.levels.size();out[2]=shoe::D;
    for(auto& l:m.levels){
        out[3]+=sizeof(l.s.state);out[4]+=sizeof(l.s.memory);
        out[5]+=sizeof(l.s.addressed.memory);out[6]+=sizeof(l.retained);
        out[7]+=sizeof(l.s.children)+sizeof(l.s.stable_children);
        out[8]+=l.loops.traces.capacity()*sizeof(shoe::LoopTrace);
    }
    out[9]=shoe::RECALL_SLOTS;
});}
}
