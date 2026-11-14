#include "native.hpp"
#include <memory>

namespace shoe {
Model::Model(int count,int target,uint32_t seed,bool use_index,bool memory):indexed(use_index),associations(memory){
    if(count<1||count>4||target<50000||target>6000000)throw std::invalid_argument("levels must be 1..4 and parameters 50,000..6,000,000");
    levels.resize(count);
    std::mt19937 rng(seed);std::normal_distribution<float> normal(0,1);
    for(auto& v:embedding)for(auto& x:v)x=.35f*normal(rng);
    for(auto& v:decoder)for(auto& x:v)x=.02f*normal(rng);
    // Level zero fits recurrence/read maps, but no slow composer or projector.
    int fixed=int(sizeof(embedding)/sizeof(float)+sizeof(decoder)/sizeof(float)+decoder_bias.size()+D*D+D*K+(count-1)*(sizeof(Shared)+sizeof(AddressParameters))/sizeof(float));
    int available=(target-fixed)/int(sizeof(Expert)/sizeof(float));
    if(available<count*2)throw std::invalid_argument("parameter budget too small for this topology");
    const float shares[4]{.5f,.25f,.15f,.1f};float total=0;for(int l=0;l<count;++l)total+=shares[l];
    int left=available;
    for(int scale=0;scale<count;++scale){
        auto& l=levels[scale];
        int e=scale==count-1?left:std::min(left-2*(count-1-scale),std::max(2,int(available*shares[scale]/total)));
        left-=e;
        l.experts.resize(e);
        for(auto& expert:l.experts){
            for(auto& x:expert.w)x=.025f*normal(rng);
            for(auto& x:expert.key)x=normal(rng);
            normalize(expert.key);
        }
        for(int i=0;i<D;++i){
            l.p.recurrent[i*D+i]=.45f;
            l.p.gain[i*D+i]=.05f/count;
            l.p.bias[i*D+i]=.35f/count;
            l.p.predict[i*D+i]=.15f;
            for(int j=0;j<K;++j)l.p.read[i*K+j]=.025f*normal(rng);
            for(int k=0;k<GROUP;++k)l.p.role[k*D+i]=.5f;
            l.p.length[i]=.01f*normal(rng);
        }
        if(scale>0){
            for(auto& x:l.address.read)x=.025f*normal(rng);
            for(int i=0;i<TIME_D;++i)l.address.query_clock[i*TIME_D+i]=.25f;
            for(int i=0;i<D;++i)l.address.reconstruct[i*D+i]=.15f;
        }
        rebuild_index(l);
    }
}
size_t Model::parameter_count()const{
    size_t n=sizeof(embedding)/sizeof(float)+sizeof(decoder)/sizeof(float)+decoder_bias.size();
    for(size_t i=0;i<levels.size();++i)n+=(i?(sizeof(Shared)+sizeof(AddressParameters))/sizeof(float):D*D+D*K)+levels[i].experts.size()*sizeof(Expert)/sizeof(float);
    return n;
}
void Model::rebuild_index(Level& l){
    l.index.clear();if(!indexed||l.experts.size()<=32)return;
    std::vector<Vec> current;current.reserve(l.experts.size());for(const auto& e:l.experts)current.push_back(e.key);
    while(current.size()>16){
        std::vector<Vec> parent((current.size()+15)/16);
        for(size_t i=0;i<current.size();++i)for(int j=0;j<D;++j)parent[i/16][j]+=current[i][j];
        for(auto& v:parent)normalize(v);
        l.index.push_back(parent);current=std::move(parent);
    }
}
void Model::route(Level& l,Event& e){
    int lo=0,hi=int(l.experts.size());
    if(!l.index.empty()){
        int selected=0;
        for(int depth=int(l.index.size())-1;depth>=0;--depth){
            const auto& layer=l.index[depth];
            lo=depth==int(l.index.size())-1?0:selected*16;
            hi=std::min(int(layer.size()),lo+16);float best=-1e30f;
            for(int k=lo;k<hi;++k){++route_comparisons;float score=dot(layer[k].data(),e.input.data());if(score>best){best=score;selected=k;}}
        }
        lo=selected*16;hi=std::min(int(l.experts.size()),lo+16);
    }
    float best=-1e30f,second=-1e30f;int winner=lo,runner=lo;
    for(int k=lo;k<hi;++k){++route_comparisons;float score=dot(l.experts[k].key.data(),e.input.data());
        if(score>best){second=best;runner=winner;best=score;winner=k;}
        else if(score>second){second=score;runner=k;}
    }
    e.expert=winner;e.runner=runner;
    if(hi-lo>1){float a=std::tanh((best-second)/.5f);e.gate=1+.25f*a;e.gate_derivative=.5f*(1-a*a);}
}
void Model::update_index(Level& l,int expert){
    int node=expert/16;
    for(size_t depth=0;depth<l.index.size();++depth){
        auto& v=l.index[depth][node];v={};
        int end=std::min((node+1)*16,int(depth?l.index[depth-1].size():l.experts.size()));
        for(int k=node*16;k<end;++k){const Vec& child=depth?l.index[depth-1][k]:l.experts[k].key;
            for(int j=0;j<D;++j)v[j]+=child[j];}
        normalize(v);node/=16;
    }
}

void Model::forward_event(int scale,const std::array<Vec,GROUP>& child,const std::array<float,GROUP>& lengths,int byte){
    auto& l=levels[scale];auto& s=l.s;
    // Retirement occurs before the next event. No output-lifetime gradient can
    // arrive after this update. Cross-block native forecasts are truncated.
    if(training&&s.count==BLOCK)flush(scale);
    int slot=training?s.count:0;auto& e=s.events[slot];e=Event{};e.byte=byte;e.before=s.state;
    e.reconstruction_role=int(s.executions%GROUP);
    if(scale==0){e.input=embedding[byte];e.span=1;s.stable=stable_bytes()[byte];}
    else {
        s.stable=stable_group(s.stable_children);
        e.children=child;e.lengths=lengths;e.span=0;for(float n:lengths)e.span+=n;
        for(int i=0;i<D;++i){float z=l.p.offset[i];
            for(int k=0;k<GROUP;++k){int j=(i*(2*k+1)+13*k)%D;
                z+=l.p.role[k*D+i]*child[k][j];
                z+=l.p.length[i]*(k+1)*lengths[k]/std::max(e.span,1.f)*.25f;
            }e.input[i]=std::tanh(z);
        }
    }
    route(l,e);const auto& expert=l.experts[e.expert];
    mv(expert.w,e.input,e.residual);
    for(int i=0;i<D;++i)e.content[i]=std::tanh(e.input[i]+expert.bias[i]+e.gate*e.residual[i]);
    context_forward(e,s.memory);
    if(scale>0)address_forward(s.address_events[slot],e.content,e.before,s.stable,s.addressed,l.address,memory_mode);
    Vec recurrent{};mv(l.p.recurrent,e.before,recurrent);
    for(int i=0;i<D;++i){float a=scale?dot(l.address.read.data()+i*ADDRESS_V,s.address_events[slot].read.data(),ADDRESS_V):0;
        e.state[i]=std::tanh(e.content[i]+recurrent[i]+dot(l.p.read.data()+i*K,e.read.data(),K)+a);}
    if(training&&scale>0){
        // Same-scale contrastive next-unit objective. Only older published
        // contents enter negatives. Neither target nor negatives receive VJPs.
        if(s.count>0&&s.negative_count>0){
            auto& prev=s.events[s.count-1];Vec target=e.content;normalize(target);
            std::array<float,NEG+1> scores{},probs{};std::array<Vec,NEG+1> candidates{};
            candidates[0]=target;int n=1;
            for(int k=0;k<s.negative_count;++k)if(dot(target.data(),s.negatives[k].data())<.9999f)candidates[n++]=s.negatives[k];
            if(n>1){
                float mx=-1e30f;for(int k=0;k<n;++k){scores[k]=dot(prev.unit.data(),candidates[k].data())/.2f;mx=std::max(mx,scores[k]);}
                float denom=0;for(int k=0;k<n;++k){probs[k]=std::exp(scores[k]-mx);denom+=probs[k];}
                s.native_loss+=std::log(denom)+mx-scores[0];++s.native_targets;
                Vec dp{};for(int k=0;k<n;++k){float g=(probs[k]/denom-(k==0))/.2f;for(int j=0;j<D;++j)dp[j]+=g*candidates[k][j];}
                float radial=dot(dp.data(),prev.unit.data());
                for(int j=0;j<D;++j)prev.native[j]=(dp[j]-radial*prev.unit[j])/prev.unit_norm;
                prev.has_target=1;
            }
        }
        mv(l.p.predict,e.state,e.forecast);e.unit=e.forecast;
        e.unit_norm=std::sqrt(energy(e.forecast)+1e-8f);for(auto& x:e.unit)x/=e.unit_norm;
        Vec detached=e.content;normalize(detached);s.negatives[s.negative_cursor]=detached;
        s.negative_cursor=(s.negative_cursor+1)%NEG;s.negative_count=std::min(NEG,s.negative_count+1);
    }
    s.state=e.state;s.content=e.content;++s.executions;
    if(training)++s.count;
    if(scale>0){mv(l.p.gain,s.state,s.gain);mv(l.p.bias,s.state,s.bias);}
    if(scale+1<int(levels.size())){
        auto& parent=levels[scale+1].s;int k=parent.children_count++;
        parent.children[k]=e.content;parent.lengths[k]=e.span;
        parent.stable_children[k]=s.stable;
        if(parent.children_count==GROUP){parent.children_count=0;forward_event(scale+1,parent.children,parent.lengths,0);}
    }
}

void Model::flush(int scale){
    auto& l=levels[scale];auto& s=l.s;if(!s.count)return;
    Shared g{};AddressParameters ag{};std::array<Expert,BLOCK> eg{};std::array<int,BLOCK> selected{};int unique=0;
    std::array<Vec,BLOCK> embed_grad{};std::array<Vec,BLOCK> key_delta{},runner_delta{};
    int uses=0,targets=0;for(int t=0;t<s.count;++t){uses+=s.events[t].uses;targets+=s.events[t].has_target;}
    // Average task loss over the predictions actually served by this block;
    // native contrastive loss has its own event count and declared weight.
    float task_scale=1.f/std::max(1,uses),native_scale=aux_weight/std::max(1,targets);
    Vec future{};Memory memory_grad{};AddressMatrix address_grad{};
    for(int t=s.count-1;t>=0;--t){
        const auto& e=s.events[t];int k=0;for(;k<unique;++k)if(selected[k]==e.expert)break;
        if(k==unique)selected[unique++]=e.expert;
        Vec dh=future,dgain{},dbias{},dp{};
        for(int i=0;i<D;++i){dh[i]+=task_scale*e.task[i];dgain[i]=task_scale*e.gain_grad[i];dbias[i]=task_scale*e.bias_grad[i];dp[i]=native_scale*e.native[i];}
        if(scale>0){vjp(l.p.gain,dgain,dh);vjp(l.p.bias,dbias,dh);outer(g.gain,dgain,e.state);outer(g.bias,dbias,e.state);
            vjp(l.p.predict,dp,dh);outer(g.predict,dp,e.state);}
        Vec dz{},dc{};Small dr{};AddressValue ar{};future={};
        for(int i=0;i<D;++i){dz[i]=dh[i]*(1-e.state[i]*e.state[i]);dc[i]=dz[i];
            for(int j=0;j<K;++j){g.read[i*K+j]+=dz[i]*e.read[j];dr[j]+=l.p.read[i*K+j]*dz[i];}
            if(scale>0)for(int j=0;j<ADDRESS_V;++j){ag.read[i*ADDRESS_V+j]+=dz[i]*s.address_events[t].read[j];ar[j]+=l.address.read[i*ADDRESS_V+j]*dz[i];}}
        outer(g.recurrent,dz,e.before);vjp(l.p.recurrent,dz,future);
        context_backward(e,dr,memory_grad,dc);
        if(scale>0){
            address_backward(s.address_events[t],e.content,e.before,l.address,ar,address_grad,dc,future,ag);
            if(reconstruction_weight>0){s.reconstruction_loss+=reconstruct(e.content,e.children[e.reconstruction_role],e.reconstruction_role,
                    l.address,reconstruction_weight/s.count,dc,ag);++s.reconstruction_targets;}
        }
        const auto& expert=l.experts[e.expert];Vec pre{},du{};float dgate=0;
        for(int i=0;i<D;++i){pre[i]=dc[i]*(1-e.content[i]*e.content[i]);eg[k].bias[i]+=pre[i];du[i]=pre[i];dgate+=pre[i]*e.residual[i];}
        Vec gated=pre;for(auto& x:gated)x*=e.gate;outer(eg[k].w,gated,e.input);vjp(expert.w,gated,du);
        float margin=dgate*e.gate_derivative;
        for(int i=0;i<D;++i){
            key_delta[t][i]=margin*e.input[i];runner_delta[t][i]=-margin*e.input[i];
            du[i]+=margin*(expert.key[i]-l.experts[e.runner].key[i]);
        }
        if(scale==0)embed_grad[t]=du;
        else for(int i=0;i<D;++i){float a=du[i]*(1-e.input[i]*e.input[i]);g.offset[i]+=a;
            for(int pos=0;pos<GROUP;++pos){int j=(i*(2*pos+1)+13*pos)%D;
                g.role[pos*D+i]+=a*e.children[pos][j];
                g.length[i]+=a*(pos+1)*e.lengths[pos]/std::max(1.f,e.span)*.25f;}
        }
    }
    step(l.p.recurrent,g.recurrent,learning_rate);step(l.p.read,g.read,learning_rate);
    if(scale>0){step(l.p.gain,g.gain,learning_rate);step(l.p.bias,g.bias,learning_rate);step(l.p.predict,g.predict,learning_rate);
        step(l.p.role,g.role,learning_rate);step(l.p.length,g.length,learning_rate);step(l.p.offset,g.offset,learning_rate);
        step(l.address.query_content,ag.query_content,learning_rate);step(l.address.query_state,ag.query_state,learning_rate);
        step(l.address.query_clock,ag.query_clock,learning_rate);step(l.address.read,ag.read,learning_rate);
        step(l.address.reconstruct,ag.reconstruct,learning_rate);step(l.address.reconstruct_bias,ag.reconstruct_bias,learning_rate);}
    for(int k=0;k<unique;++k){step(l.experts[selected[k]].w,eg[k].w,learning_rate);step(l.experts[selected[k]].bias,eg[k].bias,learning_rate);}
    for(int t=0;t<s.count;++t){const auto& e=s.events[t];
        if(scale==0)step(embedding[e.byte],embed_grad[t],learning_rate);
        // Preserve SHoE's competitive prototype attraction plus the derivative
        // of its task-informed top-two margin gate. Hard top-1 is not softened.
        auto& key=l.experts[e.expert].key;auto& runner=l.experts[e.runner].key;
        step(key,key_delta[t],learning_rate);if(e.runner!=e.expert)step(runner,runner_delta[t],learning_rate);
        Vec cue=e.input;normalize(cue);for(int i=0;i<D;++i)key[i]+=route_rate*(cue[i]-key[i]);
        normalize(key);if(e.runner!=e.expert)normalize(runner);
    }
    for(int t=0;t<s.count;++t){update_index(l,s.events[t].expert);update_index(l,s.events[t].runner);}
    s.count=0;++s.updates;
}
void Model::refresh(){
    cached_gain={};cached_bias={};
    for(size_t l=1;l<levels.size();++l){if(int(l)==mask)continue;
        const auto& s=levels[l].s;
        for(int i=0;i<D;++i){int j=int(l)==shuffle?(i+23)%D:i;cached_gain[i]+=s.gain[j];cached_bias[i]+=s.bias[j];}
    }
    if(associations)for(int i=0;i<D;++i)cached_bias[i]+=.2f*recalled[i];
}
void Model::observe(uint8_t byte){
    forward_event(0,{}, {},byte);++bytes;
    payload[payload_count++]=byte;
    if(payload_count==GROUP){
        payload_count=0;
        if(associations&&levels.size()>1){
            // Query before write; the successor value is admitted only once
            // its complete observed payload exists. Recalled records detach.
            Vec query=levels[1].s.content;
            for(size_t l=2;l<levels.size();++l)for(int i=0;i<D;++i)query[i]+=.1f*levels[l].s.state[i];
            recalled=store.query(query);
            if(memory_writes&&have_query)store.put(AssociationStore::bucket(previous_query),previous_query,levels[1].s.content,payload.data(),GROUP);
            previous_query=query;have_query=true;
        }
    }
    // Projections themselves are cached per slow event. Summation is 3x64.
    refresh();for(int i=0;i<D;++i)features[i]=std::tanh(levels[0].s.state[i]*(1+cached_gain[i])+cached_bias[i]);
}
float Model::teach(uint8_t target){
    int node=0;float loss=0;Vec feature_grad{};
    for(int depth=0;depth<8;++depth){
        int bit=(target>>(7-depth))&1;float z=decoder_bias[node]+dot(decoder[node].data(),features.data());
        loss+=softplus(z)-bit*z;
        if(training){float error=sigmoid(z)-bit;
            // Read the exact weights which produced z before changing them.
            for(int j=0;j<D;++j)feature_grad[j]+=error*decoder[node][j];
            float lr=head_rate/(1+energy(features));
            for(int j=0;j<D;++j)decoder[node][j]-=lr*error*features[j];
            decoder_bias[node]-=head_rate*.05f*error;
        }
        node=2*node+1+bit;
    }
    if(training){auto& fast=levels[0].s.events[levels[0].s.count-1];
        for(int i=0;i<D;++i){float g=feature_grad[i]*(1-features[i]*features[i]);fast.task[i]+=g*(1+cached_gain[i]);
            for(size_t l=1;l<levels.size();++l){auto& s=levels[l].s;if(!s.count)continue;
                s.events[s.count-1].gain_grad[i]+=g*levels[0].s.state[i];s.events[s.count-1].bias_grad[i]+=g;}
        }
        for(auto& l:levels)if(l.s.count)++l.s.events[l.s.count-1].uses;
    }
    return loss;
}
void Model::log_probs(float* out)const{
    std::array<float,511> p{};
    for(int n=0;n<255;++n){float z=decoder_bias[n]+dot(decoder[n].data(),features.data());
        p[2*n+1]=p[n]-softplus(z);p[2*n+2]=p[n]-softplus(-z);}
    std::copy(p.begin()+255,p.end(),out);
}
double Model::run(const uint8_t* x,const uint8_t* y,size_t n){
    if(!n)throw std::invalid_argument("at least one prediction is required");
    if(training&&(mask>=0||shuffle>=0||memory_mode))throw std::logic_error("interventions require frozen inference");
    double loss=0;for(size_t t=0;t<n;++t){observe(x[t]);loss+=teach(y[t]);}return loss/n;
}
void Model::finish(){if(training)for(size_t l=0;l<levels.size();++l)flush(int(l));}
void Model::reset(bool clear_store){
    finish();for(auto& l:levels)l.s=Stream{};
    recalled={};previous_query={};cached_gain={};cached_bias={};features={};payload={};payload_count=0;have_query=false;bytes=0;route_comparisons=0;
    if(clear_store)store.clear();
}
void Model::inference(bool cold){
    // Snapshot fitted weights as they stand. Do not silently train/drain the
    // source to evaluate; pending derivatives are deliberately discarded.
    training=false;memory_writes=false;
    for(auto& l:levels){l.s.count=0;l.s.events={};l.s.address_events={};l.s.negatives={};l.s.negative_count=0;l.s.negative_cursor=0;}
    if(cold)reset(false);
}

std::string save(const Model& m){
    Writer w;uint64_t magic=0x31534d454f4853ULL;w.pod(magic);uint32_t version=3;w.pod(version);
    uint32_t abi=sizeof(Event)^sizeof(Stream)^sizeof(AssociationStore);w.pod(abi);
    int count=int(m.levels.size()),target=int(m.parameter_count());w.pod(count);w.pod(target);w.pod(m.indexed);w.pod(m.associations);
    w.pod(m.training);w.pod(m.memory_writes);w.pod(m.have_query);
    w.pod(m.embedding);w.pod(m.decoder);w.pod(m.decoder_bias);w.pod(m.store);
    w.pod(m.recalled);w.pod(m.previous_query);w.pod(m.cached_gain);w.pod(m.cached_bias);w.pod(m.features);
    w.pod(m.payload);w.pod(m.payload_count);w.pod(m.bytes);w.pod(m.route_comparisons);
    w.pod(m.learning_rate);w.pod(m.head_rate);w.pod(m.aux_weight);w.pod(m.route_rate);w.pod(m.mask);w.pod(m.shuffle);
    w.pod(m.reconstruction_weight);w.pod(m.memory_mode);
    for(const auto& l:m.levels){w.vector(l.experts);w.pod(l.p);w.pod(l.address);w.pod(l.s);}return w.out;
}
Model load(const char* p,size_t n){
    Reader r{p,n};uint64_t magic;uint32_t version,abi;r.pod(magic);r.pod(version);r.pod(abi);
    if(magic!=0x31534d454f4853ULL||version!=3||abi!=(sizeof(Event)^sizeof(Stream)^sizeof(AssociationStore)))throw std::runtime_error("incompatible multiscale checkpoint");
    int count,target;bool index,assoc;r.pod(count);r.pod(target);r.pod(index);r.pod(assoc);
    Model m(count,target,7,index,assoc);r.pod(m.training);r.pod(m.memory_writes);r.pod(m.have_query);
    r.pod(m.embedding);r.pod(m.decoder);r.pod(m.decoder_bias);r.pod(m.store);
    r.pod(m.recalled);r.pod(m.previous_query);r.pod(m.cached_gain);r.pod(m.cached_bias);r.pod(m.features);
    r.pod(m.payload);r.pod(m.payload_count);r.pod(m.bytes);r.pod(m.route_comparisons);
    r.pod(m.learning_rate);r.pod(m.head_rate);r.pod(m.aux_weight);r.pod(m.route_rate);r.pod(m.mask);r.pod(m.shuffle);
    r.pod(m.reconstruction_weight);r.pod(m.memory_mode);
    if(!std::isfinite(m.reconstruction_weight)||m.reconstruction_weight<0||m.memory_mode<0||m.memory_mode>3||(m.training&&m.memory_mode))throw std::runtime_error("invalid addressed-memory configuration");
    if(m.payload_count<0||m.payload_count>=GROUP||m.mask>=count||m.shuffle>=count)throw std::runtime_error("invalid checkpoint counters");
    for(size_t level=0;level<m.levels.size();++level){auto& l=m.levels[level];r.vector(l.experts);r.pod(l.p);r.pod(l.address);r.pod(l.s);
        const auto& s=l.s;if(s.count<0||s.count>BLOCK||s.children_count<0||s.children_count>=GROUP||s.negative_count<0||s.negative_count>NEG||s.negative_cursor<0||s.negative_cursor>=NEG)throw std::runtime_error("invalid event ring");
        if(level&&s.addressed.clock.ticks!=s.executions)throw std::runtime_error("invalid native clock");
        for(double v:s.addressed.clock.phase)if(!std::isfinite(v))throw std::runtime_error("invalid clock phase");
        for(int t=0;t<s.count;++t)if(s.events[t].expert<0||s.events[t].expert>=int(l.experts.size())||s.events[t].runner<0||s.events[t].runner>=int(l.experts.size())||s.events[t].byte<0||s.events[t].byte>=256||s.events[t].reconstruction_role<0||s.events[t].reconstruction_role>=GROUP)throw std::runtime_error("invalid event route");
        m.rebuild_index(l);
    }
    for(const auto& rec:m.store.records)if(rec.key_size>KEY_BYTES||rec.payload_size>PAYLOAD)throw std::runtime_error("invalid association record");
    if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");
    return m;
}
} // namespace shoe

namespace {thread_local std::string error;template<class F> int guard(F f){try{f();error.clear();return 0;}catch(const std::exception& e){error=e.what();return -1;}}}
extern "C" {
const char* ms_error(){return error.c_str();}
void* ms_create(int levels,int target,uint32_t seed,int indexed,int associations){
    void* p=nullptr;guard([&]{p=new shoe::Model(levels,target,seed,indexed,associations);});return p;
}
void ms_destroy(void* p){delete static_cast<shoe::Model*>(p);}
void* ms_clone(void* p,int inference,int cold){void* out=nullptr;guard([&]{auto q=std::make_unique<shoe::Model>(*static_cast<shoe::Model*>(p));if(inference)q->inference(cold);out=q.release();});return out;}
int ms_run(void* p,const uint8_t* x,const uint8_t* y,size_t n,double* loss){return guard([&]{*loss=static_cast<shoe::Model*>(p)->run(x,y,n);});}
int ms_observe(void* p,uint8_t byte,float* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);if(m.training)throw std::logic_error("observe requires inference; use run for training");m.observe(byte);m.log_probs(out);});}
int ms_logits(void* p,float* out){return guard([&]{static_cast<shoe::Model*>(p)->log_probs(out);});}
int ms_reset(void* p,int clear){return guard([&]{static_cast<shoe::Model*>(p)->reset(clear);});}
int ms_finish(void* p){return guard([&]{static_cast<shoe::Model*>(p)->finish();});}
int ms_configure(void* p,float lr,float head,float aux,float route){return guard([&]{
    if(!std::isfinite(lr)||!std::isfinite(head)||!std::isfinite(aux)||!std::isfinite(route)||lr<0||head<0||aux<0||route<0)throw std::invalid_argument("rates must be finite and nonnegative");
    auto& m=*static_cast<shoe::Model*>(p);m.learning_rate=lr;m.head_rate=head;m.aux_weight=aux;m.route_rate=route;
});}
int ms_reconstruction(void* p,float weight){return guard([&]{
    if(!std::isfinite(weight)||weight<0)throw std::invalid_argument("reconstruction weight must be finite and nonnegative");
    static_cast<shoe::Model*>(p)->reconstruction_weight=weight;
});}
int ms_memory_intervene(void* p,int mode){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||mode<0||mode>3)throw std::invalid_argument("memory interventions require frozen inference and a valid mode");m.memory_mode=mode;
});}
int ms_intervene(void* p,int mask,int shuffle){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||mask>=int(m.levels.size())||shuffle>=int(m.levels.size())||mask< -1||shuffle< -1||mask==0||shuffle==0)throw std::invalid_argument("interventions require inference and a valid slow scale");
    m.mask=mask;m.shuffle=shuffle;
});}
int ms_stats(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+48,0.);
    out[0]=m.parameter_count();out[1]=m.bytes;out[2]=m.training;out[3]=m.route_comparisons;out[4]=m.store.queries;out[5]=m.store.candidates;out[6]=m.store.writes;out[7]=m.levels.size();
    for(size_t i=0;i<m.levels.size();++i){auto& l=m.levels[i];out[8+i]=l.experts.size();out[12+i]=l.s.executions;out[16+i]=l.s.updates;out[20+i]=l.s.count;out[24+i]=l.s.native_targets;out[28+i]=l.s.native_loss;out[32+i]=shoe::energy(l.s.state);out[36+i]=l.s.children_count;out[40+i]=l.s.reconstruction_targets;out[44+i]=l.s.reconstruction_loss;}
});}
int ms_address_stats(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+24,0.);
    for(size_t i=1;i<m.levels.size();++i){const auto& l=m.levels[i];out[i]=l.s.addressed.clock.ticks;out[4+i]=shoe::ADDRESS_K;out[8+i]=shoe::ADDRESS_V;
        for(float x:l.s.addressed.memory)out[12+i]+=double(x)*x;
        for(float x:l.address.query_content)out[16+i]+=double(x)*x;}
    out[20]=m.reconstruction_weight;out[21]=m.memory_mode;
});}
size_t ms_save_size(void* p){size_t n=0;guard([&]{n=shoe::save(*static_cast<shoe::Model*>(p)).size();});return n;}
int ms_save(void* p,char* out,size_t n){return guard([&]{auto data=shoe::save(*static_cast<shoe::Model*>(p));if(n!=data.size())throw std::runtime_error("checkpoint buffer size mismatch");std::memcpy(out,data.data(),n);});}
void* ms_load(const char* p,size_t n){void* out=nullptr;guard([&]{out=new shoe::Model(shoe::load(p,n));});return out;}
int ms_bind(void* p,const uint8_t* key,size_t kn,const uint8_t* value,size_t vn,uint64_t* handle){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);*handle=m.store.bind(key,kn,value,vn);});}
int ms_lookup(void* p,const uint8_t* key,size_t kn,uint64_t* handle){return guard([&]{*handle=static_cast<shoe::Model*>(p)->store.lookup(key,kn);});}
int ms_payload(void* p,uint64_t handle,uint8_t* value,size_t* n){return guard([&]{auto* r=static_cast<shoe::Model*>(p)->store.resolve(handle);if(!r)throw std::runtime_error("missing or evicted payload handle");*n=r->payload_size;std::memcpy(value,r->payload.data(),*n);});}
}
