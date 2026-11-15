#include "native.hpp"
#include <memory>

namespace shoe {
Model::Model(int count,int target,uint32_t seed,bool use_index,bool memory):indexed(use_index),associations(memory){
    if(count<1||count>4||target<50000||target>100000000)throw std::invalid_argument("levels must be 1..4 and parameters 50,000..100,000,000");
    levels.resize(count);
    for(int scale=1;scale<count;++scale)epoch_limit*=GROUP;
    std::mt19937 rng(seed);std::normal_distribution<float> normal(0,1);
    for(auto& v:embedding)for(auto& x:v)x=.35f*normal(rng);
    for(auto& v:decoder)for(auto& x:v)x=.02f*normal(rng);
    // Level zero fits recurrence/read maps, but no slow composer or projector.
    int fixed=int(sizeof(embedding)/sizeof(float)+sizeof(decoder)/sizeof(float)+decoder_bias.size()+D*D+D*K+((count-1)*(sizeof(Shared)+sizeof(AddressParameters))+count*sizeof(RecallParameters))/sizeof(float));
    int available=(target-fixed)/int(sizeof(Expert)/sizeof(float));
    if(available<count*2)throw std::invalid_argument("parameter budget too small for this topology");
    const float shares[4]{.5f,.25f,.15f,.1f};float total=0;for(int l=0;l<count;++l)total+=shares[l];
    int left=available;
    for(int scale=0;scale<count;++scale){
        auto& l=levels[scale];
        int span=1;for(int i=0;i<scale;++i)span*=GROUP;
        l.constituents.reserve(epoch_limit/span+1);
        int e=scale==count-1?left:std::min(left-2*(count-1-scale),std::max(2,int(available*shares[scale]/total)));
        left-=e;
        l.experts.resize(e);l.lr_visits.resize(e);l.laces.initialize(e);l.routing.initialize(e);
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
        for(auto& x:l.retrieval.query)x=.02f*normal(rng);
        for(auto& x:l.retrieval.read)x=.02f*normal(rng);
        rebuild_index(l);
    }
}
size_t Model::parameter_count()const{
    size_t n=sizeof(embedding)/sizeof(float)+sizeof(decoder)/sizeof(float)+decoder_bias.size();
    for(size_t i=0;i<levels.size();++i)n+=(i?(sizeof(Shared)+sizeof(AddressParameters))/sizeof(float):D*D+D*K)+sizeof(RecallParameters)/sizeof(float)+levels[i].experts.size()*sizeof(Expert)/sizeof(float);
    if(routing_arm==3)n+=(levels.size()-1)*D*D;
    return n;
}
void Model::rebuild_index(Level& l){
    l.routing_keys.clear();
    if(l.experts.size()>=256){
        l.routing_keys.resize(((l.experts.size()+15)/16)*D*16);
        for(size_t k=0;k<l.experts.size();++k)
            for(int j=0;j<D;++j)l.routing_keys[(k/16)*D*16+j*16+k%16]=l.experts[k].key[j];
    }
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
    if(!l.index.empty()&&!(semantic_laces&&l.laces.allocations)){
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
    for(int k=lo;k<hi;++k){if(semantic_laces&&l.laces.reserved[k])continue;
        ++route_comparisons;float score=dot(l.experts[k].key.data(),e.input.data());
        if(score>best){second=best;runner=winner;best=score;winner=k;}
        else if(score>second){second=score;runner=k;}
    }
    e.expert=winner;e.runner=runner;
    if(hi-lo>1){float a=std::tanh((best-second)/.5f);e.gate=1+.25f*a;e.gate_derivative=.5f*(1-a*a);}
}
void Model::update_index(Level& l,int expert){
    if(!l.routing_keys.empty())
        for(int j=0;j<D;++j)l.routing_keys[(expert/16)*D*16+j*16+expert%16]=l.experts[expert].key[j];
    int node=expert/16;
    for(size_t depth=0;depth<l.index.size();++depth){
        auto& v=l.index[depth][node];v={};
        int end=std::min((node+1)*16,int(depth?l.index[depth-1].size():l.experts.size()));
        for(int k=node*16;k<end;++k){const Vec& child=depth?l.index[depth-1][k]:l.experts[k].key;
            for(int j=0;j<D;++j)v[j]+=child[j];}
        normalize(v);node/=16;
    }
}

#include "semantic_laces_impl.hpp"
#include "routing_geometry_impl.hpp"

void Model::forward_event(int scale,const std::array<Vec,GROUP>& child,const std::array<float,GROUP>& lengths,int byte){
    auto& l=levels[scale];auto& s=l.s;
    // Retirement occurs before the next event. No output-lifetime gradient can
    // arrive after this update. Cross-block native forecasts are truncated.
    if(training&&routing_arm==3&&scale>0)routing_target(scale,routing_signature(stable_group(s.stable_children)));
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
    if(scale==0){
        // Canonical byte content is independent of context. All fast experts
        // now model a transition conditioned on both content and prior state.
        for(int i=0;i<D;++i){e.content[i]=std::tanh(embedding[byte][i]);
            e.input[i]=std::tanh(e.content[i]+.5f*e.before[i]);}
    }
    if(routing_arm)routing_prepare(scale,e,slot);else route(l,e);
    if(semantic_laces&&scale>0)lace_route(scale,e,slot);
    if(swap_level==scale)e.expert=(e.expert+1)%int(l.experts.size());
    const auto& expert=l.experts[e.expert];
    mv(expert.w,routing_arm?l.routing.events[slot].expert_input:e.input,e.residual);
    if(scale)for(int i=0;i<D;++i)e.content[i]=std::tanh(e.input[i]+expert.bias[i]+e.gate*e.residual[i]);
    if(training){
        e.content_id=int(l.constituents.size());Constituent c;
        c.input=e.input;c.content=e.content;c.residual=e.residual;c.children=e.children;c.lengths=e.lengths;
        c.child_ids=s.child_ids;c.gate=e.gate;c.gate_derivative=e.gate_derivative;c.span=e.span;
        c.expert=e.expert;c.runner=e.runner;c.byte=byte;l.constituents.push_back(c);
        if(recall_credit[scale]>0)l.recall_content_gradient.emplace_back();
        if(routing_arm)l.routing.constituents.push_back(l.routing.events[slot]);
    }
    context_forward(e,s.memory);
    if(scale>0)address_forward(s.address_events[slot],e.content,e.before,s.stable,s.addressed,l.address,memory_mode);
    Vec recurrent{};mv(l.p.recurrent,e.before,recurrent);
    for(int i=0;i<D;++i){float a=scale?dot(l.address.read.data()+i*ADDRESS_V,s.address_events[slot].read.data(),ADDRESS_V):0;
        float transition=scale?0:expert.bias[i]+e.gate*e.residual[i];
        e.state[i]=std::tanh(e.content[i]+recurrent[i]+dot(l.p.read.data()+i*K,e.read.data(),K)+a+transition);}
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
    if(training&&routing_arm==3&&scale>0)mv(l.routing.predict,e.state,l.routing.events[slot].forecast);
    s.state=e.state;s.content=e.content;++s.executions;
    Vec recall_combined{};
    if(training&&recall_credit[scale]>0)
        for(int i=0;i<D;++i)recall_combined[i]=std::tanh(.5f*(e.content[i]+l.retained.previous_content[(3*i+13)%D]));
    int recall_slot=l.retained.admit(s.stable,e.content);
    if(training&&recall_credit[scale]>0){
        if(recall_slot>=0){
            drain_recall_credit(scale,recall_slot);
            auto& trace=l.recall_credit_state.units[recall_slot];trace=RecallCreditUnit{};
            trace.current=e.content_id;trace.previous=l.recall_credit_state.previous;trace.combined=recall_combined;
            std::copy_n(l.retained.units[recall_slot].value.begin(),RECALL_K,trace.value.begin());
        }
        l.recall_credit_state.previous=e.content_id;
    }
    if(training)++s.count;
    if(scale>0){mv(l.p.gain,s.state,s.gain);mv(l.p.bias,s.state,s.bias);}
    if(scale+1<int(levels.size())){
        auto& parent=levels[scale+1].s;int k=parent.children_count++;
        parent.children[k]=e.content;parent.lengths[k]=e.span;
        parent.child_ids[k]=e.content_id;
        if(semantic_laces)levels[scale+1].laces.stream.children[k]=e.expert;
        parent.stable_children[k]=s.stable;
        if(parent.children_count==GROUP){parent.children_count=0;forward_event(scale+1,parent.children,parent.lengths,0);}
    }
}

void Model::flush(int scale){
    auto& l=levels[scale];auto& s=l.s;if(!s.count)return;
    if(semantic_laces&&scale>0)lace_feedback(scale);
    Shared g{};AddressParameters ag{};std::array<Expert,BLOCK> eg{};std::array<int,BLOCK> selected{};int unique=0;
    std::array<Vec,BLOCK> key_delta{},runner_delta{};
    int uses=0,targets=0;for(int t=0;t<s.count;++t){uses+=s.events[t].uses;targets+=s.events[t].has_target;}
    float task_scale=1.f/std::max(1,uses),native_scale=aux_weight/std::max(1,targets);
    Mat sdr_head_grad{};int sdr_targets=0;
    if(routing_arm==3&&scale>0)for(int t=0;t<s.count;++t)sdr_targets+=l.routing.events[t].target;
    Vec future{};Memory memory_grad{};AddressMatrix address_grad{};
    for(int t=s.count-1;t>=0;--t){
        const auto& e=s.events[t];
        Vec dh=future,dgain{},dbias{},dp{},task{};
        for(int i=0;i<D;++i){task[i]=task_scale*e.task[i];dgain[i]=task_scale*e.gain_grad[i];dbias[i]=task_scale*e.bias_grad[i];dp[i]=native_scale*e.native[i];}
        if(scale>0){vjp(l.p.gain,dgain,task);vjp(l.p.bias,dbias,task);outer(g.gain,dgain,e.state);outer(g.bias,dbias,e.state);
            vjp(l.p.predict,dp,dh);outer(g.predict,dp,e.state);}
        if(routing_arm==3&&scale>0&&l.routing.events[t].target){
            Vec grad=l.routing.events[t].prediction_grad;for(auto& x:grad)x*=.1f/std::max(1,sdr_targets);
            vjp(l.routing.predict,grad,dh);outer(sdr_head_grad,grad,e.state);
        }
        l.task_gradient_energy+=energy(task);
        for(int i=0;i<D;++i)dh[i]+=language_credit[scale]*task[i];
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
        }else{
            int k=0;for(;k<unique;++k)if(selected[k]==e.expert)break;
            if(k==unique)selected[unique++]=e.expert;
            const auto& expert=l.experts[e.expert];Vec du{},gated=dz;float dgate=dot(dz.data(),e.residual.data());
            for(int i=0;i<D;++i){eg[k].bias[i]+=dz[i];gated[i]*=e.gate;}
            const auto& input=routing_arm?l.routing.events[t].expert_input:e.input;
            outer(eg[k].w,gated,input);vjp(expert.w,gated,du);
            float margin=dgate*e.gate_derivative;
            Vec route_grad{};for(int i=0;i<D;++i)route_grad[i]=margin*(routing_arm?l.routing.events[t].key_difference[i]:(expert.key[i]-l.experts[e.runner].key[i]));
            if(routing_arm)route_grad=routing_vjp(l.routing.events[t],route_grad);
            const auto& cue=routing_arm?l.routing.events[t].cue:e.input;
            for(int i=0;i<D;++i){key_delta[t][i]=margin*cue[i];runner_delta[t][i]=-margin*cue[i];
                du[i]+=route_grad[i];
                float input_grad=du[i]*(1-e.input[i]*e.input[i]);dc[i]+=input_grad;future[i]+=.5f*input_grad;}
        }
        // Keep the exact publication-time content VJP until all parent credit
        // has arrived. Context weights still retire at eight native events.
        auto& c=l.constituents.at(e.content_id);for(int i=0;i<D;++i)c.grad[i]+=dc[i];
    }
    float lr=(scale?slow_rate:learning_rate)*route_learning.shared();
    if(routing_arm==3&&scale>0)step(l.routing.predict,sdr_head_grad,lr);
    step(l.p.recurrent,g.recurrent,lr);step(l.p.read,g.read,lr);
    if(scale>0){step(l.p.gain,g.gain,lr);step(l.p.bias,g.bias,lr);step(l.p.predict,g.predict,lr);
        step(l.address.query_content,ag.query_content,lr);step(l.address.query_state,ag.query_state,lr);
        step(l.address.query_clock,ag.query_clock,lr);step(l.address.read,ag.read,lr);
        step(l.address.reconstruct,ag.reconstruct,lr);step(l.address.reconstruct_bias,ag.reconstruct_bias,lr);}
    else{
        for(int k=0;k<unique;++k){float rate=learning_rate*route_multiplier(l,selected[k]);
            step(l.experts[selected[k]].w,eg[k].w,rate);step(l.experts[selected[k]].bias,eg[k].bias,rate);}
        for(int t=0;t<s.count;++t){const auto& e=s.events[t];auto& key=l.experts[e.expert].key;auto& runner=l.experts[e.runner].key;
            float factor=route_multiplier(l,e.expert);
            step(key,key_delta[t],learning_rate*factor);if(e.runner!=e.expert)step(runner,runner_delta[t],learning_rate*route_multiplier(l,e.runner));
            Vec cue=routing_arm?l.routing.events[t].cue:e.input;normalize(cue);for(int i=0;i<D;++i)key[i]+=(route_rate*factor)*(cue[i]-key[i]);
            normalize(key);if(e.runner!=e.expert)normalize(runner);}
        for(int t=0;t<s.count;++t){update_index(l,s.events[t].expert);update_index(l,s.events[t].runner);}
        // Advance only after the whole sparse parameter update. A route's
        // clock counts selected training events, not other experts' traffic.
        for(int t=0;t<s.count;++t)advance_route(l,s.events[t].expert);
    }
    s.count=0;++s.updates;
}
void Model::drain_recall_credit(int scale,int slot){
    auto& l=levels[scale];auto& unit=l.recall_credit_state.units[slot];
    Vec current{},previous{};recall_value_vjp(unit,current,previous);
    if(unit.current>=0){auto& dst=l.recall_content_gradient.at(unit.current);for(int i=0;i<D;++i)dst[i]+=current[i];}
    if(unit.previous>=0){auto& dst=l.recall_content_gradient.at(unit.previous);for(int i=0;i<D;++i)dst[i]+=previous[i];}
    unit.gradient={};
}
void Model::retire_constituents(){
    // Query losses arrive per byte; normalize once over the observed epoch.
    // Credit is injected before the reverse constituent tree is retired.
    for(size_t scale=0;scale<levels.size();++scale)if(recall_credit[scale]>0){
        auto& l=levels[scale];for(int slot=0;slot<RECALL_SLOTS;++slot)drain_recall_credit(int(scale),slot);
        float weight=recall_credit[scale]/std::max(1,epoch_bytes);
        for(size_t k=0;k<l.constituents.size();++k)for(int i=0;i<D;++i)
            l.constituents[k].grad[i]+=weight*l.recall_content_gradient[k][i];
        l.recall_content_gradient.clear();l.recall_credit_state=RecallCreditState{};
    }
    // A reverse pass through a bounded tree of observed constituents only.
    // No recurrent graph, stale-weight replay, or unbounded ancestry is kept.
    for(int scale=int(levels.size())-1;scale>=0;--scale){
        auto& l=levels[scale];if(l.constituents.empty()){l.s.child_ids.fill(-1);continue;}
        if(scale==0){
            std::array<Vec,VOCAB> gradient{};
            for(const auto& c:l.constituents)for(int i=0;i<D;++i)
                gradient[c.byte][i]+=c.grad[i]*(1-c.content[i]*c.content[i]);
            for(int b=0;b<VOCAB;++b)step(embedding[b],gradient[b],learning_rate*route_learning.shared());
        }else{
            Shared g{};std::vector<Expert> eg;std::vector<int> selected;
            eg.reserve(l.constituents.size());selected.reserve(l.constituents.size());
            std::vector<Vec> keys(l.experts.size());std::vector<uint8_t> touched(l.experts.size());
            size_t ci=0;
            for(const auto& c:l.constituents){
                const RoutingTrace* rt=routing_arm?&l.routing.constituents[ci]:nullptr;++ci;
                int k=0;for(;k<int(selected.size());++k)if(selected[k]==c.expert)break;
                if(k==int(selected.size())){selected.push_back(c.expert);eg.emplace_back();}
                const auto& expert=l.experts[c.expert];Vec pre{},du{};float dgate=0;
                for(int i=0;i<D;++i){pre[i]=c.grad[i]*(1-c.content[i]*c.content[i]);eg[k].bias[i]+=pre[i];du[i]=pre[i];dgate+=pre[i]*c.residual[i];}
                Vec gated=pre;for(auto& x:gated)x*=c.gate;outer(eg[k].w,gated,rt?rt->expert_input:c.input);vjp(expert.w,gated,du);
                float margin=dgate*c.gate_derivative;
                touched[c.expert]=touched[c.runner]=1;
                Vec route_grad{};for(int i=0;i<D;++i)route_grad[i]=margin*(rt?rt->key_difference[i]:(expert.key[i]-l.experts[c.runner].key[i]));
                if(rt)route_grad=routing_vjp(*rt,route_grad);
                const auto& cue=rt?rt->cue:c.input;
                for(int i=0;i<D;++i){keys[c.expert][i]+=margin*cue[i];keys[c.runner][i]-=margin*cue[i];du[i]+=route_grad[i];}
                std::array<Vec,GROUP> child_grad{};
                for(int i=0;i<D;++i){float a=du[i]*(1-c.input[i]*c.input[i]);g.offset[i]+=a;
                    for(int pos=0;pos<GROUP;++pos){int j=(i*(2*pos+1)+13*pos)%D;
                        g.role[pos*D+i]+=a*c.children[pos][j];g.length[i]+=a*(pos+1)*c.lengths[pos]/std::max(1.f,c.span)*.25f;
                        child_grad[pos][j]+=constituent_credit*a*l.p.role[pos*D+i];}}
                auto& child=levels[scale-1];
                for(int pos=0;pos<GROUP;++pos)if(c.child_ids[pos]>=0){
                    auto& destination=child.constituents.at(c.child_ids[pos]).grad;
                    for(int i=0;i<D;++i)destination[i]+=child_grad[pos][i];
                    child.child_gradient_energy+=energy(child_grad[pos]);if(constituent_credit>0)++child.credit_edges;}
            }
            float shared_rate=slow_rate*route_learning.shared();
            step(l.p.role,g.role,shared_rate);step(l.p.length,g.length,shared_rate);step(l.p.offset,g.offset,shared_rate);
            for(size_t k=0;k<selected.size();++k){float rate=slow_rate*route_multiplier(l,selected[k]);
                step(l.experts[selected[k]].w,eg[k].w,rate);step(l.experts[selected[k]].bias,eg[k].bias,rate);}
            for(size_t k=0;k<keys.size();++k)if(touched[k])step(l.experts[k].key,keys[k],slow_rate*route_multiplier(l,int(k)));
            ci=0;
            for(const auto& c:l.constituents){auto& key=l.experts[c.expert].key;Vec cue=routing_arm?l.routing.constituents[ci].cue:c.input;++ci;normalize(cue);
                float rate=route_rate*route_multiplier(l,c.expert);
                for(int i=0;i<D;++i)key[i]+=rate*(cue[i]-key[i]);}
            for(size_t k=0;k<keys.size();++k)if(touched[k])normalize(l.experts[k].key);
            for(size_t k=0;k<keys.size();++k)if(touched[k])update_index(l,int(k));
            for(const auto& c:l.constituents)advance_route(l,c.expert);
        }
        l.constituents.clear();l.routing.constituents.clear();l.s.child_ids.fill(-1);++l.content_updates;
    }
    epoch_bytes=0;
    if(semantic_laces)lace_maintain();
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
    if(training&&epoch_bytes==epoch_limit)finish();
    forward_event(0,{}, {},byte);++bytes;if(training){++epoch_bytes;if(semantic_laces)++lace_training_bytes;}
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
    refresh();
    // Retained pairs contain observed input only. Queries execute now, including
    // when no slow level completes on this byte; they never consume the target.
    for(size_t l=0;l<levels.size();++l){
        auto& level=levels[l];
        recall_forward(level.retained,level.retrieval,levels[0].s.state,byte,level.read_now,int(l)==mask?1:recall_mode);
        if(int(l)==shuffle)std::rotate(level.read_now.value.begin(),level.read_now.value.begin()+11,level.read_now.value.end());
    }
    for(int i=0;i<D;++i)features[i]=std::tanh(levels[0].s.state[i]*(1+cached_gain[i])+cached_bias[i]);
}
float Model::teach(uint8_t target){
    int node=0;float loss=0;Vec feature_grad{};std::array<RecallValue,4> recall_grad{};
    const float effective_head=head_rate*route_learning.shared(),effective_query=query_rate*route_learning.shared();
    for(int depth=0;depth<8;++depth){
        int bit=(target>>(7-depth))&1;float z=decoder_bias[node]+dot(decoder[node].data(),features.data());
        for(size_t l=0;l<levels.size();++l)if(int(l)!=mask)z+=dot(levels[l].retrieval.read.data()+node*RECALL_V,levels[l].read_now.value.data(),RECALL_V);
        loss+=softplus(z)-bit*z;
        if(training){float error=sigmoid(z)-bit;
            for(size_t l=0;l<levels.size();++l){auto& p=levels[l].retrieval;auto& read=levels[l].read_now;
                float lr=effective_head/(1+dot(read.value.data(),read.value.data(),RECALL_V));
                for(int j=0;j<RECALL_V;++j){recall_grad[l][j]+=error*p.read[node*RECALL_V+j];p.read[node*RECALL_V+j]-=lr*error*read.value[j];}}
            // Read the exact weights which produced z before changing them.
            for(int j=0;j<D;++j)feature_grad[j]+=error*decoder[node][j];
            float lr=effective_head/(1+energy(features));
            for(int j=0;j<D;++j)decoder[node][j]-=lr*error*features[j];
            decoder_bias[node]-=effective_head*.05f*error;
        }
        node=2*node+1+bit;
    }
    if(training){
        if(semantic_laces)for(size_t scale=1;scale<levels.size();++scale){auto& l=levels[scale];if(l.s.count){auto& credit=l.laces.stream.events[l.s.count-1];credit.loss+=loss;++credit.targets;}}
        auto& fast=levels[0].s.events[levels[0].s.count-1];
        for(size_t l=0;l<levels.size();++l){RecallGradient g{};Vec di{};recall_backward(levels[l].retained,levels[l].retrieval,levels[l].read_now,recall_grad[l],g,di);
            if(recall_credit[l]>0){auto& level=levels[l];
                for(int k=0;k<level.retained.count;++k){auto& trace=level.recall_credit_state.units[k];
                    if(trace.current<0&&trace.previous<0)continue;
                    for(int j=0;j<RECALL_K;++j)trace.gradient[j]+=level.read_now.weights[k]*recall_grad[l][j];}}
            step(levels[l].retrieval.query,g.query,effective_query);
            for(int j=0;j<RECALL_K;++j)
                levels[l].retrieval.query_byte[levels[l].read_now.byte*RECALL_K+j]-=effective_query*std::clamp(g.query_byte[j],-2.f,2.f);
            for(int i=0;i<D;++i)fast.task[i]+=di[i];}
        for(int i=0;i<D;++i){float g=feature_grad[i]*(1-features[i]*features[i]);fast.task[i]+=g*(1+cached_gain[i]);
            for(size_t l=1;l<levels.size();++l){auto& s=levels[l].s;if(!s.count)continue;
                s.events[s.count-1].gain_grad[i]+=g*levels[0].s.state[i];s.events[s.count-1].bias_grad[i]+=g;}
        }
        for(auto& l:levels)if(l.s.count)++l.s.events[l.s.count-1].uses;
        route_learning.advance();
    }
    return loss;
}
void Model::log_probs(float* out)const{
    std::array<float,511> p{};
    for(int n=0;n<255;++n){float z=decoder_bias[n]+dot(decoder[n].data(),features.data());
        for(size_t l=0;l<levels.size();++l)if(int(l)!=mask)z+=dot(levels[l].retrieval.read.data()+n*RECALL_V,levels[l].read_now.value.data(),RECALL_V);
        p[2*n+1]=p[n]-softplus(z);p[2*n+2]=p[n]-softplus(-z);}
    std::copy(p.begin()+255,p.end(),out);
}
double Model::run(const uint8_t* x,const uint8_t* y,size_t n){
    if(!n)throw std::invalid_argument("at least one prediction is required");
    if(training&&(mask>=0||shuffle>=0||memory_mode||recall_mode||swap_level>=0))throw std::logic_error("interventions require frozen inference");
    double loss=0;for(size_t t=0;t<n;++t){observe(x[t]);loss+=teach(y[t]);}return loss/n;
}
void Model::finish(){if(training){for(size_t l=0;l<levels.size();++l)flush(int(l));retire_constituents();}}
void Model::reset(bool clear_store){
    finish();for(auto& l:levels){l.routing.reset_stream();l.laces.stream=LaceStream{};l.s=Stream{};l.retained=RecallStream{};l.read_now=RecallRead{};l.credit_edges=l.content_updates=0;l.task_gradient_energy=l.child_gradient_energy=0;}
    recalled={};previous_query={};cached_gain={};cached_bias={};features={};payload={};payload_count=0;have_query=false;bytes=0;route_comparisons=0;
    if(clear_store)store.clear();
}
void Model::inference(bool cold){
    // Snapshot fitted weights as they stand. Do not silently train/drain the
    // source to evaluate; pending derivatives are deliberately discarded.
    training=false;memory_writes=false;epoch_bytes=0;
    for(auto& l:levels){l.recall_credit_state=RecallCreditState{};l.recall_content_gradient.clear();l.routing.reset_stream();l.constituents.clear();l.s.child_ids.fill(-1);l.s.count=0;l.s.events={};l.s.address_events={};l.s.negatives={};l.s.negative_count=0;l.s.negative_cursor=0;}
    if(cold)reset(false);
}

std::string save(const Model& m){
    Writer w;uint64_t magic=0x31534d454f4853ULL;w.pod(magic);uint32_t version=m.checkpoint_version();w.pod(version);
    uint32_t abi=sizeof(Constituent)^sizeof(Event)^sizeof(Stream)^sizeof(AssociationStore)^sizeof(RecallParameters)^sizeof(RecallStream)^sizeof(RecallRead);w.pod(abi);
    if(version>=11){uint32_t payload=m.payload_version();w.pod(payload);}
    int count=int(m.levels.size()),target=int(m.parameter_count())-(m.routing_arm==3?int(m.levels.size()-1)*D*D:0);w.pod(count);w.pod(target);w.pod(m.indexed);w.pod(m.associations);
    w.pod(m.training);w.pod(m.memory_writes);w.pod(m.have_query);
    w.pod(m.embedding);w.pod(m.decoder);w.pod(m.decoder_bias);w.pod(m.store);
    w.pod(m.recalled);w.pod(m.previous_query);w.pod(m.cached_gain);w.pod(m.cached_bias);w.pod(m.features);
    w.pod(m.payload);w.pod(m.payload_count);w.pod(m.bytes);w.pod(m.route_comparisons);
    w.pod(m.learning_rate);w.pod(m.head_rate);w.pod(m.aux_weight);w.pod(m.route_rate);w.pod(m.mask);w.pod(m.shuffle);
    w.pod(m.reconstruction_weight);w.pod(m.memory_mode);w.pod(m.recall_mode);
    w.pod(m.slow_rate);w.pod(m.query_rate);w.pod(m.constituent_credit);w.pod(m.epoch_bytes);
    for(const auto& l:m.levels){w.vector(l.experts);w.pod(l.p);w.pod(l.address);w.pod(l.s);w.pod(l.retrieval);w.pod(l.retained);w.pod(l.read_now);w.vector(l.constituents);
        w.pod(l.credit_edges);w.pod(l.content_updates);w.pod(l.task_gradient_energy);w.pod(l.child_gradient_energy);}
    w.pod(m.semantic_laces);w.pod(m.lace_intervention);w.pod(m.lace_training_bytes);
    for(const auto& l:m.levels){const auto& b=l.laces;w.pod(b.records);w.vector(b.natural_use);w.vector(b.last_use);w.vector(b.expert_loss);w.vector(b.reserved);
        w.pod(b.content_mean);w.pod(b.context_mean);w.pod(b.stream);w.pod(b.clock);w.pod(b.next_identity);w.pod(b.matches);w.pod(b.misses);
        w.pod(b.allocations);w.pod(b.promotions);w.pod(b.rejections);w.pod(b.trial_routes);w.pod(b.retained_routes);w.pod(b.no_spare);}
    if(m.routing_arm){
        w.pod(m.routing_arm);w.pod(m.swap_level);
        for(const auto& l:m.levels){const auto& q=l.routing;
            w.pod(q.mean);w.vector(q.seed_cues);w.vector(q.bias);w.vector(q.selections);w.vector(q.window);
            w.pod(q.events);w.vector(q.constituents);w.pod(q.predict);w.pod(q.seen);w.pod(q.predict_targets);w.pod(q.predict_loss);
            w.pod(q.seeded);w.pod(q.seed_count);w.pod(q.window_count);
        }
    }
    if(version>=11)w.pod(m.language_credit);
    if(version>=12){w.pod(m.recall_credit);for(const auto& l:m.levels){w.pod(l.recall_credit_state);w.vector(l.recall_content_gradient);}}
    if(version==17){const auto& s=m.route_learning;
        uint32_t enabled=s.enabled;w.pod(enabled);w.pod(s.lower);w.pod(s.upper);w.pod(s.start);w.pod(s.target);w.pod(s.amplitude);
        w.pod(s.period);w.pod(s.targets);w.pod(s.ramp_elapsed);w.pod(s.ramp_targets);
        for(const auto& l:m.levels)w.vector(l.lr_visits);
    }
    return w.out;
}
Model load(const char* p,size_t n){
    Reader r{p,n};uint64_t magic;uint32_t version,abi;r.pod(magic);r.pod(version);r.pod(abi);
    if(magic!=0x31534d454f4853ULL||version<5||(version>12&&version!=17)||abi!=(sizeof(Constituent)^sizeof(Event)^sizeof(Stream)^sizeof(AssociationStore)^sizeof(RecallParameters)^sizeof(RecallStream)^sizeof(RecallRead)))throw std::runtime_error("incompatible multiscale checkpoint");
    const bool credit_extension=version>=11,recall_extension=version>=12;
    const bool route_lr_extension=version==17;
    if(credit_extension){r.pod(version);if(version<6||version>10)throw std::runtime_error("invalid credit payload version");}
    int count,target;bool index,assoc;r.pod(count);r.pod(target);r.pod(index);r.pod(assoc);
    Model m(count,target,7,index,assoc);r.pod(m.training);r.pod(m.memory_writes);r.pod(m.have_query);
    r.pod(m.embedding);r.pod(m.decoder);r.pod(m.decoder_bias);r.pod(m.store);
    r.pod(m.recalled);r.pod(m.previous_query);r.pod(m.cached_gain);r.pod(m.cached_bias);r.pod(m.features);
    r.pod(m.payload);r.pod(m.payload_count);r.pod(m.bytes);r.pod(m.route_comparisons);
    r.pod(m.learning_rate);r.pod(m.head_rate);r.pod(m.aux_weight);r.pod(m.route_rate);r.pod(m.mask);r.pod(m.shuffle);
    r.pod(m.reconstruction_weight);r.pod(m.memory_mode);r.pod(m.recall_mode);
    r.pod(m.slow_rate);r.pod(m.query_rate);r.pod(m.constituent_credit);r.pod(m.epoch_bytes);
    for(float rate:{m.learning_rate,m.slow_rate,m.query_rate,m.head_rate,m.aux_weight,m.route_rate,m.constituent_credit})
        if(!std::isfinite(rate)||rate<0)throw std::runtime_error("invalid learning policy");
    if(m.epoch_bytes<0||m.epoch_bytes>m.epoch_limit||(!m.training&&m.epoch_bytes))throw std::runtime_error("invalid constituent epoch");
    if(!std::isfinite(m.reconstruction_weight)||m.reconstruction_weight<0||m.memory_mode<0||m.memory_mode>3||(m.training&&m.memory_mode))throw std::runtime_error("invalid addressed-memory configuration");
    if(m.recall_mode<0||m.recall_mode>4||(m.training&&m.recall_mode))throw std::runtime_error("invalid retained-recall intervention");
    if(m.payload_count<0||m.payload_count>=GROUP||m.mask>=count||m.shuffle>=count)throw std::runtime_error("invalid checkpoint counters");
    for(size_t level=0;level<m.levels.size();++level){auto& l=m.levels[level];r.vector(l.experts);r.pod(l.p);r.pod(l.address);r.pod(l.s);r.pod(l.retrieval);r.pod(l.retained);r.pod(l.read_now);
        uint64_t size;r.pod(size);int span=1;for(size_t i=0;i<level;++i)span*=GROUP;
        if(size>uint64_t(m.epoch_limit/span+1)||(!m.training&&size))throw std::runtime_error("invalid constituent buffer size");
        l.constituents.resize(size);for(auto& c:l.constituents)r.pod(c);
        r.pod(l.credit_edges);r.pod(l.content_updates);r.pod(l.task_gradient_energy);r.pod(l.child_gradient_energy);
        for(const auto& c:l.constituents){
            if(c.byte<0||c.byte>=VOCAB||c.expert<0||c.expert>=int(l.experts.size())||c.runner<0||c.runner>=int(l.experts.size()))throw std::runtime_error("invalid constituent route");
            for(int id:c.child_ids)if(id < -1||(level&&id>=int(m.levels[level-1].constituents.size()))||(!level&&id!=-1))throw std::runtime_error("invalid constituent child");
        }
        for(int id:l.s.child_ids)if(id < -1||(level&&id>=int(m.levels[level-1].constituents.size()))||(!level&&id!=-1))throw std::runtime_error("invalid pending constituent child");
        for(int t=0;t<l.s.count&&t<BLOCK;++t)if(l.s.events[t].content_id<0||l.s.events[t].content_id>=int(size))throw std::runtime_error("invalid content eligibility");
        if(l.retained.count<0||l.retained.count>RECALL_SLOTS||l.read_now.byte<0||l.read_now.byte>=VOCAB)
            throw std::runtime_error("invalid retained-recall counters");
        for(int i=0;i<l.retained.count;++i){const auto& u=l.retained.units[i];
            if(!u.born||u.born>l.retained.clock)throw std::runtime_error("invalid retained occurrence");
            for(float x:u.key)if(!std::isfinite(x))throw std::runtime_error("invalid retained key");
            for(float x:u.value)if(!std::isfinite(x))throw std::runtime_error("invalid retained value");}
        const auto& s=l.s;if(s.count<0||s.count>BLOCK||s.children_count<0||s.children_count>=GROUP||s.negative_count<0||s.negative_count>NEG||s.negative_cursor<0||s.negative_cursor>=NEG)throw std::runtime_error("invalid event ring");
        if(level&&s.addressed.clock.ticks!=s.executions)throw std::runtime_error("invalid native clock");
        for(double v:s.addressed.clock.phase)if(!std::isfinite(v))throw std::runtime_error("invalid clock phase");
        for(int t=0;t<s.count;++t)if(s.events[t].expert<0||s.events[t].expert>=int(l.experts.size())||s.events[t].runner<0||s.events[t].runner>=int(l.experts.size())||s.events[t].byte<0||s.events[t].byte>=256||s.events[t].reconstruction_role<0||s.events[t].reconstruction_role>=GROUP)throw std::runtime_error("invalid event route");
        m.rebuild_index(l);
    }
    for(const auto& rec:m.store.records)if(rec.key_size>KEY_BYTES||rec.payload_size>PAYLOAD)throw std::runtime_error("invalid association record");
    m.semantic_laces=false; // v5 has no semantic policy: never silently migrate it.
    m.lace_policy=version==6?1:2;
    if(version>=6){
        r.pod(m.semantic_laces);r.pod(m.lace_intervention);
        if(m.lace_intervention<0||m.lace_intervention>1||(m.training&&m.lace_intervention))throw std::runtime_error("invalid lace intervention");r.pod(m.lace_training_bytes);
        for(auto& l:m.levels){auto& b=l.laces;r.pod(b.records);r.vector(b.natural_use);r.vector(b.last_use);r.vector(b.expert_loss);r.vector(b.reserved);
            r.pod(b.content_mean);r.pod(b.context_mean);r.pod(b.stream);r.pod(b.clock);r.pod(b.next_identity);r.pod(b.matches);r.pod(b.misses);
            r.pod(b.allocations);r.pod(b.promotions);r.pod(b.rejections);r.pod(b.trial_routes);r.pod(b.retained_routes);r.pod(b.no_spare);
            if(b.last_use.size()!=l.experts.size()||b.natural_use.size()!=l.experts.size()||b.expert_loss.size()!=l.experts.size()||b.reserved.size()!=l.experts.size())throw std::runtime_error("invalid lace bank size");
            for(auto v:b.reserved)if(v>1)throw std::runtime_error("invalid reserved expert");
            for(const auto& a:b.records)if(a.identity){
                if(a.status<0||a.status>2||a.incumbent<0||a.incumbent>=int(l.experts.size())||a.specialist < -1||a.specialist>=int(l.experts.size())||(a.status&&a.specialist<0)||!std::isfinite(a.loss))throw std::runtime_error("invalid semantic lace");}
            for(const auto& e:b.stream.events)if(e.record < -1||e.record>=LACE_SLOTS||e.arm<0||e.arm>1||!std::isfinite(e.loss))throw std::runtime_error("invalid lace eligibility");
        }
    }
    if(version>=8){
        m.balance_policy=version>=10?2:(version>=9?1:0);
        r.pod(m.routing_arm);r.pod(m.swap_level);
        if(m.routing_arm<1||m.routing_arm>3||m.swap_level < -1||m.swap_level>=count||(m.training&&m.swap_level>=0)||m.semantic_laces)
            throw std::runtime_error("invalid routing policy");
        for(auto& l:m.levels){auto& q=l.routing;
            r.pod(q.mean);r.vector(q.seed_cues);r.vector(q.bias);r.vector(q.selections);r.vector(q.window);r.pod(q.events);
            q.constituents.resize(l.constituents.size());r.vector(q.constituents);r.pod(q.predict);r.pod(q.seen);r.pod(q.predict_targets);r.pod(q.predict_loss);
            r.pod(q.seeded);r.pod(q.seed_count);r.pod(q.window_count);
            if(q.seeded<0||q.seeded>1||q.seed_count<0||q.seed_count>int(q.seed_cues.size())||q.window_count<0||q.window_count>=256)
                throw std::runtime_error("invalid routing counters");
            for(float v:q.mean)if(!std::isfinite(v))throw std::runtime_error("invalid routing mean");
            for(float v:q.bias)if(!std::isfinite(v))throw std::runtime_error("invalid routing bias");
        }
    }
    if(credit_extension){r.pod(m.language_credit);
        for(size_t level=0;level<m.language_credit.size();++level){float value=m.language_credit[level];
            if(!std::isfinite(value)||value<0||(level>=m.levels.size()&&value!=1))throw std::runtime_error("invalid language credit");}}
    if(recall_extension){r.pod(m.recall_credit);
        for(size_t level=0;level<m.recall_credit.size();++level){float value=m.recall_credit[level];
            if(!std::isfinite(value)||value<0||(level>=m.levels.size()&&value!=0))throw std::runtime_error("invalid recall credit");}
        for(size_t level=0;level<m.levels.size();++level){auto& l=m.levels[level];r.pod(l.recall_credit_state);
            uint64_t size;r.pod(size);
            if(size!=(m.training&&m.recall_credit[level]>0?l.constituents.size():0))throw std::runtime_error("invalid recall credit buffer");
            l.recall_content_gradient.resize(size);for(auto& v:l.recall_content_gradient)r.pod(v);
            auto valid_id=[&](int id){return id==-1||(id>=0&&uint64_t(id)<size);};
            if(!valid_id(l.recall_credit_state.previous))throw std::runtime_error("invalid prior recall producer");
            for(const auto& unit:l.recall_credit_state.units){
                if(!valid_id(unit.current)||!valid_id(unit.previous))throw std::runtime_error("invalid recall producer");
                for(float v:unit.gradient)if(!std::isfinite(v))throw std::runtime_error("invalid recall derivative");}
        }
    }
    if(route_lr_extension){auto& s=m.route_learning;uint32_t enabled;r.pod(enabled);
        if(enabled>1)throw std::runtime_error("invalid route learning switch");
        s.configured=true;s.enabled=bool(enabled);r.pod(s.lower);r.pod(s.upper);r.pod(s.start);r.pod(s.target);r.pod(s.amplitude);
        r.pod(s.period);r.pod(s.targets);r.pod(s.ramp_elapsed);r.pod(s.ramp_targets);s.validate();
        for(float rate:{m.learning_rate,m.slow_rate,m.query_rate,m.head_rate,m.route_rate})s.validate_rate(rate);
        if(m.semantic_laces)throw std::runtime_error("route learning requires stable expert identities; disable semantic laces");
        for(auto& l:m.levels)r.vector(l.lr_visits);
    }
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
    auto& m=*static_cast<shoe::Model*>(p);for(float rate:{lr,head,route})m.route_learning.validate_rate(rate);
    m.learning_rate=lr;m.head_rate=head;m.aux_weight=aux;m.route_rate=route;
});}
int ms_learning(void* p,float fast,float slow,float query,float credit){return guard([&]{
    for(float value:{fast,slow,query,credit})if(!std::isfinite(value)||value<0)throw std::invalid_argument("learning policy must be finite and nonnegative");
    auto& m=*static_cast<shoe::Model*>(p);for(float rate:{fast,slow,query})m.route_learning.validate_rate(rate);
    m.learning_rate=fast;m.slow_rate=slow;m.query_rate=query;m.constituent_credit=credit;
});}
int ms_route_learning(void* p,int enabled,float lower,float upper,float anchor,float amplitude,uint64_t period){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(!m.training||m.epoch_bytes)throw std::logic_error("configure route learning at an explicit training boundary");
    if(m.semantic_laces)throw std::invalid_argument("route learning requires stable expert identities; disable semantic laces");
    if(enabled<0||enabled>1)throw std::invalid_argument("invalid route learning switch");
    auto s=m.route_learning;s.configured=true;s.enabled=enabled;s.lower=lower;s.upper=upper;s.start=s.target=anchor;
    s.amplitude=amplitude;s.period=period;s.ramp_elapsed=s.ramp_targets=0;s.validate();
    for(float rate:{m.learning_rate,m.slow_rate,m.query_rate,m.head_rate,m.route_rate})s.validate_rate(rate);
    m.route_learning=s;
});}
int ms_route_anchor(void* p,float target,uint64_t targets){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(!m.training||!m.route_learning.configured)throw std::logic_error("anchor changes require a configured training model");
    auto s=m.route_learning;s.start=s.anchor();s.target=target;s.ramp_elapsed=0;s.ramp_targets=targets;
    if(!targets)s.start=target;
    s.validate();m.route_learning=s;
});}
int ms_route_learning_stats(void* p,double* out,uint64_t* clocks){return guard([&]{
    const auto& s=static_cast<shoe::Model*>(p)->route_learning;
    out[0]=s.configured;out[1]=s.enabled;out[2]=s.lower;out[3]=s.upper;out[4]=s.anchor();out[5]=s.amplitude;
    out[6]=double(s.period);out[7]=double(s.targets);out[8]=s.target;out[9]=double(s.ramp_elapsed);out[10]=double(s.ramp_targets);
    clocks[0]=s.period;clocks[1]=s.targets;clocks[2]=s.ramp_elapsed;clocks[3]=s.ramp_targets;
});}
int ms_route_learning_routes(void* p,int scale,uint64_t* visits,float* factors,size_t n){return guard([&]{
    const auto& m=*static_cast<shoe::Model*>(p);
    if(scale<0||scale>=int(m.levels.size())||n!=m.levels[scale].experts.size())throw std::invalid_argument("invalid route inspection shape");
    const auto& l=m.levels[scale];for(size_t k=0;k<n;++k){visits[k]=l.lr_visits[k];factors[k]=m.route_multiplier(l,int(k));}
});}
int ms_language_credit(void* p,const float* gains,size_t n){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(!m.training)throw std::logic_error("language credit is a training policy");
    if(n!=m.levels.size())throw std::invalid_argument("one language credit gain is required per level");
    for(size_t i=0;i<n;++i)if(!std::isfinite(gains[i])||gains[i]<0)throw std::invalid_argument("language credit must be finite and nonnegative");
    std::copy(gains,gains+n,m.language_credit.begin());
});}
int ms_credit_stats(void* p,double* out){return guard([&]{
    const auto& m=*static_cast<shoe::Model*>(p);std::copy(m.language_credit.begin(),m.language_credit.end(),out);
    std::copy(m.recall_credit.begin(),m.recall_credit.end(),out+4);
});}
int ms_recall_credit(void* p,const float* gains,size_t n){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(!m.training||m.epoch_bytes)throw std::logic_error("change recall credit at an explicit training boundary");
    if(n!=m.levels.size())throw std::invalid_argument("one recall credit gain is required per level");
    for(size_t i=0;i<n;++i)if(!std::isfinite(gains[i])||gains[i]<0)throw std::invalid_argument("recall credit must be finite and nonnegative");
    std::copy(gains,gains+n,m.recall_credit.begin());
    for(auto& l:m.levels){l.recall_credit_state=shoe::RecallCreditState{};l.recall_content_gradient.clear();}
});}
int ms_routing(void* p,int arm){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(arm<0||arm>3||m.epoch_bytes||m.bytes||m.routing_arm)throw std::invalid_argument("set routing once on a fresh model");
    if(arm&&m.semantic_laces)throw std::invalid_argument("routing arms require laces disabled");m.routing_arm=arm;
});}
int ms_balance(void* p,int policy){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if((policy!=1&&policy!=2)||m.routing_arm<2||!m.training)
        throw std::invalid_argument("balance policy requires a training C/D model and policy 1 or 2");
    m.balance_policy=policy;
});}
int ms_swap_experts(void* p,int level){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||level < -1||level>=int(m.levels.size()))throw std::invalid_argument("expert swapping requires frozen inference");m.swap_level=level;
});}
int ms_routing_stats(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+64,0.);out[0]=m.routing_arm;out[1]=m.balance_policy;
    for(size_t i=0;i<m.levels.size();++i){const auto& q=m.levels[i].routing;auto* a=out+4+i*15;
        a[0]=q.seen;a[1]=q.seeded;a[2]=*std::min_element(q.selections.begin(),q.selections.end());a[3]=*std::max_element(q.selections.begin(),q.selections.end());
        a[4]=q.predict_targets;a[5]=q.predict_loss;a[6]=*std::min_element(q.bias.begin(),q.bias.end());a[7]=*std::max_element(q.bias.begin(),q.bias.end());
        a[8]=q.seed_count;a[9]=q.window_count;
    }
});}
int ms_laces(void* p,int enabled){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(enabled&&m.routing_arm)throw std::invalid_argument("routing arms cannot enable semantic laces");
    if(enabled&&m.route_learning.configured)throw std::invalid_argument("route learning requires stable expert identities");
    if(enabled!=0&&enabled!=1)throw std::invalid_argument("invalid lace policy");
    if(m.semantic_laces!=bool(enabled)){m.finish();m.semantic_laces=enabled;if(enabled)m.lace_policy=2;}
});}
int ms_lace_intervene(void* p,int mode){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||mode<0||mode>1)throw std::invalid_argument("lace intervention requires frozen inference");m.lace_intervention=mode;
});}
int ms_lace_records(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+4*shoe::LACE_SLOTS*16,0.);
    for(size_t l=0;l<m.levels.size();++l)for(int k=0;k<shoe::LACE_SLOTS;++k){const auto& r=m.levels[l].laces.records[k];double* a=out+(l*shoe::LACE_SLOTS+k)*16;
        a[0]=r.identity;a[1]=r.incumbent;a[2]=r.specialist;a[3]=r.status;a[4]=r.hits;a[5]=r.loss;a[6]=r.control_n;a[7]=r.trial_n;
        a[8]=r.control_n?r.control_sum/r.control_n:0;a[9]=r.trial_n?r.trial_sum/r.trial_n:0;a[10]=r.last;a[11]=r.attempts;
        for(int i=0;i<4;++i)a[12+i]=r.code.path[i];}
});}
int ms_lace_stats(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+64,0.);
    out[0]=m.semantic_laces;out[1]=m.lace_training_bytes;out[2]=m.checkpoint_version();
    for(size_t i=0;i<m.levels.size();++i){const auto& b=m.levels[i].laces;double* a=out+4+i*15;
        a[0]=b.clock;a[1]=b.matches;a[2]=b.misses;a[3]=b.allocations;a[4]=b.promotions;a[5]=b.rejections;a[6]=b.trial_routes;a[7]=b.retained_routes;a[8]=b.no_spare;
        for(const auto& r:b.records)if(r.identity){++a[9];a[10]+=r.status==1;a[11]+=r.status==2;}
        for(size_t k=0;k<b.reserved.size();++k){a[12]+=b.reserved[k];a[13]+=!b.reserved[k]&&(!b.natural_use[k]||b.clock-b.last_use[k]>std::max(uint64_t(256),b.clock/20));}
        a[14]=sizeof(shoe::LaceBank)+b.natural_use.size()*(2*sizeof(uint64_t)+sizeof(double)+sizeof(uint8_t));}
});}
int ms_learning_stats(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+32,0.);
    out[0]=m.learning_rate;out[1]=m.slow_rate;out[2]=m.query_rate;out[3]=m.head_rate;out[4]=m.constituent_credit;
    out[5]=m.epoch_bytes;out[6]=m.epoch_limit;out[7]=m.aux_weight;out[8]=m.route_rate;
    for(size_t i=0;i<m.levels.size();++i){const auto& l=m.levels[i];out[12+i]=l.constituents.size();out[16+i]=l.credit_edges;
        out[20+i]=l.task_gradient_energy;out[24+i]=l.child_gradient_energy;out[28+i]=l.content_updates;}
});}
int ms_reconstruction(void* p,float weight){return guard([&]{
    if(!std::isfinite(weight)||weight<0)throw std::invalid_argument("reconstruction weight must be finite and nonnegative");
    static_cast<shoe::Model*>(p)->reconstruction_weight=weight;
});}
int ms_memory_intervene(void* p,int mode){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||mode<0||mode>3)throw std::invalid_argument("memory interventions require frozen inference and a valid mode");m.memory_mode=mode;
});}
int ms_recall_intervene(void* p,int mode){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);
    if(m.training||mode<0||mode>4)throw std::invalid_argument("recall interventions require frozen inference and a valid mode");
    m.recall_mode=mode;
});}
int ms_recall_stats(void* p,double* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);std::fill(out,out+16,0.);
    for(size_t i=0;i<m.levels.size();++i){out[i]=m.levels[i].retained.count;out[4+i]=m.levels[i].retained.clock;}
    out[8]=shoe::RECALL_SLOTS;out[9]=m.recall_mode;out[10]=sizeof(shoe::RecallStream)*m.levels.size();
    out[11]=sizeof(shoe::RecallParameters)/sizeof(float)*m.levels.size();
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

// Snapshot-only inspection. The training byte loop has no instrumentation.
extern "C" int plateau_probe(void* p,const uint8_t* x,size_t n,double* out,int* routes){return guard([&]{
    using namespace shoe;Model m=*static_cast<Model*>(p);m.finish();m.inference(true);
    std::fill(out,out+4*24,0.);std::fill(routes,routes+4*n,-1);
    std::array<std::vector<uint64_t>,4> counts;
    for(size_t l=0;l<m.levels.size();++l)counts[l].resize(m.levels[l].experts.size());
    for(size_t t=0;t<n;++t){std::array<uint64_t,4> before{};
        for(size_t l=0;l<m.levels.size();++l)before[l]=m.levels[l].s.executions;
        m.observe(x[t]);
        for(size_t l=0;l<m.levels.size();++l){auto& level=m.levels[l];if(level.s.executions==before[l])continue;
            const auto& e=level.s.events[0];double* o=out+l*24;++o[0];++counts[l][e.expert];routes[l*n+t]=e.expert;
            o[1]+=e.gate_derivative;o[2]+=e.gate_derivative<.01;o[3]+=e.gate;
            for(int j=0;j<D;++j){o[4]+=std::abs(e.input[j])>.95;o[5]+=std::abs(e.content[j])>.95;o[6]+=std::abs(e.state[j])>.95;
                o[7]+=1-e.input[j]*e.input[j];o[8]+=1-e.content[j]*e.content[j];o[9]+=1-e.state[j]*e.state[j];
                if(l==0){o[10]+=std::abs(m.features[j])>.95;o[11]+=1-m.features[j]*m.features[j];}}
            for(float v:level.read_now.query)o[12]+=std::abs(v)>.95;
        }
    }
    for(size_t l=0;l<m.levels.size();++l){auto& level=m.levels[l];double* o=out+l*24;double h=0,ss=0;
        for(auto c:counts[l]){double q=c/o[0];if(c){++o[13];h-=q*std::log(q);}ss+=q*q;o[14]=std::max(o[14],q);}
        o[15]=std::exp(h);o[16]=1/ss;o[17]=counts[l].size();
        for(const auto& e:level.experts){for(float v:e.w)o[18]+=double(v)*v;for(float v:e.bias)o[19]+=double(v)*v;}
        for(size_t i=0;i<level.experts.size();++i)for(size_t j=0;j<i;++j){o[20]+=dot(level.experts[i].key.data(),level.experts[j].key.data());++o[21];}
        for(float v:level.p.recurrent)o[22]+=double(v)*v;
        o[23]=level.s.executions;
    }
});}
