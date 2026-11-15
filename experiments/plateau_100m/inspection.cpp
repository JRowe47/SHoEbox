// Offline diagnostics, compiled beside an otherwise identical runtime.
// No instrumentation is inserted into the production training loop.
extern "C" int diag_policy(int balance,float sdr){
    shoe::diagnostic_balance_update=balance;
    shoe::diagnostic_sdr_weight=sdr;
    return 0;
}
extern "C" int diag_aux_levels(float a,float b,float c,float d){
    shoe::diagnostic_aux_factor[0]=a;shoe::diagnostic_aux_factor[1]=b;
    shoe::diagnostic_aux_factor[2]=c;shoe::diagnostic_aux_factor[3]=d;return 0;
}
extern "C" int diag_set_bias(void* p,float value){return guard([&]{
    for(auto& l:static_cast<shoe::Model*>(p)->levels)std::fill(l.routing.bias.begin(),l.routing.bias.end(),value);
});}
extern "C" int diag_control(void* p,int mode,int level){return guard([&]{
    auto& m=*static_cast<shoe::Model*>(p);
    if(m.training)throw std::logic_error("snapshot interventions require inference");
    if(mode==1)for(auto& l:m.levels)std::fill(l.routing.bias.begin(),l.routing.bias.end(),0.f);
    if(mode==2)for(size_t i=1;i<m.levels.size();++i){
        m.levels[i].p.gain={};m.levels[i].p.bias={};m.levels[i].retrieval.read={};
    }
    if(mode==3){auto& l=m.levels.at(level);for(auto& e:l.experts){e.w={};e.bias={};}}
});}
extern "C" int diag_trace(void* p,const uint8_t* x,const uint8_t* y,size_t n,
                           double* out,int* routes,double* losses){return guard([&]{
    using namespace shoe;auto& m=*static_cast<Model*>(p);
    if(m.training)throw std::logic_error("trace requires inference");
    std::fill(out,out+4*24,0.);std::fill(routes,routes+4*n,-1);
    for(size_t t=0;t<n;++t){
        std::array<uint64_t,4> before{};
        for(size_t l=0;l<m.levels.size();++l)before[l]=m.levels[l].s.executions;
        m.observe(x[t]);losses[t]=m.teach(y[t]);
        for(size_t l=0;l<m.levels.size();++l){
            auto& level=m.levels[l];if(level.s.executions==before[l])continue;
            const auto& e=level.s.events[0];const auto& rt=level.routing.events[0];
            double* o=out+l*24;++o[0];routes[l*n+t]=e.expert;
            o[1]+=e.gate_derivative;o[2]+=e.gate_derivative<.01;o[3]+=e.gate;
            for(int j=0;j<D;++j){
                o[4]+=std::abs(e.input[j])>.95;o[5]+=std::abs(e.content[j])>.95;o[6]+=std::abs(e.state[j])>.95;
                o[7]+=1-e.input[j]*e.input[j];o[8]+=1-e.content[j]*e.content[j];o[9]+=1-e.state[j]*e.state[j];
                if(l==0){o[10]+=std::abs(m.features[j])>.95;o[11]+=1-m.features[j]*m.features[j];}
            }
            o[12]+=energy(e.residual);o[13]+=rt.norm;
            if(level.s.executions%32==1){
                ++o[14];float best=-1e30f;int winner=0;
                for(size_t k=0;k<level.experts.size();++k){
                    float s=dot(level.experts[k].key.data(),rt.cue.data());
                    if(s>best){best=s;winner=k;}
                }
                o[15]+=winner!=e.expert;o[16]+=dot(level.experts[e.expert].key.data(),rt.cue.data());
                o[17]+=best;o[18]+=level.routing.bias[e.expert];o[19]+=level.routing.bias[winner];
            }
            o[20]+=energy(e.state);o[21]+=energy(e.input);o[22]+=energy(e.content);
            if(l==0)o[23]+=energy(m.features);
        }
    }
});}
extern "C" int diag_bank(void* p,int scale,double* out){return guard([&]{
    using namespace shoe;auto& l=static_cast<Model*>(p)->levels.at(scale);
    for(size_t k=0;k<l.experts.size();++k){const auto& e=l.experts[k];auto* q=out+k*5;
        q[0]=l.routing.selections[k];q[1]=l.routing.bias[k];q[2]=dot(e.w.data(),e.w.data(),D*D);
        q[3]=energy(e.bias);q[4]=energy(e.key);
    }
});}

namespace shoe {
// Recompute objective-specific content VJPs on immutable pending events.
// This mirrors flush's local derivatives but performs no weight update.
std::vector<float> diagnostic_content_gradient(const Model& m,int scale,int channel){
    const auto& l=m.levels[scale];const auto& s=l.s;
    int uses=0,targets=0,sdr_targets=0;
    for(int t=0;t<s.count;++t){uses+=s.events[t].uses;targets+=s.events[t].has_target;sdr_targets+=l.routing.events[t].target;}
    bool task_on=channel==0||channel==4,contrast_on=channel==1||channel==4;
    bool sdr_on=channel==2||channel==4,reconstruct_on=channel==3||channel==4;
    Vec future{};Memory memory_grad{};AddressMatrix address_grad{};AddressParameters ag{};
    std::vector<float> result(s.count*D);
    for(int t=s.count-1;t>=0;--t){const auto& e=s.events[t];Vec dh=future,task{},dgain{},dbias{},dp{};
        for(int i=0;i<D;++i){
            if(task_on){task[i]=e.task[i]/std::max(1,uses);dgain[i]=e.gain_grad[i]/std::max(1,uses);dbias[i]=e.bias_grad[i]/std::max(1,uses);}
            if(contrast_on)dp[i]=m.aux_weight/std::max(1,targets)*e.native[i];
        }
        if(scale){vjp(l.p.gain,dgain,task);vjp(l.p.bias,dbias,task);vjp(l.p.predict,dp,dh);}
        if(sdr_on&&scale&&l.routing.events[t].target){Vec g=l.routing.events[t].prediction_grad;
            for(auto& x:g)x*=diagnostic_sdr_weight/std::max(1,sdr_targets);vjp(l.routing.predict,g,dh);}
        for(int i=0;i<D;++i)dh[i]+=m.language_credit[scale]*task[i];
        Vec dz{},dc{};Small dr{};AddressValue ar{};future={};
        for(int i=0;i<D;++i){dz[i]=dh[i]*(1-e.state[i]*e.state[i]);dc[i]=dz[i];
            for(int j=0;j<K;++j)dr[j]+=l.p.read[i*K+j]*dz[i];
            if(scale)for(int j=0;j<ADDRESS_V;++j)ar[j]+=l.address.read[i*ADDRESS_V+j]*dz[i];}
        vjp(l.p.recurrent,dz,future);context_backward(e,dr,memory_grad,dc);
        if(scale){
            address_backward(s.address_events[t],e.content,e.before,l.address,ar,address_grad,dc,future,ag);
            if(reconstruct_on&&m.reconstruction_weight>0)reconstruct(e.content,e.children[e.reconstruction_role],e.reconstruction_role,l.address,m.reconstruction_weight/s.count,dc,ag);
        }else{
            const auto& expert=l.experts[e.expert];Vec du{},gated=dz;float dgate=dot(dz.data(),e.residual.data());
            for(auto& x:gated)x*=e.gate;vjp(expert.w,gated,du);float margin=dgate*e.gate_derivative;
            Vec route_grad{};for(int i=0;i<D;++i)route_grad[i]=margin*l.routing.events[t].key_difference[i];
            route_grad=routing_vjp(l.routing.events[t],route_grad);
            for(int i=0;i<D;++i){du[i]+=route_grad[i];float input_grad=du[i]*(1-e.input[i]*e.input[i]);dc[i]+=input_grad;future[i]+=.5f*input_grad;}
        }
        std::copy(dc.begin(),dc.end(),result.begin()+t*D);
    }
    return result;
}
}
extern "C" int diag_gradient(void* p,const uint8_t* x,const uint8_t* y,size_t n,double* out){return guard([&]{
    using namespace shoe;auto& m=*static_cast<Model*>(p);
    if(!m.training)throw std::logic_error("gradient probe needs disposable training state");
    std::fill(out,out+4*20,0.);
    for(size_t t=0;t<n;++t){
        int span=1;
        for(size_t l=0;l<m.levels.size();++l){
            auto& s=m.levels[l].s;bool next=(t+1)%span==0;
            if(s.count==BLOCK&&(next||m.epoch_bytes==m.epoch_limit)){
                double* o=out+l*20;++o[0];std::array<std::vector<float>,5> gs;
                for(int channel=0;channel<5;++channel)gs[channel]=diagnostic_content_gradient(m,int(l),channel);
                for(size_t j=0;j<gs[0].size();++j){
                    double sum=0;for(int c=0;c<4;++c){o[1+c]+=double(gs[c][j])*gs[c][j];sum+=gs[c][j];}
                    int index=5;for(int c=0;c<4;++c)for(int d=c+1;d<4;++d)o[index++]+=double(gs[c][j])*gs[d][j];
                    o[11]+=sum*sum;o[12]=std::max(o[12],std::abs(sum-gs[4][j]));o[13]=std::max(o[13],std::abs(double(gs[4][j])));
                }
            }
            span*=GROUP;
        }
        m.observe(x[t]);m.teach(y[t]);
    }
});}
extern "C" int diag_validate_gradient(void* p,double* out){return guard([&]{
    using namespace shoe;auto& m=*static_cast<Model*>(p);std::fill(out,out+8,0.);
    for(size_t scale=0;scale<m.levels.size();++scale){auto& l=m.levels[scale];int count=l.s.count;
        auto predicted=diagnostic_content_gradient(m,int(scale),4);
        std::vector<Vec> prior;std::vector<int> ids;
        for(int t=0;t<count;++t){int id=l.s.events[t].content_id;ids.push_back(id);prior.push_back(l.constituents[id].grad);}
        m.flush(int(scale));
        for(int t=0;t<count;++t)for(int j=0;j<D;++j){double actual=l.constituents[ids[t]].grad[j]-prior[t][j];
            out[scale*2]=std::max(out[scale*2],std::abs(actual-predicted[t*D+j]));out[scale*2+1]=std::max(out[scale*2+1],std::abs(actual));}
    }
});}
