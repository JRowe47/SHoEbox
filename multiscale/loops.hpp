#pragma once
// Refinement depth is separate from stream time. No extra routing, admission,
// memory writes, clocks or native prediction targets occur inside this cell.
constexpr int MAX_LOOPS=4;
struct LoopPolicy {
    bool configured=false;
    std::array<int,4> steps{{0,0,0,0}};
    float strength=.25f,tolerance=.01f;
    void validate(size_t levels)const{
        if(!std::isfinite(strength)||strength<=0||strength>1||
           !std::isfinite(tolerance)||tolerance<0||tolerance>1)
            throw std::invalid_argument("loop strength must be in (0,1] and tolerance in [0,1]");
        for(size_t i=0;i<steps.size();++i)
            if(steps[i]<0||steps[i]>MAX_LOOPS||((i==0||i>=levels)&&steps[i]))
                throw std::invalid_argument("loops require 0..4 extra passes per upper scale and zero byte-scale passes");
    }
};
struct LoopTrace {
    std::array<Vec,MAX_LOOPS+1> state{};
    std::array<Vec,MAX_LOOPS> proposal{};
    int steps=0;
};
struct LoopStats {
    uint64_t events=0,passes=0,converged=0,budget_hits=0;
    int last_steps=0;
    float last_relative_change=0;
};
struct LoopState {
    // Training only; bounded by the existing constituent publication epoch.
    std::vector<LoopTrace> traces;
    LoopStats stats{};
};

inline Vec loop_forward(const Expert& expert,const Vec& anchor,float gate,
                        const LoopPolicy& policy,int scale,LoopTrace* trace,LoopStats& stats){
    Vec state=anchor;
    if(trace){*trace=LoopTrace{};trace->state[0]=anchor;}
    ++stats.events;stats.last_steps=0;stats.last_relative_change=0;
    bool converged=false;
    for(int k=0;k<policy.steps[scale];++k){
        Vec product{},proposal{},next{};mv(expert.w,state,product);
        float change=0;
        for(int i=0;i<D;++i){
            proposal[i]=std::tanh(anchor[i]+expert.bias[i]+gate*product[i]);
            next[i]=(1-policy.strength)*state[i]+policy.strength*proposal[i];
            float delta=next[i]-state[i];change+=delta*delta;
        }
        stats.last_relative_change=std::sqrt(change/(energy(state)+D*1e-8f));
        ++stats.last_steps;++stats.passes;
        if(trace){trace->proposal[k]=proposal;trace->state[k+1]=next;++trace->steps;}
        state=next;
        if(policy.tolerance>0&&stats.last_relative_change<=policy.tolerance){converged=true;break;}
    }
    if(converged)++stats.converged;else ++stats.budget_hits;
    return state;
}

// Differentiate the executed path, treating the hard stopping decision and
// selected expert identity as fixed, like the existing hard routing decision.
// Expert weights remain frozen until every parent has supplied its credit.
inline Vec loop_backward(const Expert& expert,float gate,const LoopPolicy& policy,
                         const LoopTrace& trace,const Vec& incoming,Expert& gradient,float& dgate){
    Vec state_grad=incoming,anchor_grad{};
    for(int k=trace.steps-1;k>=0;--k){
        Vec pre{},gated{},previous{},product{};mv(expert.w,trace.state[k],product);
        for(int i=0;i<D;++i){
            pre[i]=policy.strength*state_grad[i]*(1-trace.proposal[k][i]*trace.proposal[k][i]);
            anchor_grad[i]+=pre[i];gradient.bias[i]+=pre[i];
            gated[i]=gate*pre[i];previous[i]=(1-policy.strength)*state_grad[i];
            dgate+=pre[i]*product[i];
        }
        outer(gradient.w,gated,trace.state[k]);vjp(expert.w,gated,previous);
        state_grad=previous;
    }
    for(int i=0;i<D;++i)anchor_grad[i]+=state_grad[i];
    return anchor_grad;
}
