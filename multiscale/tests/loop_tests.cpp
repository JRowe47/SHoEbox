#include "../native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
int checks=0;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);++checks;}

// Independent double-precision oracle for base expert + tied refinement.
double objective(const Expert& e,const Vec& input,float gate,const Vec& gradient,int steps,float strength){
    std::array<double,D> anchor{},h{},next{};
    for(int i=0;i<D;++i){double z=input[i]+e.bias[i];
        for(int j=0;j<D;++j)z+=double(gate)*e.w[i*D+j]*input[j];
        anchor[i]=h[i]=std::tanh(z);}
    for(int k=0;k<steps;++k){
        for(int i=0;i<D;++i){double z=anchor[i]+e.bias[i];
            for(int j=0;j<D;++j)z+=double(gate)*e.w[i*D+j]*h[j];
            next[i]=(1-double(strength))*h[i]+strength*std::tanh(z);}
        h=next;
    }
    double loss=0;for(int i=0;i<D;++i)loss+=h[i]*gradient[i];return loss;
}
void derivatives(){
    Expert e;Vec x{},incoming{};
    for(int i=0;i<D;++i){x[i]=.25f*std::sin(i+.1f);incoming[i]=.1f*std::cos(i+.2f);e.bias[i]=.02f*std::cos(float(i));}
    for(int i=0;i<D*D;++i)e.w[i]=.025f*std::sin(i*.3f);
    for(int depth=1;depth<=MAX_LOOPS;++depth)for(float strength:{.25f,1.f}){
        const float gate=1.17f,eps=.001f;
        LoopPolicy p;p.steps[1]=depth;p.strength=strength;p.tolerance=0;
        Vec residual{},base{};mv(e.w,x,residual);
        for(int i=0;i<D;++i)base[i]=std::tanh(x[i]+e.bias[i]+gate*residual[i]);
        LoopTrace trace;LoopStats stats;Vec out=loop_forward(e,base,gate,p,1,&trace,stats);
        require(std::abs(dot(out.data(),incoming.data())-objective(e,x,gate,incoming,depth,strength))<1e-6,"forward oracle");
        Expert g;float dg=0;Vec dc=loop_backward(e,gate,p,trace,incoming,g,dg),dx{},pre{},gated{};
        for(int i=0;i<D;++i){pre[i]=dc[i]*(1-base[i]*base[i]);g.bias[i]+=pre[i];dx[i]=pre[i];gated[i]=pre[i]*gate;dg+=pre[i]*residual[i];}
        outer(g.w,gated,x);vjp(e.w,gated,dx);
        for(int i=0;i<D;++i){
            Vec a=x,b=x;a[i]+=eps;b[i]-=eps;
            double num=(objective(e,a,gate,incoming,depth,strength)-objective(e,b,gate,incoming,depth,strength))/(a[i]-b[i]);
            require(std::abs(num-dx[i])<2e-6,"anchor/input derivative");
            Expert ea=e,eb=e;ea.bias[i]+=eps;eb.bias[i]-=eps;
            num=(objective(ea,x,gate,incoming,depth,strength)-objective(eb,x,gate,incoming,depth,strength))/(ea.bias[i]-eb.bias[i]);
            require(std::abs(num-g.bias[i])<2e-6,"shared bias derivative");
        }
        for(int j=0;j<D*D;j+=97){
            Expert a=e,b=e;a.w[j]+=eps;b.w[j]-=eps;
            double num=(objective(a,x,gate,incoming,depth,strength)-objective(b,x,gate,incoming,depth,strength))/(a.w[j]-b.w[j]);
            require(std::abs(num-g.w[j])<2e-6,"shared expert derivative");
        }
        double num=(objective(e,x,gate+eps,incoming,depth,strength)-objective(e,x,gate-eps,incoming,depth,strength))/((gate+eps)-(gate-eps));
        require(std::abs(num-dg)<2e-6,"gate derivative through all depths");
    }
}
void slow_language_gradient(){
    auto base=std::make_unique<Model>(2,150000,7,false,false);base->semantic_laces=false;
    base->loop_policy.configured=true;base->loop_policy.steps[1]=3;base->loop_policy.tolerance=0;
    for(auto& l:base->levels)l.retrieval.read.fill(0);
    base->reconstruction_weight=base->route_rate=base->aux_weight=base->head_rate=0;
    base->learning_rate=0;
    std::vector<uint8_t> x(32),y(32);for(int i=0;i<32;++i){x[i]=i*7+13;y[i]=i*19+1;}
    auto a=std::make_unique<Model>(*base);a->run(x.data(),y.data(),32);
    int expert=a->levels[1].s.events[0].expert;
    auto before=std::make_unique<Model>(*a);a->slow_rate=.01f;a->finish();
    auto param=[&](Model& m,int kind,int i)->float&{auto& l=m.levels[1];
        if(kind==0)return l.experts[expert].w[i];if(kind==1)return l.experts[expert].bias[i];
        if(kind==2)return l.p.role[i];if(kind==3)return l.p.offset[i];return l.p.recurrent[i];};
    for(int kind=0;kind<5;++kind)for(int i:{0,7,23,41}){
        float eps=.003f;auto plus=std::make_unique<Model>(*base),minus=std::make_unique<Model>(*base);
        param(*plus,kind,i)+=eps;param(*minus,kind,i)-=eps;
        plus->training=minus->training=false;
        double num=(32./29)*(plus->run(x.data(),y.data(),32)-minus->run(x.data(),y.data(),32))/(2*eps);
        float grad=(param(*before,kind,i)-param(*a,kind,i))/.01f;
        require(std::abs(num-grad)<.0008,"integrated language credit");
    }
}
void boundaries_and_stopping(){
    LoopPolicy policy;policy.steps[1]=4;LoopStats stats;LoopTrace trace;
    Expert zero;Vec anchor{};
    require(loop_forward(zero,anchor,1,policy,1,&trace,stats)==anchor,"zero fixed point");
    require(trace.steps==1&&stats.converged==1&&stats.passes==1,"early convergence");
    policy.tolerance=0;loop_forward(zero,anchor,1,policy,1,&trace,stats);
    require(trace.steps==4&&stats.passes==5&&stats.budget_hits==1,"fixed depth");
    auto m=std::make_unique<Model>(4,250000,7,false,false);m->semantic_laces=false;
    m->loop_policy.configured=true;m->loop_policy.steps={{0,1,2,4}};m->loop_policy.tolerance=0;
    std::vector<uint8_t> data(1025);for(size_t i=0;i<data.size();++i)data[i]=(i*19)%256;
    m->run(data.data(),data.data()+1,511);
    auto checkpoint=save(*m);auto restored=std::make_unique<Model>(load(checkpoint.data(),checkpoint.size()));
    require(save(*restored)==checkpoint,"pending tape round trip");
    m->run(data.data()+511,data.data()+512,513);
    restored->run(data.data()+511,data.data()+512,513);
    require(save(*m)==save(*restored),"exact continuation through publication");
    for(int level=0,span=1;level<4;++level,span*=4){auto& l=m->levels[level];
        require(l.s.executions==1024/uint64_t(span),"real event count");
        if(level){require(l.s.addressed.clock.ticks==l.s.executions,"one clock tick per completion");
            require(l.loops.stats.events==l.s.executions,"one loop event per completion");
            require(l.loops.stats.passes==l.s.executions*m->loop_policy.steps[level],"per-scale budget");
            require(l.loops.traces.size()<=size_t(m->epoch_limit/span),"bounded eligibility tape");}
        else require(l.loops.stats.passes==0,"byte path never loops");
    }
    m->finish();for(auto& l:m->levels)require(l.loops.traces.empty(),"retirement clears tape");
    auto frozen=std::make_unique<Model>(*m);frozen->inference(false);
    frozen->run(data.data(),data.data()+1,1024);
    for(auto& l:frozen->levels)require(l.loops.traces.empty(),"inference has no training tape");
    auto saved=save(*frozen);require(save(load(saved.data(),saved.size()))==saved,"warm inference continuation");
    frozen->reset();for(auto& l:frozen->levels)require(l.loops.stats.events==0,"reset clears context statistics");
}
int main(){derivatives();slow_language_gradient();boundaries_and_stopping();std::cout<<checks<<" loop checks passed\n";}
