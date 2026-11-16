#include <iostream>
#include <cstdlib>
using namespace shoe;
int checks=0;
void require(bool ok,const char* what){++checks;if(!ok)throw std::runtime_error(what);}
float objective(const QueryHead& p,const QueryInput& q,const MemoryInput& m,int kind,const std::array<float,255>& e){
    auto c=query_forward(p,q,m,kind);float z=0;
    for(int n=0;n<255;++n)z+=e[n]*(p.bias[n]+dot(p.output.data()+n*QRANK,c.h.data(),QRANK));return z;
}
int main(){
    std::mt19937 rng(7);std::normal_distribution<float> normal(0,.1f);
    QueryHead p{};QueryInput q{};MemoryInput m{};std::array<float,255> e{};
    for(auto& x:p.query)x=normal(rng);for(auto& x:p.memory)x=normal(rng);
    for(auto& x:p.output)x=normal(rng);for(auto& x:p.query_bias)x=normal(rng);for(auto& x:p.memory_bias)x=normal(rng);
    for(auto& x:q)x=normal(rng);for(auto& x:m)x=normal(rng);for(int n:{0,1,4,10,22,46,94,190})e[n]=normal(rng);
    for(int kind:{1,2}){
        QueryHead g{};QueryInput dq{};MemoryInput dm{};query_vjp(p,query_forward(p,q,m,kind),kind,e,g,dq,dm);
        auto finite=[&](auto& xs,const auto& gs){for(size_t j=0;j<xs.size();j+=std::max(size_t(1),xs.size()/31)){
            float old=xs[j],eps=.001f;xs[j]=old+eps;float hi=objective(p,q,m,kind,e);xs[j]=old-eps;
            float lo=objective(p,q,m,kind,e);xs[j]=old;float numerical=(hi-lo)/(2*eps);
            require(std::abs(numerical-gs[j])<2e-5f+2e-3f*std::abs(gs[j]),"query finite difference");
        }};
        finite(p.query,g.query);finite(p.memory,g.memory);finite(p.query_bias,g.query_bias);finite(p.memory_bias,g.memory_bias);
        finite(p.output,g.output);finite(p.bias,g.bias);finite(q,dq);finite(m,dm);
    }
    // Exact public-runtime parity with no adapter in scope and with kind=0.
    for(bool minimal:{false,true}){
        QueryAdapter a(500000,19,minimal,0,true);Model control=*a.model;
        {QueryScope scope(a);for(int t=0;t<769;++t){uint8_t b=(t*37)%256,y=(t*37+11)%256;
            a.model->observe(b);float loss=a.model->teach(y);
            control.observe(b);require(loss==control.teach(y),"inactive adapter parity");
        }}require(save(*a.model)==save(control),"native serialization parity");
        QueryAdapter learned(500000,19,minimal,2,true);learned.head=p;
        {QueryScope scope(learned);for(int t=0;t<129;++t){learned.model->observe(t%256);learned.model->teach((t+9)%256);}}
        auto raw=query_save(learned);auto restored=query_load(raw.data(),raw.size());
        for(int t=0;t<213;++t){for(auto* a:{&learned,restored.get()}){QueryScope scope(*a);a->model->observe((t+31)%256);a->model->teach((t+40)%256);}}
        require(query_save(learned)==query_save(*restored),"partial-boundary continuation");
        QueryAdapter frozen(500000,19,minimal,2,false);frozen.head=p;
        Model observed=*frozen.model;
        for(int t=0;t<129;++t){QueryScope scope(frozen);frozen.model->observe(t%256);frozen.model->teach((t+9)%256);observed.observe(t%256);}
        require(save(*frozen.model)==save(observed),"head-only core freeze");
    }
    std::cout<<checks<<" query-readout numerical/lifecycle checks passed\n";
}
