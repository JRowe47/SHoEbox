#include "../native.hpp"
#include <iostream>
#include <functional>
#include <memory>
using namespace shoe;
static int checks=0;
static void check(bool ok,const char* message){++checks;if(!ok)throw std::runtime_error(message);}
static std::array<Vec,GROUP> children(int t){
    std::array<Vec,GROUP> x{};
    for(int k=0;k<GROUP;++k)for(int i=0;i<D;++i)x[k][i]=.25f*std::sin(float(i+3*k+7*t));
    return x;
}
static void inputs(Model& m){
    for(int t=0;t<3;++t)m.forward_event(2,children(t),{{4,4,4,4}},0);
}
static float objective(const Model& original){auto owned=std::make_unique<Model>(original);auto& m=*owned;inputs(m);float z=0;
    for(int i=0;i<D;++i)z+=m.levels[2].s.state[i]*.025f*std::cos(float(i));return z;
}
int main(){
    auto base_owned=std::make_unique<Model>(3,250000,7,false,false,true);auto& base=*base_owned;
    base.reconstruction_weight=0;base.aux_weight=0;base.slow_rate=.01f;
    for(auto pick:std::vector<std::function<float&(Model&)>>{
        [](Model& m)->float&{return m.levels[2].p.role[17];},
        [](Model& m)->float&{return m.levels[2].p.offset[3];},
        [](Model& m)->float&{return m.levels[2].p.length[3];},
        [](Model& m)->float&{return m.levels[2].p.recurrent[3*D+3];},
        [](Model& m)->float&{return m.levels[2].p.read[3*K+2];},
        [](Model& m)->float&{return m.levels[2].address.read[3*ADDRESS_V+2];}}){
        auto pa=std::make_unique<Model>(base),pb=std::make_unique<Model>(base),pt=std::make_unique<Model>(base);
        auto& a=*pa;auto& b=*pb;auto& trained=*pt;float eps=.002f;pick(a)+=eps;pick(b)-=eps;
        float numeric=(objective(a)-objective(b))/(2*eps);
        float prior=pick(trained);inputs(trained);
        for(int i=0;i<D;++i)trained.levels[2].s.events[2].task[i]=.025f*std::cos(float(i));
        trained.levels[2].s.events[2].uses=1;trained.flush(2);trained.retire_constituents();
        float analytic=(prior-pick(trained))/base.slow_rate;
        check(std::abs(numeric-analytic)<.001f,"summary gradient mismatch");
    }
    auto pa=std::make_unique<Model>(4,500000,7,false,false,true),pb=std::make_unique<Model>(*pa);
    auto& a=*pa;auto& b=*pb;
    // Poison every removed dense operation; it must not affect execution.
    for(int s=2;s<4;++s){auto& p=b.levels[s].p;
        p.predict.fill(std::numeric_limits<float>::quiet_NaN());
        for(int i=0;i<D;++i)for(int j=0;j<D;++j)if(i!=j)p.recurrent[i*D+j]=std::numeric_limits<float>::quiet_NaN();
        for(int i=D;i<D*D;++i)p.gain[i]=p.bias[i]=std::numeric_limits<float>::quiet_NaN();
    }
    for(int i=0;i<700;++i){a.observe(uint8_t(i));b.observe(uint8_t(i));check(a.teach(uint8_t(i+1))==b.teach(uint8_t(i+1)),"removed dense operation used");}
    a.finish();b.finish();check(a.levels[3].s.state==b.levels[3].s.state,"summary state diverged");
    check(a.levels[2].experts.empty()&&a.levels[3].experts.empty(),"upper expert bank allocated");
    check(a.levels[2].routing.seen==0&&a.levels[3].routing.seen==0,"summary routed");
    check(a.levels[3].s.reconstruction_targets>0,"summary reconstruction credit missing");
    auto saved=save(a);auto resumed=load(saved.data(),saved.size());check(save(resumed)==saved,"v19 roundtrip");
    check(a.levels[3].s.addressed.clock.ticks==a.levels[3].s.executions,"summary clock mismatch");
    a.loop_policy.configured=true;
    auto invalid=save(a);bool rejected=false;
    try{load(invalid.data(),invalid.size());}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"minimal payload accepted inside a refinement wrapper");
    std::cout<<"PASS "<<checks<<" minimal native checks\n";
}
