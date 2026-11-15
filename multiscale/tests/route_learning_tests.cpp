#include "../native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
int checks=0;
void check(bool ok,const char* message){++checks;if(!ok)throw std::runtime_error(message);}
template<class T> void same(const T& a,const T& b,const char* message){check(std::memcmp(&a,&b,sizeof(T))==0,message);}
int main(){
    RouteLearning s;s.configured=s.enabled=true;s.period=100;
    check(s.multiplier(0)==1&&s.multiplier(25)==1.5f&&s.multiplier(75)==.5f,"cycle extrema");
    check(s.multiplier(125)==s.multiplier(25),"cycle period");
    s.start=s.target=.6f;
    for(uint64_t k=0;k<1000;++k)check(s.multiplier(k)>=.5f&&s.multiplier(k)<=.7f+.000001f,"shifted bounds");
    s.start=s.target=.5f;check(s.multiplier(25)==.5f,"bound must collapse symmetric range");
    s.start=1;s.target=.7f;s.ramp_targets=100;
    for(int k=0;k<200;++k){s.advance();check(s.anchor()>=.7f&&s.anchor()<=1.f,"anchor ramp bounds");}
    check(s.anchor()==.7f&&s.ramp_elapsed==100,"ramp endpoint");

    auto base=std::make_unique<Model>(4,1000000,7,false,false);
    base->semantic_laces=false;base->routing_arm=2;
    base->learning_rate=base->slow_rate=base->query_rate=base->head_rate=base->route_rate=0;
    std::vector<uint8_t> x(511),y(511);for(int k=0;k<511;++k){x[k]=(k*17+k/7)%251;y[k]=(k*29+k/11)%251;}
    base->run(x.data(),y.data(),x.size());

    // Use identical accumulated gradients, then change only the optimizer.
    // Both fast transitions and delayed upper experts must get owner-specific
    // rates; shared updates and cross-level derivatives must remain identical.
    for(int target:{0,1,2,3}){
        auto plain=std::make_unique<Model>(*base),cycled=std::make_unique<Model>(*base);
        plain->learning_rate=cycled->learning_rate=.003f;plain->slow_rate=cycled->slow_rate=.003f;
        cycled->route_learning.configured=cycled->route_learning.enabled=true;cycled->route_learning.period=100;
        for(auto& l:cycled->levels)for(size_t k=0;k<l.lr_visits.size();++k)l.lr_visits[k]=k%2?75:25;
        if(target==0){plain->flush(0);cycled->flush(0);}
        else{for(int l=0;l<4;++l){plain->flush(l);cycled->flush(l);}
             plain->retire_constituents();cycled->retire_constituents();}
        auto& old=base->levels[target];auto& a=plain->levels[target];auto& b=cycled->levels[target];
        double movement=0;
        for(size_t k=0;k<a.experts.size();++k){double factor=k%2?.5:1.5;
            for(size_t j=0;j<a.experts[k].w.size();++j){
                double d=a.experts[k].w[j]-old.experts[k].w[j];movement+=d*d;
                check(std::abs((b.experts[k].w[j]-old.experts[k].w[j])-factor*d)<2e-8,"expert matrix LR ratio");}
            for(size_t j=0;j<D;++j){double d=a.experts[k].bias[j]-old.experts[k].bias[j];
                check(std::abs((b.experts[k].bias[j]-old.experts[k].bias[j])-factor*d)<2e-8,"expert bias LR ratio");}}
        check(movement>1e-14,"no measured expert update");same(a.p,b.p,"shared weights must use anchor only");
        same(a.address,b.address,"address weights must use anchor only");
        check(a.constituents.size()==b.constituents.size(),"constituent layout changed");
        for(size_t k=0;k<a.constituents.size();++k)check(a.constituents[k].grad==b.constituents[k].grad,"optimizer scaled propagated derivatives");
    }

    // Neutral cycles must reproduce all fitted parameters and stream state.
    auto a=std::make_unique<Model>(*base),b=std::make_unique<Model>(*base);
    a->learning_rate=b->learning_rate=.003f;a->slow_rate=b->slow_rate=.01f;
    a->head_rate=b->head_rate=.02f;a->query_rate=b->query_rate=.003f;a->route_rate=b->route_rate=.002f;
    b->route_learning.configured=b->route_learning.enabled=true;b->route_learning.amplitude=0;
    check(a->run(x.data(),y.data(),511)==b->run(x.data(),y.data(),511),"neutral cycle loss parity");
    b->route_learning.configured=false;
    check(save(*a)==save(*b),"neutral cycle parameter and pending-state parity");

    // A nonneutral constant anchor must scale every actual rate exactly once.
    a=std::make_unique<Model>(*base);b=std::make_unique<Model>(*base);
    a->learning_rate=.003f;a->slow_rate=.01f;a->query_rate=.003f;a->head_rate=.02f;a->route_rate=.002f;
    b->learning_rate=2*a->learning_rate;b->slow_rate=2*a->slow_rate;b->query_rate=2*a->query_rate;
    b->head_rate=2*a->head_rate;b->route_rate=2*a->route_rate;
    b->route_learning.configured=b->route_learning.enabled=true;b->route_learning.start=b->route_learning.target=.5f;
    check(a->run(x.data(),y.data(),511)==b->run(x.data(),y.data(),511),"all-family anchor rate parity");
    a->finish();b->finish();
    b->route_learning.configured=false;b->learning_rate=a->learning_rate;b->slow_rate=a->slow_rate;
    b->head_rate=a->head_rate;b->query_rate=a->query_rate;b->route_rate=a->route_rate;
    check(save(*a)==save(*b),"anchor must preserve objectives and gradient credit");
    std::cout<<checks<<" route learning checks passed\n";
}
