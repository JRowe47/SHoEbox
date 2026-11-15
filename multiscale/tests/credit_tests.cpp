#include "../native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
int checks=0;
void check(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
int main(){
    auto base=std::make_unique<Model>(4,1000000,19,false,false);
    base->routing_arm=2;base->balance_policy=2;base->semantic_laces=false;
    base->learning_rate=base->slow_rate=base->query_rate=base->head_rate=base->route_rate=0;
    std::vector<uint8_t> x(511),y(511);
    for(int i=0;i<511;++i){x[i]=(i*13+i/7)%251;y[i]=(i*23+i/11)%251;}
    base->run(x.data(),y.data(),x.size());
    for(int scale:{2,3}){
        auto zero=std::make_unique<Model>(*base),one=std::make_unique<Model>(*base),four=std::make_unique<Model>(*base);
        zero->language_credit[scale]=0;four->language_credit[scale]=4;
        zero->slow_rate=one->slow_rate=four->slow_rate=.01f;
        zero->flush(scale);one->flush(scale);four->flush(scale);
        const auto& a=zero->levels[scale];const auto& b=one->levels[scale];const auto& c=four->levels[scale];
        check(a.p.gain==b.p.gain&&b.p.gain==c.p.gain,"language gain changed readout-gain update");
        check(a.p.bias==b.p.bias&&b.p.bias==c.p.bias,"language gain changed readout-bias update");
        check(a.p.predict==b.p.predict&&b.p.predict==c.p.predict,"language gain changed auxiliary head update");
        check(a.address.reconstruct==b.address.reconstruct&&b.address.reconstruct==c.address.reconstruct,"language gain changed reconstruction head update");
        double signal=0;
        for(size_t k=0;k<a.constituents.size();++k)for(int j=0;j<D;++j){
            double g1=b.constituents[k].grad[j]-a.constituents[k].grad[j];
            double g4=c.constituents[k].grad[j]-a.constituents[k].grad[j];
            signal+=g1*g1;check(std::abs(g4-4*g1)<1e-5,"language content VJP is not independently scaled");
        }
        check(signal>1e-10,"upper-scale language credit absent");
        auto aux1=std::make_unique<Model>(*base),aux4=std::make_unique<Model>(*base);
        aux4->language_credit[scale]=4;aux1->slow_rate=aux4->slow_rate=.01f;
        for(auto* m:{aux1.get(),aux4.get()})for(auto& e:m->levels[scale].s.events)e.task=e.gain_grad=e.bias_grad=Vec{};
        aux1->flush(scale);aux4->flush(scale);
        check(aux1->levels[scale].p.recurrent==aux4->levels[scale].p.recurrent,"auxiliary state update was scaled");
        for(size_t k=0;k<aux1->levels[scale].constituents.size();++k)
            check(aux1->levels[scale].constituents[k].grad==aux4->levels[scale].constituents[k].grad,"auxiliary content update was scaled");
    }
    for(int arm:{0,1,2,3})for(int policy:{0,1,2}){
        auto m=std::make_unique<Model>(*base);m->routing_arm=arm;m->balance_policy=policy;m->language_credit={{1,1,4,16}};
        auto raw=save(*m);auto q=std::make_unique<Model>(load(raw.data(),raw.size()));
        check(q->language_credit==m->language_credit&&save(*q)==raw,"language gains or old payload policy lost on reload");
        check(m->run(x.data(),y.data(),511)==q->run(x.data(),y.data(),511)&&save(*m)==save(*q),"pending-credit continuation changed");
    }
    std::cout<<checks<<" language credit checks passed\n";
}
