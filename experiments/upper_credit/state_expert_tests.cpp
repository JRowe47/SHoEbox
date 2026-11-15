#include "native.hpp"
#include <memory>
#include <iostream>
using namespace shoe;
int checks=0;
void check(bool value,const char* msg){++checks;if(!value)throw std::runtime_error(msg);}
int main(){
    uint8_t x[512],y[512];for(int i=0;i<512;++i){x[i]=(i*13+i/9)%251;y[i]=(i*19+i/7)%251;}
    auto base=std::make_unique<Model>(4,1000000,7,false,false);base->routing_arm=2;base->balance_policy=2;base->semantic_laces=false;
    base->learning_rate=base->slow_rate=base->head_rate=base->query_rate=base->route_rate=0;
    base->run(x,y,511);auto amplified=std::make_unique<Model>(*base);
    amplified->state_expert_gain={{1,1,4,16}};
    base->learning_rate=amplified->learning_rate=.001f;base->slow_rate=amplified->slow_rate=.01f;
    base->finish();amplified->finish();
    check(base->embedding==amplified->embedding,"expert amplification changed the shared byte representation");
    for(int scale=0;scale<4;++scale){
        auto& a=base->levels[scale];auto& b=amplified->levels[scale];
        check(std::memcmp(&a.p,&b.p,sizeof(Shared))==0,"expert amplification changed shared composer/state/head updates");
        check(std::memcmp(&a.address,&b.address,sizeof(AddressParameters))==0,"expert amplification changed address/reconstruction updates");
        bool changed=false;for(size_t k=0;k<a.experts.size();++k){
            check(a.experts[k].key==b.experts[k].key,"expert amplification changed routing keys");
            changed|=a.experts[k].w!=b.experts[k].w||a.experts[k].bias!=b.experts[k].bias;}
        check(changed==(scale>=2),"expert amplification touched the wrong bank");
    }
    for(int scale:{2,3}){
        int n=scale==2?127:511,span=scale==2?16:64;
        auto reference=std::make_unique<Model>(scale+1,1000000,7,false,false);
        reference->routing_arm=2;reference->balance_policy=2;reference->semantic_laces=false;
        reference->learning_rate=reference->slow_rate=reference->head_rate=reference->query_rate=reference->route_rate=reference->aux_weight=reference->reconstruction_weight=0;
        for(auto& l:reference->levels)l.retrieval.read={};
        auto trained=std::make_unique<Model>(*reference);trained->state_expert_gain[scale]=4;
        trained->run(x,y,n);int selected=trained->levels[scale].constituents[0].expert;
        auto original=trained->levels[scale].experts[selected];trained->slow_rate=.01f;trained->finish();double signal=0;
        for(int kind:{0,1})for(int index:{0,3,13,33,63}){
            auto plus=std::make_unique<Model>(*reference),minus=std::make_unique<Model>(*reference);
            auto param=[&](Model& m)->float&{auto& e=m.levels[scale].experts[selected];return kind?e.bias[index]:e.w[index];};
            float eps=.02f;param(*plus)+=eps;param(*minus)-=eps;
            double numeric=4.*n/(n-span+1)*(plus->run(x,y,n)-minus->run(x,y,n))/(2*eps);
            double analytic=((kind?original.bias[index]:original.w[index])-param(*trained))/.01;
            check(std::abs(numeric-analytic)<6e-5,"amplified expert gradient disagrees with the primary byte objective");signal+=analytic*analytic;
        }
        check(signal>1e-12,"main-language expert gradient absent");
        trained->run(x,y,333);auto raw=save(*trained);auto restored=std::make_unique<Model>(load(raw.data(),raw.size()));
        check(save(*restored)==raw,"expert-state checkpoint changed");
        check(trained->run(x,y,512)==restored->run(x,y,512)&&save(*trained)==save(*restored),"expert-state continuation changed");
    }
    std::cout<<checks<<" isolated state-language checks passed\n";
}
