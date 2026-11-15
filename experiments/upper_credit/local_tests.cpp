#include "native.hpp"
#include <memory>
#include <iostream>
using namespace shoe;
int checks=0;
void check(bool value,const char* msg){++checks;if(!value)throw std::runtime_error(msg);}
double objective(Model& m,const uint8_t* x,const uint8_t* y,int n,int scale){
    double result=0;int uses=0;
    for(int i=0;i<n;++i){m.observe(x[i]);auto& l=m.levels[scale];
        if(l.s.count){++uses;const auto& trace=l.local_language.events[l.s.count-1];int node=0;
            for(int depth=0;depth<8;++depth){int bit=(y[i]>>(7-depth))&1;float z=trace.scores[node];result+=softplus(z)-bit*z;node=2*node+1+bit;}}
        m.teach(y[i]);}
    return result/uses;
}
int main(){
    uint8_t x[512],y[512];for(int i=0;i<512;++i){x[i]=(i*17+i/13)%251;y[i]=(i*19+i/9)%251;}
    for(int scale:{2,3}){
        int n=scale==2?127:511;
        auto base=std::make_unique<Model>(4,5000000,7,false,false);base->routing_arm=2;base->balance_policy=2;base->semantic_laces=false;
        base->language_credit={{0,0,0,0}};base->local_language_weight[scale]=1;
        base->head_rate=base->learning_rate=base->slow_rate=base->query_rate=base->route_rate=base->aux_weight=base->reconstruction_weight=0;
        auto trained=std::make_unique<Model>(*base);objective(*trained,x,y,n,scale);
        int selected=trained->levels[scale].constituents[0].expert;auto original=trained->levels[scale].experts[selected];
        trained->slow_rate=.01f;trained->finish();double signal=0;
        for(int kind:{0,1})for(int index:{0,3,13,33,63}){
            auto plus=std::make_unique<Model>(*base),minus=std::make_unique<Model>(*base);
            auto param=[&](Model& m)->float&{auto& e=m.levels[scale].experts[selected];return kind?e.bias[index]:e.w[index];};
            float eps=.01f;param(*plus)+=eps;param(*minus)-=eps;
            double numeric=(objective(*plus,x,y,n,scale)-objective(*minus,x,y,n,scale))/(2*eps);
            double analytic=((kind?original.bias[index]:original.w[index])-param(*trained))/.01;
            check(std::abs(numeric-analytic)<4e-5,"local byte objective does not reach experts correctly");signal+=analytic*analytic;
        }
        check(signal>1e-8,"local byte expert gradient absent");
        trained->run(x,y,333);auto raw=save(*trained);auto restored=std::make_unique<Model>(load(raw.data(),raw.size()));
        check(save(*restored)==raw,"local head checkpoint changed");
        check(trained->run(x,y,512)==restored->run(x,y,512)&&save(*trained)==save(*restored),"local head continuation changed");
    }
    std::cout<<checks<<" local language checks passed\n";
}
