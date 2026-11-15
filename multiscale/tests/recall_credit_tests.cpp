#include "../native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
int checks=0;
void check(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
double objective(const Vec& a,const Vec& b,const RecallKey& gradient){
    double result=0;
    for(int j=0;j<RECALL_K;++j){double value=0;
        for(int k=0;k<4;++k){int i=j+RECALL_K*k;value+=(k%2?-.5:.5)*std::tanh(.5*(double(a[i])+b[(3*i+13)%D]));}
        result+=std::tanh(value)*gradient[j];}
    return result;
}
int main(){
    Vec a{},b{};RecallCreditUnit unit;
    for(int i=0;i<D;++i){a[i]=.3f*std::sin(float(i));b[i]=.2f*std::cos(float(i));}
    for(int i=0;i<D;++i)unit.combined[i]=std::tanh(.5f*(a[i]+b[(3*i+13)%D]));
    unit.value=recall_pool(unit.combined);for(int j=0;j<RECALL_K;++j){unit.value[j]=std::tanh(unit.value[j]);unit.gradient[j]=.2f*std::sin(float(j));}
    Vec da{},db{};recall_value_vjp(unit,da,db);
    for(int kind=0;kind<2;++kind)for(int i=0;i<D;++i){Vec plus=kind?b:a,minus=plus;float eps=.001f;plus[i]+=eps;minus[i]-=eps;
        double n=kind?(objective(a,plus,unit.gradient)-objective(a,minus,unit.gradient))/(2*eps):(objective(plus,b,unit.gradient)-objective(minus,b,unit.gradient))/(2*eps);
        check(std::abs(n-(kind?db[i]:da[i]))<2e-6,"ordered-pair value VJP mismatch");}
    std::vector<uint8_t> x(512),y(512);for(int i=0;i<512;++i){x[i]=(i*13+i/17)%251;y[i]=(i*23+i/11)%251;}
    for(int scale:{2,3}){
        auto base=std::make_unique<Model>(4,1000000,7,false,false);
        base->semantic_laces=false;base->routing_arm=2;base->balance_policy=2;base->recall_credit[scale]=1;
        base->head_rate=base->learning_rate=base->slow_rate=base->query_rate=base->route_rate=base->aux_weight=base->reconstruction_weight=0;
        for(auto& l:base->levels){l.p.gain={};l.p.bias={};l.retrieval.query={};l.retrieval.query_byte={};if(&l!=&base->levels[scale])l.retrieval.read={};}
        for(auto& d:base->decoder)d={};
        auto trained=std::make_unique<Model>(*base);trained->run(x.data(),y.data(),512);
        int selected=trained->levels[scale].constituents[0].expert;
        auto before=trained->levels[scale].experts[selected];trained->slow_rate=.01f;trained->finish();
        double signal=0;
        for(int kind:{0,1})for(int index:{0,3,13,33,63}){
            auto plus=std::make_unique<Model>(*base),minus=std::make_unique<Model>(*base);
            auto parameter=[&](Model& m)->float&{auto& e=m.levels[scale].experts[selected];return kind?e.bias[index]:e.w[index];};
            float eps=.01f;parameter(*plus)+=eps;parameter(*minus)-=eps;
            double n=(plus->run(x.data(),y.data(),512)-minus->run(x.data(),y.data(),512))/(2*eps);
            float original=kind?before.bias[index]:before.w[index];double actual=(original-parameter(*trained))/.01;
            signal+=actual*actual;check(std::abs(n-actual)<4e-5,"end-to-end retained-value expert gradient mismatch");
        }
        check(signal>1e-10,"retained value loss did not train its producer");
        for(auto& unit:trained->levels[scale].recall_credit_state.units)check(unit.current==-1&&unit.previous==-1,"retired producer alias remained live");
        check(trained->levels[scale].recall_credit_state.previous==-1&&trained->levels[scale].recall_content_gradient.empty(),"retirement left credit buffers live");
        trained->run(x.data(),y.data(),333);auto raw=save(*trained);auto restored=std::make_unique<Model>(load(raw.data(),raw.size()));
        check(save(*restored)==raw,"value credit checkpoint changed pending state");
        check(trained->run(x.data(),y.data(),512)==restored->run(x.data(),y.data(),512)&&save(*trained)==save(*restored),"value credit continuation changed");
        trained->inference(false);raw=save(*trained);restored=std::make_unique<Model>(load(raw.data(),raw.size()));
        check(save(*restored)==raw,"frozen value-credit checkpoint failed");
    }
    std::cout<<checks<<" retained value credit checks passed\n";
}
