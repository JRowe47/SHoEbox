#include <iostream>
using namespace shoe;
int checks=0;
void require(bool yes,const char* name){++checks;if(!yes)throw std::runtime_error(name);}
float objective(const QueryHead& p,const Vec& x,const Normalizer& norm,const std::array<float,255>& e,const Labels& labels,float aux){
    auto c=query_forward(p,x,norm);QueryHead g{};Vec dx{};float value=query_vjp(p,c,{},labels,aux,g,dx);
    for(int n=0;n<255;++n)value+=e[n]*(p.bias[n]+dot(p.output.data()+n*QRANK,c.h.data(),QRANK));return value;
}
int main(){
    std::mt19937 rng(71);std::normal_distribution<float> normal(0,.15f);
    QueryHead p{};Vec x{};Normalizer norm;norm.count=2;norm.m2.fill(1.3);
    for(auto& v:p.projection)v=normal(rng);for(auto& v:p.offset)v=normal(rng);
    for(auto& v:p.output)v=normal(rng);for(auto& v:p.bias)v=normal(rng);for(auto& v:x)v=normal(rng);
    std::array<float,255> e{};for(int n:{0,1,4,10,22,46,94,190})e[n]=normal(rng);
    Labels labels{{32,41,32,130,131}};
    for(float aux:{0.f,.1f}){
        QueryHead g{};Vec dx{};query_vjp(p,query_forward(p,x,norm),e,labels,aux,g,dx);
        auto finite=[&](auto& xs,const auto& gs){for(size_t j=0;j<xs.size();j+=std::max(size_t(1),xs.size()/113)){
            float old=xs[j],eps=.002f;xs[j]=old+eps;float hi=objective(p,x,norm,e,labels,aux);
            xs[j]=old-eps;float lo=objective(p,x,norm,e,labels,aux);xs[j]=old;
            require(std::abs((hi-lo)/(2*eps)-gs[j])<.0004f+.006f*std::abs(gs[j]),"factorized finite difference");
        }};
        finite(p.projection,g.projection);finite(p.offset,g.offset);finite(p.output,g.output);finite(p.bias,g.bias);finite(x,dx);
    }
    for(bool minimal:{true,false}){
        QueryAdapter a(500000,7,minimal,0,true);Model control=*a.model;
        for(int t=0;t<769;++t){QueryScope scope(a);a.model->observe(t%256);control.observe(t%256);
            require(a.model->teach((t+11)%256)==control.teach((t+11)%256),"native parity");}
        require(save(*a.model)==save(control),"native checkpoint parity");
        for(bool joint:{true,false}){
            QueryAdapter a(500000,7,minimal,2,joint);Model observed=*a.model;
            for(int t=0;t<131;++t){QueryScope scope(a);a.model->observe(t%256);a.labels=labels;a.model->teach((t+11)%256);observed.observe(t%256);}
            if(!joint)require(save(*a.model)==save(observed),"frozen-core exactness");
            auto raw=query_save(a);auto b=query_load(raw.data(),raw.size());
            for(int t=0;t<213;++t)for(auto* m:{&a,b.get()}){QueryScope scope(*m);m->model->observe((t+41)%256);m->labels=labels;m->model->teach((t+77)%256);}
            require(query_save(a)==query_save(*b),"partial continuation including normalizer");
            for(float value:a.head.projection)require(std::isfinite(value),"finite online weights");
        }
        QueryAdapter online(500000,7,minimal,3,false);
        for(int t=0;t<131;++t){QueryScope scope(online);online.model->observe(t%256);online.labels=labels;online.model->teach((t+11)%256);}
        auto raw=query_save(online);auto restored=query_load(raw.data(),raw.size());
        for(int t=0;t<79;++t)for(auto* a:{&online,restored.get()}){QueryScope scope(*a);a->model->observe((t+91)%256);a->labels=labels;a->model->teach((t+21)%256);}
        require(query_save(online)==query_save(*restored),"RLS exact continuation");
    }
    // Independent inverse identity: P * (lambda I + sum xx^T) == I.
    AddressRLS rls;QueryHead weights{};constexpr int N=AddressRLS::N;
    std::array<double,N*N> gram{};for(int j=0;j<N;++j)gram[j*N+j]=1e-6;
    for(int t=0;t<192;++t){Vec raw{};for(auto& v:raw)v=normal(rng);rls.update(weights,raw,labels);
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)gram[i*N+j]+=(i<D?raw[i]:1.)*(j<D?raw[j]:1.);}
    for(int i=0;i<N;++i)for(int j=0;j<N;++j){double product=0;
        for(int k=0;k<N;++k)product+=rls.inverse[i*N+k]*gram[k*N+j];
        require(std::abs(product-(i==j))<1e-6,"online inverse covariance identity");}
    std::cout<<checks<<" factorized numerical/lifecycle checks passed\n";
}
