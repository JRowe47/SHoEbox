#include <iostream>
using namespace shoe;
int checks=0;
void require(bool yes,const char* name){++checks;if(!yes)throw std::runtime_error(name);}
float objective(const QueryHead& p,const std::array<Vec,5>& raw,const std::array<Normalizer,5>& norm,
 const AccessSources& src,bool pool,const std::array<float,255>& e,const Labels& labels,float aux){
    auto c=query_forward(p,raw,norm,src,pool);QueryHead g{};std::array<Vec,5> dx{};
    float result=query_vjp(p,c,{},labels,aux,g,dx);
    for(int n=0;n<255;++n)result+=e[n]*(p.bias[n]+dot(p.output.data()+n*QRANK,c.h.data(),QRANK));
    return result;
}
int main(){
    std::mt19937 rng(71);std::normal_distribution<float> normal(0,.15f);
    QueryHead p{};std::array<Vec,5> raw{};std::array<Normalizer,5> norm{};
    for(auto& n:norm){n.count=2;n.m2.fill(1.3);}
    for(auto& v:p.projection)v=normal(rng);for(auto& v:p.offset)v=normal(rng);
    for(auto& v:p.output)v=normal(rng);for(auto& v:p.bias)v=normal(rng);
    for(auto& x:raw)for(auto& v:x)v=normal(rng);
    AccessSources src;src.count=7;for(auto& x:src.x)for(auto& v:x)v=normal(rng);
    std::array<float,255> e{};for(int n:{0,1,4,10,22,46,94,190})e[n]=normal(rng);
    Labels labels{{32,41,32,130,131}};
    for(bool pool:{false,true})for(float aux:{0.f,.1f}){
        QueryHead g{};std::array<Vec,5> dx{};
        query_vjp(p,query_forward(p,raw,norm,src,pool),e,labels,aux,g,dx);
        auto finite=[&](auto& xs,const auto& gs,int stride){for(size_t j=0;j<xs.size();j+=stride){
            float old=xs[j],eps=.002f;xs[j]=old+eps;float hi=objective(p,raw,norm,src,pool,e,labels,aux);
            xs[j]=old-eps;float lo=objective(p,raw,norm,src,pool,e,labels,aux);xs[j]=old;
            require(std::abs((hi-lo)/(2*eps)-gs[j])<.0005f+.008f*std::abs(gs[j]),"access finite difference");
        }};
        finite(p.projection,g.projection,17);finite(p.offset,g.offset,1);finite(p.output,g.output,31);finite(p.bias,g.bias,5);
        if(!pool)for(int lane=0;lane<5;++lane)finite(raw[lane],dx[lane],1);
    }
    for(bool minimal:{true,false}){
        QueryAdapter native(500000,7,minimal,0,true);Model control=*native.model;
        for(int t=0;t<769;++t){QueryScope scope(native);native.model->observe(t%256);control.observe(t%256);
            require(native.model->teach((t+11)%256)==control.teach((t+11)%256),"native parity");}
        require(save(*native.model)==save(control),"native checkpoint parity");
        for(int kind:{1,2,3}){
            QueryAdapter a(500000,7,minimal,kind,false);Model observed=*a.model;
            for(int t=0;t<131;++t){QueryScope scope(a);a.model->observe(t%256);a.labels=labels;a.model->teach((t+11)%256);observed.observe(t%256);}
            require(save(*a.model)==save(observed),"frozen core remains byte exact");
            auto data=query_save(a);auto b=query_load(data.data(),data.size());
            for(int t=0;t<213;++t)for(auto* c:{&a,b.get()}){QueryScope scope(*c);c->model->observe((t+41)%256);c->labels=labels;c->model->teach((t+77)%256);}
            require(query_save(a)==query_save(*b),"partial continuation exact");
            for(float v:a.head.projection)require(std::isfinite(v),"finite weight");
        }
    }
    // Invalid/stale child cells must never be consulted by either view.
    Model m(4,500000,7,false,false,true);m.inference(false);
    for(int t=0;t<257;++t){m.observe(uint8_t(t));auto before=provisional(m);auto sources=access_sources(m);
        Model poisoned=m;
        for(auto& l:poisoned.levels)for(int k=l.s.children_count;k<GROUP;++k){l.s.children[k].fill(1000);l.s.stable_children[k].fill(1000);l.s.lengths[k]=1000;}
        require(before==provisional(poisoned),"stale children excluded");
        auto other=access_sources(poisoned);require(sources.x==other.x&&sources.count==other.count,"stale stable children excluded");
        require(sources.count<=ACCESS_SOURCES,"bounded source count");
        auto saved=save(m);provisional(m);access_sources(m);require(save(m)==saved,"read views do not mutate core");
    }
    // Exercise the per-query C ABI under sanitizers, including all source
    // lesions and empty-source normalizers after reset.
    for(int kind:{1,2,3})for(int lesion=0;lesion<=8;++lesion){
        QueryAdapter a(500000,7,true,kind,false);a.training=false;a.lesion=lesion;
        const uint8_t data[]={2,16,3,40,4,128,5,7,40,4,8,128,5,7,41,4,8,129,5};
        uint8_t mask[sizeof(data)]{};mask[11]=mask[17]=1;double result[9]{};
        require(qr_score(&a,data,mask,sizeof(data),0,256,2,result)==0,"score ABI");
        require(result[0]==2&&result[6]==result[7]+result[8],"per-query scores sum");
        for(double x:result)require(std::isfinite(x),"finite lesion scores");
    }
    std::cout<<checks<<" query-access numerical/lifecycle checks passed\n";
}
