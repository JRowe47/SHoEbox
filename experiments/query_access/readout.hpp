#pragma once
#include "views.hpp"
#include <memory>

namespace shoe {
constexpr int QRANK=16, QINPUT=D, MINPUT=0;
using LowRank=std::array<float,QRANK>;
using Labels=std::array<int16_t,5>; // observed query, keys 0/1, values 0/1; -1 absent
struct QueryHead {
    std::array<float,5*QRANK*D> projection{};
    std::array<float,5*QRANK> offset{};
    std::array<float,255*QRANK> output{};
    std::array<float,255> bias{};
};
constexpr int QPARAMS=sizeof(QueryHead)/sizeof(float);
struct QueryCache {
    std::array<Vec,5> raw{},x{},scale{};
    std::array<LowRank,5> z{};LowRank h{};float attention=.5f;
    AccessSources sources{};
    std::array<std::array<float,ACCESS_SOURCES>,5> weights{};
    bool pooled=false;
};
struct Normalizer {
    uint64_t count=0;
    std::array<double,D> mean{},m2{};
    void update(const Vec& x){
        ++count;for(int j=0;j<D;++j){double delta=x[j]-mean[j];mean[j]+=delta/count;m2[j]+=delta*(x[j]-mean[j]);}
    }
};

// EXP01-B decoder retained exactly; only its input views change.
inline QueryCache query_forward(const QueryHead& p,const std::array<Vec,5>& raw,
    const std::array<Normalizer,5>& norm,const AccessSources& sources={},bool pooled=false,int lesion=0){
    QueryCache c;c.raw=raw;c.sources=sources;c.pooled=pooled;
    if(pooled)for(int lane=0;lane<5;++lane){
        float maximum=-1e30f,total=0;
        for(int k=0;k<sources.count;++k){
            c.weights[lane][k]=lesion==8?0:dot(p.projection.data()+lane*QRANK*D,sources.x[k].data());
            maximum=std::max(maximum,c.weights[lane][k]);}
        for(int k=0;k<sources.count;++k){c.weights[lane][k]=std::exp(c.weights[lane][k]-maximum);total+=c.weights[lane][k];}
        c.raw[lane]={};
        for(int k=0;k<sources.count;++k){c.weights[lane][k]/=total;
            for(int j=0;j<D;++j)c.raw[lane][j]+=c.weights[lane][k]*sources.x[k][j];}
    }
    for(int lane=0;lane<5;++lane){
        for(int j=0;j<D;++j){c.scale[lane][j]=norm[lane].count>1?float(1/std::sqrt(std::max(norm[lane].m2[j]/(norm[lane].count-1),1e-6))):1;
            c.x[lane][j]=(c.raw[lane][j]-float(norm[lane].mean[j]))*c.scale[lane][j];}
        if(lesion==3)c.x[lane]={};
        for(int r=0;r<QRANK;++r){int row=lane*QRANK+r;
            c.z[lane][r]=std::tanh(p.offset[row]+dot(p.projection.data()+row*D,c.x[lane].data()));}
    }
    if(lesion==2)c.z[0]={};
    float score=0;for(int r=0;r<QRANK;++r)score+=c.z[0][r]*(c.z[1][r]-c.z[2][r]);
    c.attention=sigmoid(score);
    if(lesion==4)c.attention=.5f;
    if(lesion==5)c.attention=1-c.attention;
    for(int r=0;r<QRANK;++r)c.h[r]=c.attention*c.z[3][r]+(1-c.attention)*c.z[4][r];
    return c;
}

// A fixed, label-only byte code provides a common online reconstruction target
// for the three address projections. It is never consulted by query_forward.
// Hash codes are bounded and reproducible; they do not encode record identity.
inline float byte_code(int byte,int coordinate){
    uint32_t x=uint32_t(byte+1)*0x9e3779b9u+uint32_t(coordinate+1)*0x85ebca6bu;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    return x&1?.7f:-.7f;
}
inline float query_vjp(const QueryHead& p,const QueryCache& c,const std::array<float,255>& errors,
                       const Labels& labels,float auxiliary,QueryHead& g,std::array<Vec,5>& dx){
    std::array<LowRank,5> dz{};LowRank dh{};float loss=0;
    for(int n=0;n<255;++n)if(errors[n]){g.bias[n]+=errors[n];
        for(int r=0;r<QRANK;++r){g.output[n*QRANK+r]+=errors[n]*c.h[r];dh[r]+=errors[n]*p.output[n*QRANK+r];}}
    float da=0;for(int r=0;r<QRANK;++r){da+=dh[r]*(c.z[3][r]-c.z[4][r]);
        dz[3][r]+=c.attention*dh[r];dz[4][r]+=(1-c.attention)*dh[r];}
    float ds=da*c.attention*(1-c.attention);
    for(int r=0;r<QRANK;++r){dz[0][r]+=ds*(c.z[1][r]-c.z[2][r]);dz[1][r]+=ds*c.z[0][r];dz[2][r]-=ds*c.z[0][r];}
    // Local prefix reconstruction gives address and value credit without ever
    // supplying an oracle selection index. Answer-only is the identical graph
    // with auxiliary=0. Every target here has already appeared in the prefix.
    if(auxiliary>0){
        for(int lane=0;lane<3;++lane)if(labels[lane]>=0)for(int r=0;r<QRANK;++r){
            float e=c.z[lane][r]-byte_code(labels[lane],r);loss+=auxiliary*.5f*e*e;dz[lane][r]+=auxiliary*e;}
        for(int lane=3;lane<5;++lane)if(labels[lane]>=0){int node=0;
            for(int depth=0;depth<8;++depth){int bit=(labels[lane]>>(7-depth))&1;
                float z=p.bias[node]+dot(p.output.data()+node*QRANK,c.z[lane].data(),QRANK);
                loss+=auxiliary*(softplus(z)-bit*z);float e=auxiliary*(sigmoid(z)-bit);g.bias[node]+=e;
                for(int r=0;r<QRANK;++r){g.output[node*QRANK+r]+=e*c.z[lane][r];dz[lane][r]+=e*p.output[node*QRANK+r];}
                node=2*node+1+bit;}
        }
    }
    for(int lane=0;lane<5;++lane)for(int r=0;r<QRANK;++r){int row=lane*QRANK+r;float e=dz[lane][r]*(1-c.z[lane][r]*c.z[lane][r]);
        g.offset[row]+=e;for(int j=0;j<D;++j){g.projection[row*D+j]+=e*c.x[lane][j];dx[lane][j]+=e*p.projection[row*D+j]*c.scale[lane][j];}}
    // The scorer is tied to each lane's first projection row. Include BOTH
    // the ordinary projection VJP and the soft-pooling path in that row.
    if(c.pooled)for(int lane=0;lane<5;++lane)for(int k=0;k<c.sources.count;++k){
        float ds=0;for(int j=0;j<D;++j)ds+=dx[lane][j]*(c.sources.x[k][j]-c.raw[lane][j]);
        ds*=c.weights[lane][k];
        for(int j=0;j<D;++j)g.projection[lane*QRANK*D+j]+=ds*c.sources.x[k][j];
    }
    return loss;
}
struct QueryAdapter {
    std::unique_ptr<Model> model;
    QueryHead head{};QueryCache cache{};std::array<Normalizer,5> norm{};
    std::array<float,255> errors{};Labels labels{{-1,-1,-1,-1,-1}};
    int kind=0; // 0 native, 1 shared L3, 2 split frontier, 3 learned pooled views.
    bool training=true,joint=true;float rate=.16f,auxiliary=1.f;
    uint64_t targets=0;int lesion=0;
    QueryAdapter(int parameters,uint32_t seed,bool minimal,int k,bool learn_core):
        model(std::make_unique<Model>(4,parameters-(k?QPARAMS:0),seed,false,false,minimal)),kind(k),joint(learn_core){
        if(k<0||k>3||(k&&learn_core))throw std::invalid_argument("access head requires frozen core");
        model->semantic_laces=false;model->routing_arm=2;model->balance_policy=2;
        std::mt19937 rng(seed^0x46414354u);std::normal_distribution<float> n(0,.02f);
        for(auto& w:head.projection)w=n(rng);
        if(!joint)model->inference(false);
    }
    QueryAdapter(const QueryAdapter& a):model(std::make_unique<Model>(*a.model)),head(a.head),cache(a.cache),norm(a.norm),errors(a.errors),labels(a.labels),
        kind(a.kind),training(a.training),joint(a.joint),rate(a.rate),auxiliary(a.auxiliary),targets(a.targets),lesion(a.lesion){}
    void prepare(){
        std::array<Vec,5> raw{};for(auto& x:raw)x=model->levels[3].s.state;
        if(kind==2)raw[0]=provisional(*model);
        cache=query_forward(head,raw,norm,kind==3?access_sources(*model,lesion):AccessSources{},kind==3,lesion);errors={};
    }
    float logit(int node)const{return !kind||lesion==1?0:head.bias[node]+dot(head.output.data()+node*QRANK,cache.h.data(),QRANK);}
    void backward(std::array<RecallValue,4>&){
        if(!training||!kind)return;
        QueryHead g{};std::array<Vec,5> dx{};query_vjp(head,cache,errors,labels,auxiliary,g,dx);
        // Per-target normalized SGD, identical rate/clip to EXP01-B.
        for(int lane=0;lane<5;++lane){float lr=rate/(1+energy(cache.x[lane]));
            for(int j=0;j<QRANK*D;++j){int i=lane*QRANK*D+j;
                head.projection[i]-=lr*std::clamp(g.projection[i],-2.f,2.f);}}
        step(head.offset,g.offset,rate*.05f);
        float denom=1+dot(cache.h.data(),cache.h.data(),QRANK);
        for(int lane=3;lane<5;++lane)if(labels[lane]>=0)denom+=auxiliary*dot(cache.z[lane].data(),cache.z[lane].data(),QRANK);
        step(head.output,g.output,rate/denom);step(head.bias,g.bias,rate*.05f);
        for(int lane=0;lane<5;++lane)norm[lane].update(cache.raw[lane]);
        labels={{-1,-1,-1,-1,-1}};++targets;
    }
};
inline thread_local QueryAdapter* active_query=nullptr;
struct QueryScope { QueryAdapter* before;explicit QueryScope(QueryAdapter& a):before(active_query){active_query=&a;}~QueryScope(){active_query=before;} };
inline QueryAdapter* query_adapter(const Model* m){return active_query&&active_query->model.get()==m?active_query:nullptr;}
inline void query_prepare(const Model* m){if(auto* a=query_adapter(m))if(a->kind)a->prepare();}
inline float query_logit(const Model* m,int n){auto* a=query_adapter(m);return a?a->logit(n):0;}
inline void query_error(const Model* m,int n,float e){if(auto* a=query_adapter(m))a->errors[n]=e;}
inline void query_backward(const Model* m,std::array<RecallValue,4>& g){if(auto* a=query_adapter(m))a->backward(g);}
inline std::string query_save(const QueryAdapter& a){
    Writer w;uint64_t magic=0x31434341514853ULL;uint32_t version=1;w.pod(magic);w.pod(version);
    auto core=save(*a.model);uint64_t size=core.size();w.pod(size);w.out+=core;
    w.pod(a.head);w.pod(a.norm);w.pod(a.kind);w.pod(a.training);w.pod(a.joint);w.pod(a.rate);w.pod(a.auxiliary);w.pod(a.targets);w.pod(a.lesion);return w.out;
}
inline std::unique_ptr<QueryAdapter> query_load(const char* raw,size_t size){
    Reader r{raw,size};uint64_t magic,n;uint32_t version;r.pod(magic);r.pod(version);r.pod(n);
    if(magic!=0x31434341514853ULL||version!=1||n>r.n)throw std::runtime_error("invalid query-access checkpoint");
    auto a=std::make_unique<QueryAdapter>(500000,1,true,0,true);a->model=std::make_unique<Model>(load(r.p,n));r.p+=n;r.n-=n;
    uint8_t training,joint;static_assert(sizeof(bool)==1,"checkpoint bool ABI");
    r.pod(a->head);r.pod(a->norm);r.pod(a->kind);r.pod(training);r.pod(joint);r.pod(a->rate);r.pod(a->auxiliary);r.pod(a->targets);r.pod(a->lesion);
    if(training>1||joint>1)throw std::runtime_error("invalid flags");a->training=training;a->joint=joint;
    if(r.n||a->kind<0||a->kind>3||a->lesion<0||a->lesion>8||a->model->levels.size()!=4||a->model->training!=(a->training&&a->joint)||
       !std::isfinite(a->rate)||a->rate<0||!std::isfinite(a->auxiliary)||a->auxiliary<0||(a->training&&a->lesion)||(a->kind&&a->joint))throw std::runtime_error("invalid policy");
    auto finite=[](const auto& xs){for(auto x:xs)if(!std::isfinite(x))throw std::runtime_error("nonfinite parameters");};
    finite(a->head.projection);finite(a->head.offset);finite(a->head.output);finite(a->head.bias);
    for(const auto& norm:a->norm){finite(norm.mean);finite(norm.m2);for(auto x:norm.m2)if(x<0)throw std::runtime_error("invalid normalizer");}
    return a;
}
} // namespace shoe
