#pragma once
#include "native.hpp"
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
struct QueryCache { Vec raw{},x{},scale{};std::array<LowRank,5> z{};LowRank h{};float attention=.5f;bool rls=false; };
struct Normalizer {
    uint64_t count=0;
    std::array<double,D> mean{},m2{};
    void update(const Vec& x){
        ++count;for(int j=0;j<D;++j){double delta=x[j]-mean[j];mean[j]+=delta/count;m2[j]+=delta*(x[j]-mean[j]);}
    }
};

// Address/value separation: Miller et al., EMNLP 2016, section 3,
// https://aclanthology.org/D16-1147/ . The two lanes below are *decoded views*
// of the existing L3 vector, not new memory slots or stored byte records.
// One soft query/key selection mixes rank-16 values into the native byte tree.
// No parser, true selected-record index, or codebook is used in inference.
inline QueryCache query_forward(const QueryHead& p,const Vec& raw,const Normalizer& norm,int lesion=0,bool rls=false){
    QueryCache c;c.raw=raw;c.rls=rls;
    for(int j=0;j<D;++j){c.scale[j]=norm.count>1?float(1/std::sqrt(std::max(norm.m2[j]/(norm.count-1),1e-6))):1;
        c.x[j]=(raw[j]-float(norm.mean[j]))*c.scale[j];}
    if(lesion==3)c.x={};
    for(int lane=0;lane<5;++lane)for(int r=0;r<QRANK;++r){int row=lane*QRANK+r;
        c.z[lane][r]=std::tanh(p.offset[row]+dot(p.projection.data()+row*D,rls&&lane<3?c.raw.data():c.x.data(),D));}
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
                       const Labels& labels,float auxiliary,QueryHead& g,Vec& dx){
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
        g.offset[row]+=e;for(int j=0;j<D;++j){g.projection[row*D+j]+=e*(c.rls&&lane<3?c.raw[j]:c.x[j]);dx[j]+=e*p.projection[row*D+j]*(c.rls&&lane<3?1:c.scale[j]);}}
    return loss;
}
// Bounded online inverse-correlation update (Sherman--Morrison / RLS).
// Sussillo & Abbott, Neuron 2009, equations 4--5:
// https://doi.org/10.1016/j.neuron.2009.07.018 . Adapt only the online readout
// optimizer, not their chaotic reservoir or FORCE feedback architecture.
// Raw L3 plus an intercept keeps the regression coordinates stationary.
struct AddressRLS {
    static constexpr int N=D+1;
    std::array<double,N*N> inverse{};
    uint64_t updates=0;
    AddressRLS(){for(int j=0;j<N;++j)inverse[j*N+j]=1e6;}
    void update(QueryHead& p,const Vec& raw,const Labels& labels){
        if(labels[0]<0||labels[1]<0||labels[2]<0)return;
        std::array<double,N> x{},px{};for(int j=0;j<D;++j)x[j]=raw[j];x[D]=1;
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)px[i]+=inverse[i*N+j]*x[j];
        double denom=1;for(int j=0;j<N;++j)denom+=x[j]*px[j];
        if(!(denom>=1)||!std::isfinite(denom))throw std::runtime_error("invalid RLS covariance");
        for(int lane=0;lane<3;++lane)for(int r=0;r<QRANK;++r){int row=lane*QRANK+r;
            double prediction=p.offset[row];for(int j=0;j<D;++j)prediction+=p.projection[row*D+j]*x[j];
            double error=prediction-std::atanh(double(byte_code(labels[lane],r)));
            for(int j=0;j<D;++j)p.projection[row*D+j]-=float(error*px[j]/denom);
            p.offset[row]-=float(error*px[D]/denom);
        }
        for(int i=0;i<N;++i)for(int j=0;j<=i;++j){double v=inverse[i*N+j]-px[i]*px[j]/denom;
            inverse[i*N+j]=inverse[j*N+i]=v;}++updates;
    }
};
struct QueryAdapter {
    std::unique_ptr<Model> model;
    QueryHead head{};QueryCache cache{};Normalizer norm{};std::unique_ptr<AddressRLS> rls;
    std::array<float,255> errors{};Labels labels{{-1,-1,-1,-1,-1}};
    int kind=0; // 0 native, 1 answer-only, 2 local SGD, 3 local RLS address + SGD value.
    bool training=true,joint=true;float rate=.5f,auxiliary=1.f;
    uint64_t targets=0;int lesion=0;
    QueryAdapter(int parameters,uint32_t seed,bool minimal,int k,bool learn_core):
        model(std::make_unique<Model>(4,parameters-(k?QPARAMS:0),seed,false,false,minimal)),kind(k),joint(learn_core){
        if(k<0||k>3||(k==3&&learn_core))throw std::invalid_argument("invalid factorized head");
        model->semantic_laces=false;model->routing_arm=2;model->balance_policy=2;
        std::mt19937 rng(seed^0x46414354u);std::normal_distribution<float> n(0,.02f);
        for(auto& w:head.projection)w=n(rng);
        if(k==3){rls=std::make_unique<AddressRLS>();std::fill_n(head.projection.begin(),3*QRANK*D,0.f);}
        if(!joint)model->inference(false);
    }
    QueryAdapter(const QueryAdapter& a):model(std::make_unique<Model>(*a.model)),head(a.head),cache(a.cache),norm(a.norm),rls(a.rls?std::make_unique<AddressRLS>(*a.rls):nullptr),errors(a.errors),labels(a.labels),
        kind(a.kind),training(a.training),joint(a.joint),rate(a.rate),auxiliary(a.auxiliary),targets(a.targets),lesion(a.lesion){}
    void prepare(){cache=query_forward(head,model->levels[3].s.state,norm,lesion,bool(rls));errors={};}
    float logit(int node)const{return !kind||lesion==1?0:head.bias[node]+dot(head.output.data()+node*QRANK,cache.h.data(),QRANK);}
    void backward(std::array<RecallValue,4>&){
        if(!training||!kind)return;
        QueryHead g{};Vec dx{};query_vjp(head,cache,errors,labels,kind>=2?auxiliary:0,g,dx);
        if(joint){auto& s=model->levels[3].s;if(s.count)for(int j=0;j<D;++j)s.events[s.count-1].task[j]+=dx[j];}
        // Normalized per-target SGD. No moments, replay, or additional pass.
        if(rls){std::fill_n(g.projection.begin(),3*QRANK*D,0.f);std::fill_n(g.offset.begin(),3*QRANK,0.f);}
        step(head.projection,g.projection,rate/(1+energy(cache.x)));
        step(head.offset,g.offset,rate*.05f);
        float denom=1+dot(cache.h.data(),cache.h.data(),QRANK);
        if(kind>=2)for(int lane=3;lane<5;++lane)if(labels[lane]>=0)denom+=auxiliary*dot(cache.z[lane].data(),cache.z[lane].data(),QRANK);
        step(head.output,g.output,rate/denom);step(head.bias,g.bias,rate*.05f);
        if(rls)rls->update(head,cache.raw,labels);
        norm.update(cache.raw);labels={{-1,-1,-1,-1,-1}};++targets;
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
    Writer w;uint64_t magic=0x314b5146414853ULL;uint32_t version=2;w.pod(magic);w.pod(version);
    auto core=save(*a.model);uint64_t size=core.size();w.pod(size);w.out+=core;
    w.pod(a.head);w.pod(a.norm);w.pod(a.kind);w.pod(a.training);w.pod(a.joint);w.pod(a.rate);w.pod(a.auxiliary);w.pod(a.targets);w.pod(a.lesion);if(a.rls)w.pod(*a.rls);return w.out;
}
inline std::unique_ptr<QueryAdapter> query_load(const char* raw,size_t size){
    Reader r{raw,size};uint64_t magic,n;uint32_t version;r.pod(magic);r.pod(version);r.pod(n);
    if(magic!=0x314b5146414853ULL||(version!=1&&version!=2)||n>r.n)throw std::runtime_error("invalid factorized checkpoint");
    auto a=std::make_unique<QueryAdapter>(500000,1,true,0,true);a->model=std::make_unique<Model>(load(r.p,n));r.p+=n;r.n-=n;
    uint8_t training,joint;static_assert(sizeof(bool)==1,"checkpoint bool ABI");
    r.pod(a->head);r.pod(a->norm);r.pod(a->kind);r.pod(training);r.pod(joint);r.pod(a->rate);r.pod(a->auxiliary);r.pod(a->targets);r.pod(a->lesion);
    if(training>1||joint>1)throw std::runtime_error("invalid flags");a->training=training;a->joint=joint;
    if(version==2&&a->kind==3){a->rls=std::make_unique<AddressRLS>();r.pod(*a->rls);}
    if(r.n||a->kind<0||a->kind>3||a->lesion<0||a->lesion>5||a->model->levels.size()!=4||a->model->training!=(a->training&&a->joint)||
       !std::isfinite(a->rate)||a->rate<0||!std::isfinite(a->auxiliary)||a->auxiliary<0||(a->training&&a->lesion)||(a->kind==3&&(!a->rls||a->joint)))throw std::runtime_error("invalid policy");
    auto finite=[](const auto& xs){for(auto x:xs)if(!std::isfinite(x))throw std::runtime_error("nonfinite parameters");};
    finite(a->head.projection);finite(a->head.offset);finite(a->head.output);finite(a->head.bias);finite(a->norm.mean);finite(a->norm.m2);
    if(a->rls){finite(a->rls->inverse);for(int j=0;j<AddressRLS::N;++j)if(a->rls->inverse[j*AddressRLS::N+j]<=0)throw std::runtime_error("invalid covariance diagonal");}
    for(auto x:a->norm.m2)if(x<0)throw std::runtime_error("invalid normalizer");return a;
}
} // namespace shoe
