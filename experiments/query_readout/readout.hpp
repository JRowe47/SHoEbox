#pragma once
#include "native.hpp"
#include <memory>

namespace shoe {
constexpr int QRANK=16, QINPUT=4*D, MINPUT=4*RECALL_V;
using QueryInput=std::array<float,QINPUT>;
using MemoryInput=std::array<float,MINPUT>;
using LowRank=std::array<float,QRANK>;
struct QueryHead {
    std::array<float,QRANK*QINPUT> query{};
    std::array<float,QRANK*MINPUT> memory{};
    LowRank query_bias{},memory_bias{};
    std::array<float,255*QRANK> output{};
    std::array<float,255> bias{};
};
constexpr int QPARAMS=sizeof(QueryHead)/sizeof(float);
struct QueryCache { QueryInput q{};MemoryInput m{};LowRank a{},b{},h{}; };

// Low-rank multiplicative conditioning: Kim et al., ICLR 2017, Eqs. 3--5,
// https://arxiv.org/abs/1610.04325 . This is an adaptation to streaming state,
// not an implementation of their multimodal attention network. Separate the
// addressing cue from retrieved values (Miller et al., EMNLP 2016,
// https://aclanthology.org/D16-1147/). Neither input receives parsed task keys.
inline QueryCache query_forward(const QueryHead& p,const QueryInput& q,const MemoryInput& m,int kind){
    QueryCache c;c.q=q;c.m=m;
    for(int r=0;r<QRANK;++r){
        c.a[r]=std::tanh(p.query_bias[r]+dot(p.query.data()+r*QINPUT,q.data(),QINPUT));
        c.b[r]=std::tanh(p.memory_bias[r]+dot(p.memory.data()+r*MINPUT,m.data(),MINPUT));
        c.h[r]=kind==2?c.a[r]*c.b[r]:.5f*(c.a[r]+c.b[r]);
    }return c;
}
inline void query_vjp(const QueryHead& p,const QueryCache& c,int kind,
                      const std::array<float,255>& errors,QueryHead& g,QueryInput& dq,MemoryInput& dm){
    LowRank dh{};
    for(int n=0;n<255;++n)if(errors[n]){
        g.bias[n]+=errors[n];
        for(int r=0;r<QRANK;++r){dh[r]+=errors[n]*p.output[n*QRANK+r];g.output[n*QRANK+r]+=errors[n]*c.h[r];}
    }
    for(int r=0;r<QRANK;++r){
        float da=dh[r]*(kind==2?c.b[r]:.5f)*(1-c.a[r]*c.a[r]);
        float db=dh[r]*(kind==2?c.a[r]:.5f)*(1-c.b[r]*c.b[r]);
        g.query_bias[r]+=da;g.memory_bias[r]+=db;
        for(int j=0;j<QINPUT;++j){g.query[r*QINPUT+j]+=da*c.q[j];dq[j]+=da*p.query[r*QINPUT+j];}
        for(int j=0;j<MINPUT;++j){g.memory[r*MINPUT+j]+=db*c.m[j];dm[j]+=db*p.memory[r*MINPUT+j];}
    }
}
struct QueryAdapter {
    std::unique_ptr<Model> model;
    QueryHead head{};QueryCache cache{};
    std::array<float,255> errors{};
    int kind=0; // 0 original decoder, 1 additive control, 2 multiplicative.
    bool training=true,joint=true;
    float rate=.16f;
    uint64_t targets=0;
    int lesion=0; // frozen only: 1 no extra head, 2 zero state cues, 3 zero reads.
    QueryAdapter(int parameters,uint32_t seed,bool minimal,int k,bool learn_core):
        model(std::make_unique<Model>(4,parameters-(k?QPARAMS:0),seed,false,false,minimal)),kind(k),joint(learn_core){
        if(k<0||k>2)throw std::invalid_argument("invalid query head");
        model->semantic_laces=false;model->routing_arm=2;model->balance_policy=2;
        std::mt19937 rng(seed^0x51524541u);std::normal_distribution<float> n(0,.1f);
        for(auto& w:head.query)w=n(rng);for(auto& w:head.memory)w=n(rng);
        // Zero residual output preserves the pretrained decoder initially.
        if(!joint)model->inference(false);
    }
    QueryAdapter(const QueryAdapter& a):model(std::make_unique<Model>(*a.model)),head(a.head),cache(a.cache),errors(a.errors),
        kind(a.kind),training(a.training),joint(a.joint),rate(a.rate),targets(a.targets),lesion(a.lesion){}
    void prepare(){
        QueryInput q{};MemoryInput m{};
        for(int l=0;l<4;++l){
            std::copy(model->levels[l].s.state.begin(),model->levels[l].s.state.end(),q.begin()+l*D);
            std::copy(model->levels[l].read_now.value.begin(),model->levels[l].read_now.value.end(),m.begin()+l*RECALL_V);
        }
        if(lesion==2)q={};if(lesion==3)m={};
        cache=query_forward(head,q,m,kind);errors={};
    }
    float logit(int node)const{return !kind||lesion==1?0:head.bias[node]+dot(head.output.data()+node*QRANK,cache.h.data(),QRANK);}
    void backward(std::array<RecallValue,4>& recalled){
        if(!training||!kind)return;
        QueryHead g{};QueryInput dq{};MemoryInput dm{};
        query_vjp(head,cache,kind,errors,g,dq,dm);
        if(joint){
            // Native bounded local credit: only live states receive a VJP;
            // entering retired states stay detached, as in Model::teach().
            for(int l=0;l<4;++l){auto& s=model->levels[l].s;
                if(s.count)for(int j=0;j<D;++j)s.events[s.count-1].task[j]+=dq[l*D+j];
                for(int j=0;j<RECALL_V;++j)recalled[l][j]+=dm[l*RECALL_V+j];
            }
        }
        // Per-observation SGD, no replay, batch, extra pass, or growing tape.
        step(head.query,g.query,rate/(1+dot(cache.q.data(),cache.q.data(),QINPUT)));
        step(head.memory,g.memory,rate/(1+dot(cache.m.data(),cache.m.data(),MINPUT)));
        step(head.query_bias,g.query_bias,rate*.05f);step(head.memory_bias,g.memory_bias,rate*.05f);
        step(head.output,g.output,rate/(1+dot(cache.h.data(),cache.h.data(),QRANK)));
        step(head.bias,g.bias,rate*.05f);++targets;
    }
};
inline thread_local QueryAdapter* active_query=nullptr;
struct QueryScope {
    QueryAdapter* before;
    explicit QueryScope(QueryAdapter& a):before(active_query){active_query=&a;}
    ~QueryScope(){active_query=before;}
};
inline QueryAdapter* query_adapter(const Model* m){return active_query&&active_query->model.get()==m?active_query:nullptr;}
inline void query_prepare(const Model* m){if(auto* a=query_adapter(m))if(a->kind)a->prepare();}
inline float query_logit(const Model* m,int n){auto* a=query_adapter(m);return a?a->logit(n):0;}
inline void query_error(const Model* m,int n,float e){if(auto* a=query_adapter(m))a->errors[n]=e;}
inline void query_backward(const Model* m,std::array<RecallValue,4>& g){if(auto* a=query_adapter(m))a->backward(g);}

inline std::string query_save(const QueryAdapter& a){
    Writer w;uint64_t magic=0x31524555514853ULL;uint32_t version=1;w.pod(magic);w.pod(version);
    auto core=save(*a.model);uint64_t size=core.size();w.pod(size);w.out+=core;
    w.pod(a.head);w.pod(a.kind);w.pod(a.training);w.pod(a.joint);w.pod(a.rate);w.pod(a.targets);w.pod(a.lesion);return w.out;
}
inline std::unique_ptr<QueryAdapter> query_load(const char* raw,size_t size){
    Reader r{raw,size};uint64_t magic,n;uint32_t version;r.pod(magic);r.pod(version);r.pod(n);
    if(magic!=0x31524555514853ULL||version!=1||n>r.n)throw std::runtime_error("invalid query checkpoint");
    auto a=std::make_unique<QueryAdapter>(500000,1,true,0,true);
    a->model=std::make_unique<Model>(load(r.p,n));r.p+=n;r.n-=n;
    r.pod(a->head);r.pod(a->kind);r.pod(a->training);r.pod(a->joint);r.pod(a->rate);r.pod(a->targets);r.pod(a->lesion);
    if(r.n||a->kind<0||a->kind>2||a->lesion<0||a->lesion>3||a->model->levels.size()!=4||
       a->model->training!=(a->training&&a->joint)||!std::isfinite(a->rate)||a->rate<0||
       (a->training&&a->lesion))throw std::runtime_error("invalid query policy");
    auto finite=[](const auto& xs){for(float x:xs)if(!std::isfinite(x))throw std::runtime_error("nonfinite query weights");};
    finite(a->head.query);finite(a->head.memory);finite(a->head.query_bias);finite(a->head.memory_bias);finite(a->head.output);finite(a->head.bias);
    return a;
}
} // namespace shoe
