#pragma once
#include "native.hpp"
#include <memory>

namespace shoe {
constexpr int QRANK=16, VRANK=8, QINPUT=D, MINPUT=0;
using LowRank=std::array<float,QRANK>;
using ValueCode=std::array<float,VRANK>;
using Labels=std::array<int32_t,5>;
using Affine=std::array<float,D+1>;
struct QueryHead {
    std::array<Affine,5> gate{}; // four within-pair phases, then current-query recognizer
    std::array<Affine,5*QRANK> key{};
    std::array<Affine,4*VRANK> value{};
};
constexpr int QPARAMS=sizeof(QueryHead)/sizeof(float);
struct QueryCache {
    Vec query_raw{}; std::array<Vec,2> raw{};
    LowRank query{};std::array<LowRank,2> keys{};
    std::array<ValueCode,2> values{};ValueCode h{};
    std::array<int,2> slots{{-1,-1}},phases{{0,0}};
    std::array<float,2> scores{{-1e30f,-1e30f}};
    int query_slot=-1;float attention=.5f;
};
inline Vec source_vector(const Vec& stable){Vec x=stable;for(auto& z:x)z*=8;return x;}
inline float affine(const Affine& p,const Vec& x){return p[D]+dot(p.data(),x.data());}
// Reconstruction targets only. Neither function is called by inference.
inline float byte_code(int byte,int coordinate){
    uint32_t x=uint32_t(byte+1)*0x9e3779b9u+uint32_t(coordinate+1)*0x85ebca6bu;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    return x&1?.7f:-.7f;
}
inline float value_code(int byte,int coordinate){return ((byte>>coordinate)&1)?.7f:-.7f;}
inline void affine_step(Affine& p,const Vec& x,float error,float rate){
    const float lr=rate/(1+energy(x));
    for(int j=0;j<D;++j)p[j]-=lr*error*x[j];p[D]-=lr*error;
}
inline void reconstruct(Affine& p,const Vec& x,float target,float rate){
    float z=std::tanh(affine(p,x));affine_step(p,x,(z-target)*(1-z*z),rate);
}

// Paired values are mixed only AFTER query/key comparison. Miller et al.,
// EMNLP 2016, sections 3.1-3.2, https://aclanthology.org/D16-1147/.
// Fixed constituent phases are coordinate roles, not parsed record locations.
// Top-two source routing is trained by a separate local presence loss; no
// derivative through the discrete top-k operation is claimed.
inline QueryCache query_forward(const QueryHead& p,const Model& m,int lesion=0,bool latest=false){
    QueryCache c;const auto& queries=m.levels[0].retained;
    float qbest=-1e30f;int latest_slot=-1;uint64_t latest_born=0;Vec latest_raw{};
    for(int k=0;k<queries.count;++k){Vec x=source_vector(queries.units[k].stable);
        float recognition=affine(p.gate[4],x);
        // Recognition and recency are distinct decisions: confidence must not
        // make an old query supersede a newer positively recognized occurrence.
        if(latest&&recognition>0&&queries.units[k].born>latest_born){latest_slot=k;latest_born=queries.units[k].born;latest_raw=x;}
        float score=recognition-.5f*float(queries.clock-queries.units[k].born);
        if(score>qbest){qbest=score;c.query_slot=k;c.query_raw=x;}}
    if(latest_slot>=0){c.query_slot=latest_slot;c.query_raw=latest_raw;}
    for(int r=0;r<QRANK;++r)c.query[r]=std::tanh(affine(p.key[4*QRANK+r],c.query_raw));
    if(lesion==2)c.query={};
    const auto& records=m.levels[1].retained;
    for(int k=0;k<records.count;++k){Vec x=source_vector(records.units[k].stable);
        if(lesion==3)x={};
        for(int phase=0;phase<4;++phase){float score=affine(p.gate[phase],x);
            int place=score>c.scores[0]?0:(score>c.scores[1]?1:2);
            if(place==2)continue;
            if(place==0){c.scores[1]=c.scores[0];c.slots[1]=c.slots[0];c.phases[1]=c.phases[0];c.raw[1]=c.raw[0];}
            c.scores[place]=score;c.slots[place]=k;c.phases[place]=phase;c.raw[place]=x;
        }
    }
    for(int k=0;k<2;++k){int phase=c.phases[k];
        for(int r=0;r<QRANK;++r)c.keys[k][r]=std::tanh(affine(p.key[phase*QRANK+r],c.raw[k]));
        for(int r=0;r<VRANK;++r)c.values[k][r]=std::tanh(affine(p.value[phase*VRANK+r],c.raw[k]));
    }
    float difference=0;
    for(int r=0;r<QRANK;++r){float a=c.query[r]-c.keys[0][r],b=c.query[r]-c.keys[1][r];difference+=b*b-a*a;}
    c.attention=sigmoid(4*difference);
    if(lesion==2||lesion==4)c.attention=.5f;
    if(lesion==5)c.attention=1-c.attention;
    if(lesion==6)std::swap(c.values[0],c.values[1]);
    for(int r=0;r<VRANK;++r)c.h[r]=c.attention*c.values[0][r]+(1-c.attention)*c.values[1][r];
    return c;
}

struct QueryAdapter {
    std::unique_ptr<Model> model;QueryHead head{};QueryCache cache{};
    std::array<float,255> errors{};
    Affine applicability{}; Vec gate_raw{};float gate_value=1;float responsibility=0,mixture_loss=0;ValueCode byte_logits{};
    uint64_t gate_targets=0,projection_steps=0,presence_steps=0;
    int kind=0;bool training=true,joint=true;float rate=.16f,auxiliary=1.f;
    uint64_t targets=0,local_updates=0;int lesion=0;
    QueryAdapter(int parameters,uint32_t seed,bool minimal,int k,bool learn_core):
        model(std::make_unique<Model>(4,parameters-(k?QPARAMS:0)-(k>=3?D+1:0),seed,false,false,minimal)),kind(k),joint(learn_core){
        if(k<0||k>4)throw std::invalid_argument("invalid constituent head");
        model->semantic_laces=false;model->routing_arm=2;model->balance_policy=2;
        std::mt19937 rng(seed^0x50414952u);std::normal_distribution<float> n(0,.002f);
        for(auto& row:head.key)for(auto& x:row)x=n(rng);
        for(auto& row:head.value)for(auto& x:row)x=n(rng);
        if(!joint)model->inference(false);
    }
    QueryAdapter(const QueryAdapter& a):model(std::make_unique<Model>(*a.model)),head(a.head),cache(a.cache),errors(a.errors),
        applicability(a.applicability),gate_raw(a.gate_raw),gate_value(a.gate_value),responsibility(a.responsibility),mixture_loss(a.mixture_loss),byte_logits(a.byte_logits),gate_targets(a.gate_targets),projection_steps(a.projection_steps),presence_steps(a.presence_steps),
        kind(a.kind),training(a.training),joint(a.joint),rate(a.rate),auxiliary(a.auxiliary),targets(a.targets),local_updates(a.local_updates),lesion(a.lesion){}
    void local(const Labels& labels){
        if(!training||!kind||auxiliary==0)return;
        const float lr=rate*auxiliary;
        if(model->bytes>=2){Vec x=source_vector(model->levels[0].retained.last);bool yes=labels[0]>=0;
            // E6: applicability is an observed-window local label. Generic
            // next-byte training never updates this classifier (kind 4).
            if(kind==4){affine_step(applicability,x,(sigmoid(affine(applicability,x))-yes)*(yes?4.f:1.f),lr);++gate_targets;}
            affine_step(head.gate[4],x,(sigmoid(affine(head.gate[4],x))-yes)*(yes?4.f:1.f),lr);
            if(yes)for(int r=0;r<QRANK;++r)reconstruct(head.key[4*QRANK+r],x,byte_code(labels[0],r),lr*4);
            ++local_updates;++presence_steps;if(yes)projection_steps+=QRANK;
        }
        if(model->bytes>=8&&model->bytes%4==0){Vec x=source_vector(model->levels[1].retained.last);
            for(int phase=0;phase<4;++phase){bool yes=labels[phase+1]>=0;
                affine_step(head.gate[phase],x,(sigmoid(affine(head.gate[phase],x))-yes)*(yes?4.f:1.f),lr);
                ++presence_steps;if(yes)projection_steps+=QRANK+VRANK;
                if(yes){int key=labels[phase+1]/256,value=labels[phase+1]%256;
                    for(int r=0;r<QRANK;++r)reconstruct(head.key[phase*QRANK+r],x,byte_code(key,r),lr*4);
                    for(int r=0;r<VRANK;++r)reconstruct(head.value[phase*VRANK+r],x,value_code(value,r),lr*4);
                }
            }++local_updates;
        }
    }
    // See et al., ACL 2017, Sec. 2.2, https://aclanthology.org/P17-1099/:
    // a learned input-dependent switch joins retrieval and generation. Here
    // E3 switches between two normalized byte distributions, not pointers.
    // It uses the CURRENT existing stable pair; no byte constants, task mask,
    // source index or future target participates in forward inference.
    void prepare(){
        cache=query_forward(head,*model,lesion,kind>=2);errors={};
        gate_raw=source_vector(model->levels[0].retained.last);
        gate_value=kind>=3?sigmoid(affine(applicability,gate_raw)):1.f;
        if(lesion==7)gate_value=0;if(lesion==8)gate_value=1;
        // Generic byte-code calibration: the original value reconstruction
        // targets are +/-0.7, so convert that signed bit expectation to a
        // Bernoulli probability. No task symbols or record parser are used.
        for(int r=0;r<VRANK;++r){float p=std::clamp(.5f+.5f*cache.h[r]/.7f,1e-6f,1.f-1e-6f);byte_logits[r]=std::log(p)-std::log1p(-p);}
    }
    float residual(int node)const{return byte_logits[7-(31-__builtin_clz(unsigned(node+1)))];}
    float logit(int node)const{return !kind||lesion==1||kind>=3?0:gate_value*residual(node);}
    void backward(std::array<RecallValue,4>&){
        if(!training||!kind)return;
        // Generic next-byte credit trains the applicability gate and native
        // core. Bit projections retain their explicit local reconstruction
        // objective; there is no separately drifting residual output decoder.
        if(kind==3){affine_step(applicability,gate_raw,gate_value-responsibility,rate);++gate_targets;}
        ++targets;
    }
};
inline thread_local QueryAdapter* active_query=nullptr;
struct QueryScope {QueryAdapter* before;explicit QueryScope(QueryAdapter& a):before(active_query){active_query=&a;}~QueryScope(){active_query=before;}};
inline QueryAdapter* query_adapter(const Model* m){return active_query&&active_query->model.get()==m?active_query:nullptr;}
inline void query_prepare(const Model* m){if(auto* a=query_adapter(m))if(a->kind)a->prepare();}
inline float query_logit(const Model* m,int n){auto* a=query_adapter(m);return a?a->logit(n):0;}
// See et al., ACL 2017, Sec. 2.2: https://aclanthology.org/P17-1099/.
// Exact online credit for a normalized mixture; target-dependent responsibility
// is training scratch only, never a source/query oracle or inference input.
inline double logadd(double x,double y){double m=std::max(x,y);return m+std::log(std::exp(x-m)+std::exp(y-m));}
inline void query_target(const Model* m,uint8_t target){
 auto* a=query_adapter(m);if(!a||a->kind<3||a->lesion==1)return;
 int node=0;float nl=0,hl=0;std::array<int,8> nodes{},bits{};
 for(int d=0;d<8;++d){int bit=(target>>(7-d))&1;nodes[d]=node;bits[d]=bit;
  float z=m->decoder_bias[node]+dot(m->decoder[node].data(),m->features.data());
  for(size_t l=0;l<m->levels.size();++l)if(int(l)!=m->mask)z+=dot(m->levels[l].retrieval.read.data()+node*RECALL_V,m->levels[l].read_now.value.data(),RECALL_V);
  float h=a->residual(node);nl+=softplus(z)-bit*z;hl+=softplus(h)-bit*h;node=2*node+1+bit;
 }
 double x=std::log(std::max(1.-double(a->gate_value),1e-30))-nl,y=std::log(std::max(double(a->gate_value),1e-30))-hl;
 double mix=logadd(x,y);a->mixture_loss=float(-mix);a->responsibility=float(std::exp(y-mix));
 for(int d=0;d<8;++d)a->errors[nodes[d]]=a->responsibility*(sigmoid(a->residual(nodes[d]))-bits[d]);
}
inline float query_native_credit(const Model* m){auto* a=query_adapter(m);return a&&a->kind>=3&&a->lesion!=1?1-a->responsibility:1.f;}
inline float query_loss(const Model* m,float native){auto* a=query_adapter(m);return a&&a->kind>=3&&a->lesion!=1?a->mixture_loss:native;}
inline void query_mix_probs(const Model* m,float* out){
 auto* a=query_adapter(m);if(!a||a->kind<3||a->lesion==1)return;
 std::array<float,511> h{};
 std::array<float,8> zero{},one{};for(int bit=0;bit<8;++bit){zero[bit]=softplus(a->byte_logits[bit]);one[bit]=softplus(-a->byte_logits[bit]);}
 for(int n=0;n<255;++n){int bit=7-(31-__builtin_clz(unsigned(n+1)));h[2*n+1]=h[n]-zero[bit];h[2*n+2]=h[n]-one[bit];}
 double ng=std::log(std::max(1.-double(a->gate_value),1e-30)),hg=std::log(std::max(double(a->gate_value),1e-30));
 for(int b=0;b<256;++b)out[b]=float(logadd(ng+out[b],hg+h[b+255]));
}
inline void query_error(const Model* m,int n,float e){if(auto* a=query_adapter(m))if(a->kind<3||a->lesion==1)a->errors[n]=e;}
inline void query_backward(const Model* m,std::array<RecallValue,4>& g){if(auto* a=query_adapter(m))a->backward(g);}
inline std::string query_save(const QueryAdapter& a){
    Writer w;uint64_t magic=0x36455041514853ULL;uint32_t version=1;w.pod(magic);w.pod(version);
    auto core=save(*a.model);uint64_t size=core.size();w.pod(size);w.out+=core;
    w.pod(a.head);w.pod(a.kind);w.pod(a.training);w.pod(a.joint);w.pod(a.rate);w.pod(a.auxiliary);w.pod(a.targets);w.pod(a.local_updates);w.pod(a.lesion);w.pod(a.applicability);w.pod(a.gate_targets);w.pod(a.projection_steps);w.pod(a.presence_steps);return w.out;
}
inline std::unique_ptr<QueryAdapter> query_load(const char* raw,size_t size){
    Reader r{raw,size};uint64_t magic,n;uint32_t version;r.pod(magic);r.pod(version);r.pod(n);
    if(magic!=0x36455041514853ULL||version!=1||n>r.n)throw std::runtime_error("invalid paired checkpoint");
    auto a=std::make_unique<QueryAdapter>(500000,1,true,0,true);a->model=std::make_unique<Model>(load(r.p,n));r.p+=n;r.n-=n;
    uint8_t training,joint;r.pod(a->head);r.pod(a->kind);r.pod(training);r.pod(joint);r.pod(a->rate);r.pod(a->auxiliary);r.pod(a->targets);r.pod(a->local_updates);r.pod(a->lesion);r.pod(a->applicability);r.pod(a->gate_targets);r.pod(a->projection_steps);r.pod(a->presence_steps);
    if(training>1||joint>1)throw std::runtime_error("invalid flags");a->training=training;a->joint=joint;
    if(r.n||a->kind<0||a->kind>4||a->lesion<0||a->lesion>8||a->model->levels.size()!=4||a->model->training!=(a->training&&a->joint)||
       !std::isfinite(a->rate)||a->rate<0||!std::isfinite(a->auxiliary)||a->auxiliary<0||(a->training&&a->lesion))throw std::runtime_error("invalid policy");
    auto finite=[](const auto& xs){for(auto x:xs)if(!std::isfinite(x))throw std::runtime_error("nonfinite parameters");};
    for(auto& x:a->head.gate)finite(x);for(auto& x:a->head.key)finite(x);for(auto& x:a->head.value)finite(x);finite(a->applicability);
    return a;
}
} // namespace shoe
