#pragma once
// Independent SHoE multiscale runtime. No regional scheduler or publication DAG.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace shoe {
constexpr int D=64, K=8, BLOCK=8, GROUP=4, NEG=16, VOCAB=256;
using Vec=std::array<float,D>;
using Mat=std::array<float,D*D>;
using Small=std::array<float,K>;
using Memory=std::array<float,K*K>;
inline float dot(const float* a,const float* b,int n=D){
    float s=0; for(int j=0;j<n;++j)s+=a[j]*b[j]; return s;
}
inline void routing_dot16(const float* keys,const float* cue,float* scores){
    // Each lane is a separate expert. No horizontal reduction, reassociation,
    // or FMA: arithmetic order matches dot() exactly, including tie behavior.
#if defined(__AVX512F__)
    __m512 sum=_mm512_setzero_ps();
    for(int j=0;j<D;++j)
        sum=_mm512_add_ps(sum,_mm512_mul_ps(_mm512_loadu_ps(keys+j*16),_mm512_set1_ps(cue[j])));
    _mm512_storeu_ps(scores,sum);
#elif defined(__AVX2__)
    __m256 lo=_mm256_setzero_ps(),hi=_mm256_setzero_ps();
    for(int j=0;j<D;++j){
        __m256 x=_mm256_set1_ps(cue[j]);
        lo=_mm256_add_ps(lo,_mm256_mul_ps(_mm256_loadu_ps(keys+j*16),x));
        hi=_mm256_add_ps(hi,_mm256_mul_ps(_mm256_loadu_ps(keys+j*16+8),x));
    }
    _mm256_storeu_ps(scores,lo);_mm256_storeu_ps(scores+8,hi);
#else
    std::fill(scores,scores+16,0.f);
    for(int j=0;j<D;++j)for(int lane=0;lane<16;++lane)scores[lane]+=keys[j*16+lane]*cue[j];
#endif
}
inline float energy(const Vec& a){return dot(a.data(),a.data());}
inline void normalize(Vec& a){float s=1/std::sqrt(energy(a)+1e-8f);for(auto& x:a)x*=s;}
inline float sigmoid(float x){return 1/(1+std::exp(-x));}
inline float softplus(float x){return std::max(x,0.f)+std::log1p(std::exp(-std::abs(x)));}
inline void mv(const Mat& a,const Vec& x,Vec& y){for(int i=0;i<D;++i)y[i]=dot(a.data()+i*D,x.data());}
inline void vjp(const Mat& a,const Vec& g,Vec& x){
    for(int i=0;i<D;++i)for(int j=0;j<D;++j)x[j]+=a[i*D+j]*g[i];
}
inline void outer(Mat& a,const Vec& g,const Vec& x){
    for(int i=0;i<D;++i)for(int j=0;j<D;++j)a[i*D+j]+=g[i]*x[j];
}
template<class T> void step(T& p,const T& g,float lr){
    for(size_t j=0;j<p.size();++j)p[j]-=lr*std::clamp(g[j],-2.f,2.f);
}

#include "address_memory.hpp"
#include "recall.hpp"
#include "recall_credit.hpp"

struct Expert {Mat w{};Vec bias{},key{};};
struct Shared {
    Mat recurrent{},gain{},bias{},predict{};
    std::array<float,D*K> read{};
    std::array<float,GROUP*D> role{};
    Vec length{},offset{};
};
struct Event {
    Vec input{},content{},residual{},state{},before{},gain_grad{},bias_grad{},task{},native{};
    std::array<Vec,GROUP> children{};
    std::array<float,GROUP> lengths{};
    Memory memory{};
    Small key{},value{},read{};
    Vec forecast{},unit{};
    float unit_norm=1,gate=1,gate_derivative=0,span=1;
    int expert=0,runner=0,byte=0,uses=0,has_target=0,reconstruction_role=0,content_id=-1;
};
// Content-only eligibility: no recurrent state, memory, readout or replay.
// Its weights stay frozen until this bounded publication epoch retires.
struct Constituent {
    Vec input{},content{},residual{},grad{};
    std::array<Vec,GROUP> children{};
    std::array<float,GROUP> lengths{};
    std::array<int,GROUP> child_ids{{-1,-1,-1,-1}};
    float gate=1,gate_derivative=0,span=1;
    int expert=0,runner=0,byte=0;
};
struct Stream {
    std::array<Event,BLOCK> events{};
    std::array<Vec,NEG> negatives{};
    std::array<Vec,GROUP> children{};
    std::array<float,GROUP> lengths{};
    std::array<int,GROUP> child_ids{{-1,-1,-1,-1}};
    Vec state{},content{},gain{},bias{};
    Memory memory{};
    // Kept outside Event: the byte loop does not clear a slow memory record.
    AddressStream addressed{};
    std::array<AddressEvent,BLOCK> address_events{};
    std::array<Vec,GROUP> stable_children{};
    Vec stable{};
    int count=0,children_count=0,negative_count=0,negative_cursor=0;
    uint64_t executions=0,updates=0,native_targets=0;
    double native_loss=0;
    uint64_t reconstruction_targets=0;
    double reconstruction_loss=0;
};
#include "semantic_laces.hpp"
#include "routing_geometry.hpp"

struct Level {
    std::vector<Expert> experts;
    // Derived exact-search cache. Coordinates are contiguous across 16 experts
    // so SIMD preserves each dot product's original accumulation order.
    // Rebuilt after load; never serialized or counted as fitted parameters.
    std::vector<float> routing_keys;
    LaceBank laces{};
    RoutingState routing{};
    Shared p{};
    AddressParameters address{};
    RecallParameters retrieval{};
    RecallStream retained{};
    RecallRead read_now{};
    RecallCreditState recall_credit_state{};
    std::vector<Vec> recall_content_gradient;
    Stream s{};
    std::vector<Constituent> constituents;
    uint64_t credit_edges=0,content_updates=0;
    double task_gradient_energy=0,child_gradient_energy=0;
    // A 16-way hierarchy, stored in bottom-up layers of prototypes. Rebuilt
    // only at local update boundaries; bounded 16 comparisons per tree depth.
    std::vector<std::vector<Vec>> index;
};

constexpr int BUCKETS=256,WAYS=4,PAYLOAD=256,KEY_BYTES=64;
struct Record {
    Vec key{},value{};
    std::array<uint8_t,KEY_BYTES> exact_key{};
    std::array<uint8_t,PAYLOAD> payload{};
    uint64_t hash=0,born=0,generation=0;
    uint16_t key_size=0,payload_size=0;
    uint8_t valid=0;
};
struct AssociationStore {
    std::array<Record,BUCKETS*WAYS> records{};
    uint64_t clock=0,queries=0,candidates=0,writes=0;
    void clear(){
        // Keep generation history so pre-reset handles cannot alias new data.
        for(auto& r:records){uint64_t generation=r.generation;r=Record{};r.generation=generation;}
        clock=queries=candidates=writes=0;
    }
    static uint64_t hash_bytes(const uint8_t* p,size_t n){
        uint64_t h=1469598103934665603ULL;
        for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;
    }
    static int bucket(const Vec& key){int b=0;for(int j=0;j<8;++j)b|=(key[j]+key[j+16]-key[j+32]>0)<<j;return b;}
    uint64_t put(int b,const Vec& key,const Vec& value,const uint8_t* payload,size_t n,
                 const uint8_t* exact=nullptr,size_t key_n=0){
        if(n>PAYLOAD||key_n>KEY_BYTES)throw std::invalid_argument("association payload/key exceeds its declared bound");
        uint64_t hash=exact?hash_bytes(exact,key_n):0;
        int slot=b*WAYS;
        for(int i=b*WAYS;i<(b+1)*WAYS;++i){
            auto& r=records[i];
            if(exact&&r.valid&&r.key_size==key_n&&r.hash==hash&&std::memcmp(r.exact_key.data(),exact,key_n)==0){slot=i;break;}
            if(!r.valid){slot=i;break;}
            if(r.born<records[slot].born)slot=i;
        }
        auto& r=records[slot];uint64_t gen=r.generation+1;r=Record{};
        r.valid=1;r.key=key;r.value=value;r.born=++clock;r.generation=gen;r.hash=hash;
        r.key_size=uint16_t(key_n);r.payload_size=uint16_t(n);
        if(n)std::memcpy(r.payload.data(),payload,n);
        if(key_n)std::memcpy(r.exact_key.data(),exact,key_n);
        ++writes;return (gen<<16)|uint64_t(slot+1);
    }
    uint64_t bind(const uint8_t* key,size_t kn,const uint8_t* value,size_t vn){
        if(kn==0||kn>KEY_BYTES)throw std::invalid_argument("exact keys require 1..64 bytes");
        return put(hash_bytes(key,kn)%BUCKETS,Vec{},Vec{},value,vn,key,kn);
    }
    uint64_t lookup(const uint8_t* key,size_t n){
        if(n==0||n>KEY_BYTES)throw std::invalid_argument("exact keys require 1..64 bytes");
        ++queries;uint64_t h=hash_bytes(key,n);int b=h%BUCKETS;
        for(int i=b*WAYS;i<(b+1)*WAYS;++i){++candidates;auto& r=records[i];
            if(r.valid&&r.hash==h&&r.key_size==n&&std::memcmp(r.exact_key.data(),key,n)==0)return (r.generation<<16)|uint64_t(i+1);
        }return 0;
    }
    const Record* resolve(uint64_t handle)const{
        uint64_t slot=(handle&65535);if(!slot||slot>records.size())return nullptr;
        auto& r=records[slot-1];return r.valid&&r.generation==(handle>>16)?&r:nullptr;
    }
    Vec query(const Vec& q){
        ++queries;int b=bucket(q),best[2]{-1,-1};float scores[2]{.1f,.1f};
        // Four fixed neighboring buckets; at most sixteen candidates, top two.
        for(int mask:{0,1,2,4})for(int i=(b^mask)*WAYS;i<((b^mask)+1)*WAYS;++i){
            ++candidates;const auto& r=records[i];if(!r.valid||r.key_size)continue;
            float s=dot(q.data(),r.key.data())/std::sqrt((energy(q)+1e-8f)*(energy(r.key)+1e-8f));
            if(s>scores[0]){scores[1]=scores[0];best[1]=best[0];scores[0]=s;best[0]=i;}
            else if(s>scores[1]){scores[1]=s;best[1]=i;}
        }
        Vec v{};float total=0;
        for(int k=0;k<2;++k)if(best[k]>=0){total+=scores[k];for(int j=0;j<D;++j)v[j]+=scores[k]*records[best[k]].value[j];}
        if(total)for(auto& x:v)x/=total;
        return v;
    }
};

struct Model {
    std::vector<Level> levels;
    std::array<Vec,VOCAB> embedding{};
    std::array<Vec,255> decoder{};
    std::array<float,255> decoder_bias{};
    AssociationStore store{};
    Vec recalled{},previous_query{},cached_gain{},cached_bias{},features{};
    std::array<uint8_t,GROUP> payload{};
    int payload_count=0;
    bool semantic_laces=true;
    int routing_arm=0,swap_level=-1; // A/B/C/D = 0/1/2/3; swap is frozen-only.
    int balance_policy=2; // v8 uniform pilot; v9 residual starvation; v10 warmup only.
    int lace_intervention=0,lace_policy=2;
    uint64_t lace_training_bytes=0;
    bool training=true,indexed=true,associations=false,memory_writes=true,have_query=false;
    uint64_t bytes=0,route_comparisons=0;
    float learning_rate=.008f,slow_rate=.08f,query_rate=.008f,head_rate=.16f,aux_weight=.1f,route_rate=.002f;
    float constituent_credit=1;
    // Scale only language VJPs entering each native state. Local objectives
    // and gain/bias readout-head updates retain their independent rates.
    std::array<float,4> language_credit{{1,1,1,1}};
    std::array<float,4> recall_credit{{0,0,0,0}};
    int epoch_bytes=0,epoch_limit=BLOCK;
    float reconstruction_weight=.05f;
    int memory_mode=0,recall_mode=0;
    int mask=-1,shuffle=-1;
    explicit Model(int count=4,int target=1000000,uint32_t seed=7,bool use_index=true,bool memory=false);
    void rebuild_index(Level& l);
    void update_index(Level& l,int expert);
    void route(Level& l,Event& e);
    void routing_prepare(int scale,Event& e,int slot);
    void routing_target(int scale,const Vec& signature);
    void lace_route(int scale,Event& e,int slot);
    void lace_feedback(int scale);
    void lace_maintain();
    void forward_event(int scale,const std::array<Vec,GROUP>& child,const std::array<float,GROUP>& lengths,int byte);
    void flush(int scale);
    void retire_constituents();
    void drain_recall_credit(int scale,int slot);
    void refresh();
    void observe(uint8_t byte);
    float teach(uint8_t target);
    void log_probs(float* out)const;
    double run(const uint8_t* x,const uint8_t* y,size_t n);
    void reset(bool clear_store=false);
    void finish();
    void inference(bool cold);
    size_t parameter_count()const;
    uint32_t payload_version()const{
        return routing_arm?(routing_arm>=2&&balance_policy?(balance_policy==2?10:9):8):(lace_policy==1?6:7);
    }
    uint32_t checkpoint_version()const{
        if(recall_credit!=std::array<float,4>{{0,0,0,0}})return 12;
        return language_credit==std::array<float,4>{{1,1,1,1}}?payload_version():11;
    }
};

inline int coord(int a,int b){return a*8+b;}
inline float sign(int a,int b){return (a+b)%3==0?-1.f:1.f;}
inline void context_forward(Event& e,Memory& memory){
    e.memory=memory;float norm=1;
    for(int a=0;a<K;++a){float q=0,v=0;
        for(int b=0;b<K;++b){int j=coord(a,b);float s=sign(a,b)*.3535533905932738f;
            q+=s*e.content[j];v+=s*e.content[(j+19)%D];}
        e.key[a]=std::tanh(q);e.value[a]=std::tanh(v);norm+=e.key[a]*e.key[a];
    }
    for(int b=0;b<K;++b){float read=0;for(int a=0;a<K;++a)read+=memory[a*K+b]*e.key[a];
        e.read[b]=read;float err=e.value[b]-.98f*read;
        for(int a=0;a<K;++a)memory[a*K+b]=.98f*memory[a*K+b]+.5f*e.key[a]*err/norm;
    }
}
// Full local VJP through both the read and normalized delta write. The entering
// matrix is detached at the eight-event boundary, just like entering h.
inline void context_backward(const Event& e,const Small& read_grad,Memory& next,Vec& dc){
    Memory before{};Small dq{},dv{},dr=read_grad;float norm=1,dnorm=0;
    for(float q:e.key)norm+=q*q;
    for(int a=0;a<K;++a)for(int b=0;b<K;++b){
        float g=next[a*K+b],err=e.value[b]-.98f*e.read[b];
        before[a*K+b]+=.98f*g;dq[a]+=.5f*g*err/norm;
        dv[b]+=.5f*g*e.key[a]/norm;dnorm-=.5f*g*e.key[a]*err/(norm*norm);
    }
    for(int b=0;b<K;++b)dr[b]-=.98f*dv[b];
    for(int a=0;a<K;++a){dq[a]+=2*e.key[a]*dnorm;
        for(int b=0;b<K;++b){before[a*K+b]+=e.key[a]*dr[b];dq[a]+=e.memory[a*K+b]*dr[b];}
        dq[a]*=1-e.key[a]*e.key[a];dv[a]*=1-e.value[a]*e.value[a];
        for(int b=0;b<K;++b){int j=coord(a,b);float s=sign(a,b)*.3535533905932738f;
            dc[j]+=s*dq[a];dc[(j+19)%D]+=s*dv[a];}
    }next=before;
}

struct Writer {
    std::string out;
    template<class T> void pod(const T& v){static_assert(std::is_trivially_copyable<T>::value);out.append(reinterpret_cast<const char*>(&v),sizeof(v));}
    template<class T> void vector(const std::vector<T>& v){uint64_t n=v.size();pod(n);for(const auto& a:v)pod(a);}
};
struct Reader {
    const char* p;size_t n;
    template<class T> void pod(T& v){static_assert(std::is_trivially_copyable<T>::value);if(n<sizeof(v))throw std::runtime_error("truncated checkpoint");std::memcpy(&v,p,sizeof(v));p+=sizeof(v);n-=sizeof(v);}
    template<class T> void vector(std::vector<T>& v){uint64_t size;pod(size);if(size!=v.size())throw std::runtime_error("checkpoint topology mismatch");for(auto& a:v)pod(a);}
};
std::string save(const Model& m);
Model load(const char* p,size_t n);
} // namespace shoe
