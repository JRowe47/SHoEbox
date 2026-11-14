#pragma once
// Included in namespace shoe. Retained ordered pairs have fixed addresses and
// immutable observed values. Every byte can query every scale, independently
// of that scale's next completion time. No symbol has a task-specific meaning.
constexpr int RECALL_SLOTS=16, RECALL_K=16, RECALL_V=32;
using RecallKey=std::array<float,RECALL_K>;
using RecallValue=std::array<float,RECALL_V>;
inline RecallKey recall_pool(const Vec& v){
    RecallKey out{};
    for(int i=0;i<RECALL_K;++i)for(int j=0;j<D/RECALL_K;++j)
        out[i]+=(j%2?-.5f:.5f)*v[i+RECALL_K*j];
    return out;
}
struct RecallParameters {
    std::array<float,RECALL_K*D> query{};
    std::array<float,VOCAB*RECALL_K> query_byte{};
    std::array<float,255*RECALL_V> read{};
};
struct RecallGradient {
    std::array<float,RECALL_K*D> query{};
    RecallKey query_byte{}; // Only the currently observed byte's row is touched.
};
struct RecallUnit {
    Vec stable{};
    RecallKey key{};
    RecallValue value{};
    uint64_t born=0;
};
struct RecallStream {
    std::array<RecallUnit,RECALL_SLOTS> units{};
    Vec previous{},previous_content{},last{};bool have_previous=false;
    uint64_t clock=0;
    int count=0;
    void admit(const Vec& current,const Vec& content){
        if(!have_previous){previous=current;previous_content=content;have_previous=true;return;}
        Vec stable{},combined{};
        for(int i=0;i<D;++i){int j=(3*i+13)%D;
            stable[i]=current[i]+previous[j];combined[i]=std::tanh(.5f*(content[i]+previous_content[j]));}
        normalize(stable);previous=current;previous_content=content;
        // Repetition neither consumes slots nor ages unrelated retained units.
        // Reoccurrence after a different pair refreshes the pair's occurrence.
        if(clock&&stable==last)return;
        last=stable;
        int slot=0;
        for(;slot<count;++slot)if(units[slot].stable==stable)break;
        if(slot==count){
            if(count<RECALL_SLOTS)++count;
            else {slot=0;for(int i=1;i<count;++i)if(units[i].born<units[slot].born)slot=i;}
        }
        auto& u=units[slot];u.stable=stable;u.key=recall_pool(stable);u.born=++clock;
        float norm=std::sqrt(dot(u.key.data(),u.key.data(),RECALL_K)+1e-8f);
        for(float& x:u.key)x/=norm;
        auto learned=recall_pool(combined);
        for(int i=0;i<RECALL_K;++i){u.value[i]=std::tanh(learned[i]);u.value[RECALL_K+i]=u.key[i];}
    }
};
struct RecallRead {
    Vec input{};int byte=0;
    RecallKey query{};
    RecallValue value{};
    std::array<float,RECALL_SLOTS> weights{};
};
// Frozen interventions: 1 no read, 2 recency only (no learned query),
// 3 remove fixed values, 4 remove learned values. Addresses remain fixed.
inline void recall_forward(const RecallStream& s,const RecallParameters& p,const Vec& input,int byte,RecallRead& e,int mode=0){
    e=RecallRead{};e.input=input;e.byte=byte;if(!s.count)return;
    if(mode==1)return;
    for(int j=0;j<RECALL_K;++j)e.query[j]=std::tanh(p.query_byte[byte*RECALL_K+j]+dot(p.query.data()+j*D,input.data()));
    float mx=-1e30f,total=0;
    for(int k=0;k<s.count;++k){e.weights[k]=(mode==2?0:4*dot(e.query.data(),s.units[k].key.data(),RECALL_K))-.5f*float(s.clock-s.units[k].born);mx=std::max(mx,e.weights[k]);}
    for(int k=0;k<s.count;++k){e.weights[k]=std::exp(e.weights[k]-mx);total+=e.weights[k];}
    for(int k=0;k<s.count;++k){e.weights[k]/=total;for(int j=0;j<RECALL_V;++j)e.value[j]+=e.weights[k]*s.units[k].value[j];}
    if(mode==3)std::fill(e.value.begin()+RECALL_K,e.value.end(),0);
    if(mode==4)std::fill(e.value.begin(),e.value.begin()+RECALL_K,0);
}
inline void recall_backward(const RecallStream& s,const RecallParameters& p,const RecallRead& e,const RecallValue& dv,
                            RecallGradient& g,Vec& di){
    if(!s.count)return;
    RecallKey dq{};
    float mean=dot(dv.data(),e.value.data(),RECALL_V);
    for(int k=0;k<s.count;++k){float ds=4*e.weights[k]*(dot(dv.data(),s.units[k].value.data(),RECALL_V)-mean);
        for(int j=0;j<RECALL_K;++j)dq[j]+=ds*s.units[k].key[j];}
    for(int j=0;j<RECALL_K;++j){float dz=dq[j]*(1-e.query[j]*e.query[j]);g.query_byte[j]+=dz;
        for(int i=0;i<D;++i){g.query[j*D+i]+=dz*e.input[i];di[i]+=dz*p.query[j*D+i];}}
}
