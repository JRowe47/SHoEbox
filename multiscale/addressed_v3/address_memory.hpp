#pragma once
// Included inside namespace shoe after the vector primitives in native.hpp.
// Fixed write coordinates; only the read and local feature decoder are fitted.
constexpr int ROTORS=8, TIME_D=2*ROTORS, CONTENT_K=2*K;
constexpr int ADDRESS_K=CONTENT_K+TIME_D, ADDRESS_V=2*K;
using AddressKey=std::array<float,ADDRESS_K>;
using AddressValue=std::array<float,ADDRESS_V>;
using AddressMatrix=std::array<float,ADDRESS_K*ADDRESS_V>;
using ClockVector=std::array<float,TIME_D>;

struct Clock {
    std::array<double,TIME_D> phase{};
    uint64_t ticks=0;
    Clock(){for(int i=0;i<ROTORS;++i)phase[2*i]=1/std::sqrt(double(ROTORS));}
    static const std::array<double,TIME_D>& rotations(){
        static const auto r=[](){std::array<double,TIME_D> a{};
            for(int i=0;i<ROTORS;++i){double period=4*std::pow(64.,double(i)/(ROTORS-1));
                double w=2*std::acos(-1.)/period;a[2*i]=std::cos(w);a[2*i+1]=std::sin(w);}return a;}();
        return r;
    }
    ClockVector value()const{ClockVector p{};for(int i=0;i<TIME_D;++i)p[i]=float(phase[i]);return p;}
    void advance(){
        const auto& r=rotations();
        for(int i=0;i<ROTORS;++i){double x=phase[2*i],y=phase[2*i+1];
            phase[2*i]=r[2*i]*x-r[2*i+1]*y;phase[2*i+1]=r[2*i+1]*x+r[2*i]*y;}
        // Bound roundoff in a persistent oscillator without changing its phase.
        if((++ticks&255)==0)for(int i=0;i<ROTORS;++i){
            double n=std::sqrt(ROTORS*(phase[2*i]*phase[2*i]+phase[2*i+1]*phase[2*i+1]));
            phase[2*i]/=n;phase[2*i+1]/=n;}
    }
};

inline const std::array<Vec,VOCAB>& stable_bytes(){
    static const auto table=[](){std::array<Vec,VOCAB> a{};
        for(int b=0;b<VOCAB;++b)for(int j=0;j<D;++j){
            uint32_t x=uint32_t(b+1)*0x9e3779b9U^uint32_t(j+1)*0x85ebca6bU;
            x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;
            a[b][j]=(x&1?1.f:-1.f)*.125f;}return a;}();
    return table;
}
inline Vec stable_group(const std::array<Vec,GROUP>& children){
    Vec v{};
    for(int p=0;p<GROUP;++p)for(int j=0;j<D;++j){
        int k=(j*(2*p+1)+13*p)%D;
        v[j]+=(j+p)%3==0?-.5f*children[p][k]:.5f*children[p][k];}
    normalize(v);return v;
}
inline Small fixed_pool(const Vec& content){
    Small a{};for(int i=0;i<K;++i)for(int j=0;j<K;++j)
        a[i]+=((i+j)%3==0?-1.f:1.f)*.3535533905932738f*content[i*K+j];
    return a;
}
inline const AddressKey& retention(){
    static const auto rates=[](){AddressKey a{};
        for(int i=0;i<CONTENT_K;++i){double half=32*std::pow(64.,double(i%K)/(K-1));
            a[i]=float(std::exp(-std::log(2.)/half));}
        // Equal decay within each sine/cosine pair preserves its phase geometry.
        for(int i=0;i<ROTORS;++i){double half=32*std::pow(64.,double(i)/(ROTORS-1));
            a[CONTENT_K+2*i]=a[CONTENT_K+2*i+1]=float(std::exp(-std::log(2.)/half));}
        return a;}();return rates;
}
struct AddressParameters {
    std::array<float,ADDRESS_K*D> query_content{},query_state{};
    std::array<float,TIME_D*TIME_D> query_clock{};
    std::array<float,D*ADDRESS_V> read{};
    Mat reconstruct{};
    std::array<float,GROUP*D> reconstruct_bias{};
};
struct AddressStream {
    AddressMatrix memory{};
    Vec previous_stable{};
    Clock clock{};
};
struct AddressEvent {
    AddressMatrix memory{};
    AddressKey key{},query{};
    AddressValue value{},read{};
    ClockVector clock{};
};

// mode: 0 full, 1 read removed, 2 temporal channel removed, 3 content channel
// removed. Nonzero modes are frozen-inference interventions only.
inline void address_forward(AddressEvent& e,const Vec& content,const Vec& before,
                            const Vec& stable,AddressStream& s,const AddressParameters& p,int mode=0){
    e=AddressEvent{};e.memory=s.memory;e.clock=s.clock.value();
    Small current=fixed_pool(stable),previous=fixed_pool(s.previous_stable),learned=fixed_pool(content);
    float cn=1e-8f,pn=1e-8f;for(int i=0;i<K;++i){cn+=current[i]*current[i];pn+=previous[i]*previous[i];}
    cn=1/std::sqrt(cn);pn=1/std::sqrt(pn);
    for(int i=0;i<K;++i){
        e.key[i]=current[i]*cn*.577350269f;e.key[K+i]=previous[i]*pn*.577350269f;
        e.value[i]=std::tanh(learned[i]);e.value[K+i]=current[i];
    }
    for(int i=0;i<TIME_D;++i)e.key[CONTENT_K+i]=e.clock[i]*.577350269f;
    for(int a=0;a<ADDRESS_K;++a){
        float z=dot(p.query_content.data()+a*D,content.data())+dot(p.query_state.data()+a*D,before.data());
        if(a<CONTENT_K)z+=.5f*current[a%K]*cn;
        else z+=dot(p.query_clock.data()+(a-CONTENT_K)*TIME_D,e.clock.data(),TIME_D);
        e.query[a]=std::tanh(z);
        if((mode==2&&a>=CONTENT_K)||(mode==3&&a<CONTENT_K))e.key[a]=e.query[a]=0;
        for(int b=0;b<ADDRESS_V;++b)e.read[b]+=e.query[a]*s.memory[a*ADDRESS_V+b];
    }
    if(mode==1)e.read={};
    const auto& decay=retention();
    // A coherent repeated write must converge, not grow like sqrt(time).
    // Predict from the decayed matrix and write only the residual. Keys and
    // their normalization are fixed input-derived constants for the VJP.
    AddressValue residual=e.value;float norm=1;
    for(float k:e.key)norm+=k*k;
    for(int a=0;a<ADDRESS_K;++a)for(int b=0;b<ADDRESS_V;++b)
        residual[b]-=e.key[a]*decay[a]*s.memory[a*ADDRESS_V+b];
    for(int a=0;a<ADDRESS_K;++a){
        for(int b=0;b<ADDRESS_V;++b)s.memory[a*ADDRESS_V+b]=decay[a]*s.memory[a*ADDRESS_V+b]+.5f*e.key[a]*residual[b]/norm;
    }
    s.previous_stable=stable;s.clock.advance();
}

// Exact local derivatives, including the residual write. Keys and clocks are fixed.
inline void address_backward(const AddressEvent& e,const Vec& content,const Vec& before,
                             const AddressParameters& p,const AddressValue& dr,AddressMatrix& carried,
                             Vec& dc,Vec& dh,AddressParameters& g){
    AddressMatrix previous{};AddressValue dv{};const auto& decay=retention();float norm=1;
    for(float k:e.key)norm+=k*k;
    for(int a=0;a<ADDRESS_K;++a)for(int b=0;b<ADDRESS_V;++b)dv[b]+=.5f*e.key[a]*carried[a*ADDRESS_V+b]/norm;
    for(int a=0;a<ADDRESS_K;++a){float dq=0;
        for(int b=0;b<ADDRESS_V;++b){int j=a*ADDRESS_V+b;
            dq+=e.memory[j]*dr[b];
            previous[j]=decay[a]*(carried[j]-e.key[a]*dv[b])+e.query[a]*dr[b];}
        float dz=dq*(1-e.query[a]*e.query[a]);
        for(int i=0;i<D;++i){g.query_content[a*D+i]+=dz*content[i];g.query_state[a*D+i]+=dz*before[i];
            dc[i]+=dz*p.query_content[a*D+i];dh[i]+=dz*p.query_state[a*D+i];}
        if(a>=CONTENT_K)for(int i=0;i<TIME_D;++i)g.query_clock[(a-CONTENT_K)*TIME_D+i]+=dz*e.clock[i];
    }
    for(int i=0;i<K;++i){float d=dv[i]*(1-e.value[i]*e.value[i]);
        for(int j=0;j<K;++j)dc[i*K+j]+=((i+j)%3==0?-1.f:1.f)*.3535533905932738f*d;}
    carried=previous;
}

inline float reconstruct(const Vec& content,const Vec& target,int role,const AddressParameters& p,
                         float weight,Vec& dc,AddressParameters& g){
    Vec input{},error{};for(int j=0;j<D;++j)input[j]=content[(j*(2*role+1)+13*role)%D];
    float loss=0;
    for(int i=0;i<D;++i){float y=std::tanh(dot(p.reconstruct.data()+i*D,input.data())+p.reconstruct_bias[role*D+i]);
        float d=y-target[i];loss+=.5f*d*d/D;error[i]=weight*d*(1-y*y)/D;
        g.reconstruct_bias[role*D+i]+=error[i];}
    outer(g.reconstruct,error,input);Vec di{};vjp(p.reconstruct,error,di);
    for(int j=0;j<D;++j)dc[(j*(2*role+1)+13*role)%D]+=di[j];
    return loss;
}
