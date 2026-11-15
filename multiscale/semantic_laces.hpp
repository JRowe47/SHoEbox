#pragma once
// Persistent sparse semantic route identities. Stream state is separate from
// learned allocations; all mutation is training-only and splits retire safely.
constexpr int LACE_SLOTS=128, LACE_MIN_HITS=32;
struct SparseCode {
    std::array<uint64_t,10> bits{}; // four ordered child roles + prior context
    std::array<int,GROUP> path{{-1,-1,-1,-1}};
};
inline void lace_top4(const Vec& value,const Vec& mean,uint64_t* bits){
    float scores[4]{-1,-1,-1,-1};int ids[4]{};
    for(int i=0;i<16;++i){float v=0;for(int j=0;j<4;++j)v+=(j%2?-1.f:1.f)*(value[i+16*j]-mean[i+16*j])*.5f;float a=std::abs(v);
        for(int k=0;k<4;++k)if(a>scores[k]){for(int j=3;j>k;--j){scores[j]=scores[j-1];ids[j]=ids[j-1];}
            scores[k]=a;ids[k]=2*i+(v<0);break;}}
    bits[0]=bits[1]=0;for(int id:ids)bits[id/64]|=uint64_t(1)<<(id%64);
}
inline float lace_similarity(const SparseCode& a,const SparseCode& b){
    int content=0,context=0,path=0;
    for(int i=0;i<8;++i)content+=__builtin_popcountll(a.bits[i]&b.bits[i]);
    for(int i=8;i<10;++i)context+=__builtin_popcountll(a.bits[i]&b.bits[i]);
    for(int i=0;i<GROUP;++i)path+=a.path[i]==b.path[i];
    return .7f*content/16+.1f*context/4+.2f*path/4;
}
struct SemanticLace {
    SparseCode code{};
    uint64_t identity=0,last=0,hits=0;
    double loss=0,control_sum=0,trial_sum=0,control_sq=0,trial_sq=0;
    uint32_t control_n=0,trial_n=0,attempts=0;
    int incumbent=-1,specialist=-1,status=0; // 0 observer, 1 trial, 2 retained
};
struct LaceEvent { int record=-1,arm=0;double loss=0;uint32_t targets=0; };
struct LaceStream {
    std::array<LaceEvent,BLOCK> events{};
    std::array<int,GROUP> children{{-1,-1,-1,-1}};
};
struct LaceBank {
    std::array<SemanticLace,LACE_SLOTS> records{};
    std::vector<uint64_t> natural_use,last_use;
    std::vector<double> expert_loss;
    std::vector<uint8_t> reserved;
    Vec content_mean{},context_mean{};
    LaceStream stream{};
    uint64_t clock=0,next_identity=1,matches=0,misses=0,allocations=0,promotions=0,rejections=0;
    uint64_t trial_routes=0,retained_routes=0,no_spare=0;
    void initialize(int n){natural_use.assign(n,0);last_use.assign(n,0);expert_loss.assign(n,0);reserved.assign(n,0);}
};
