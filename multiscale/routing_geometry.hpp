#pragma once
// Included inside namespace shoe. Side records preserve the v5-v7 event ABI.
struct RoutingTrace {
    Vec cue{},expert_input{},signature{},forecast{},prediction_grad{},key_difference{};
    float norm=1;
    int target=0;
};
struct RoutingState {
    Vec mean{};
    std::vector<Vec> seed_cues;
    std::vector<float> bias;
    std::vector<uint64_t> selections,window;
    std::array<RoutingTrace,BLOCK> events{};
    std::vector<RoutingTrace> constituents;
    Mat predict{};
    uint64_t seen=0,predict_targets=0;
    double predict_loss=0;
    int seeded=0,seed_count=0,window_count=0;
    void initialize(size_t n) {
        bias.resize(n);selections.resize(n);window.resize(n);seed_cues.resize(4*n);
        for(int j=0;j<D;++j)predict[j*D+j]=.15f;
    }
    void reset_stream(){events={};constituents.clear();}
};

// Fixed ordered child-content coordinates, independent of expert identities.
// Top-16 signed, valued coordinates keep a nonzero code; the encoder cannot
// learn a constant output. Existing fixed byte/ordered-composition maps define
// the stable target space. Context is a separate, detached input channel.
inline Vec routing_signature(const Vec& stable){
    std::array<int,D> order{};for(int j=0;j<D;++j)order[j]=j;
    std::partial_sort(order.begin(),order.begin()+16,order.end(),[&](int a,int b){
        return std::abs(stable[a])==std::abs(stable[b])?a<b:std::abs(stable[a])>std::abs(stable[b]);});
    Vec code{};for(int j=0;j<16;++j)code[order[j]]=stable[order[j]];
    if(energy(code)<1e-12f)code[0]=1;
    normalize(code);return code;
}
inline Vec routing_vjp(const RoutingTrace& r,const Vec& delta){
    Vec out{};float radial=dot(r.cue.data(),delta.data());
    for(int j=0;j<D;++j)out[j]=(delta[j]-radial*r.cue[j])/r.norm;
    return out;
}
