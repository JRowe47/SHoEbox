// EXP03/05-A5's fixed hybrid, not a learned-proxy implementation.
// Proxy distance grounding: Movshovitz-Attias et al., ICCV 2017,
// https://arxiv.org/abs/1703.07464. Exact coefficients/arithmetic are registered
// in reports/exp0305a4/A5_REGISTRATION.md. Called only by local supervision.
using A5Codebook=std::array<LowRank,256>;
inline const A5Codebook& a5_codebook(){
    static const A5Codebook code=[](){A5Codebook c{};
        for(int b=0;b<256;++b)for(int r=0;r<QRANK;++r)c[b][r]=byte_code(b,r);
        return c;}();
    return code;
}
struct A5Objective {
    std::array<double,256> probabilities{};
    std::array<double,QRANK> mean{};
    LowRank gradient{};
    double attraction=0,discrimination=0;
};
inline A5Objective a5_objective(const LowRank& z,int observed){
    const auto& code=a5_codebook();A5Objective s;
    double largest=-std::numeric_limits<double>::infinity();
    for(int b=0;b<256;++b){double d=0;
        for(int r=0;r<QRANK;++r){double e=double(z[r])-double(code[b][r]);d+=e*e;}
        s.probabilities[b]=-4.*d;largest=std::max(largest,s.probabilities[b]);
    }
    double normalizer=0;
    for(int b=0;b<256;++b){s.probabilities[b]=std::exp(s.probabilities[b]-largest);normalizer+=s.probabilities[b];}
    for(int b=0;b<256;++b)s.probabilities[b]/=normalizer;
    for(int r=0;r<QRANK;++r){
        for(int b=0;b<256;++b)s.mean[r]+=s.probabilities[b]*double(code[b][r]);
        const double v=double(z[r]),t=double(code[observed][r]),e=v-t;
        s.attraction+=e*e;
        s.gradient[r]=float(.5*(v+s.mean[r]-2.*t)*(1.-v*v));
    }
    s.attraction*=.25;s.discrimination=-std::log(s.probabilities[observed])/16.;
    return s;
}
inline void a5_reconstruct(Affine* rows,const Vec& x,int observed,float rate){
    LowRank z{};
    for(int r=0;r<QRANK;++r)z[r]=std::tanh(affine(rows[r],x));
    const auto objective=a5_objective(z,observed);
    // All 16 errors use one pre-update z and distribution. Existing normalized
    // affine updates and their float32 summation order are untouched; no FMA.
    for(int r=0;r<QRANK;++r)affine_step(rows[r],x,objective.gradient[r],rate);
}
