// EXP03/05-A4: reports/exp0305a3/A4_REGISTRATION.md, fixed before fitting.
// Paired addressing: Miller et al., https://aclanthology.org/D16-1147/.
// MQAR: Arora et al., https://proceedings.iclr.cc/paper_files/paper/2024/hash/
// 448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html.
// The query-relative band and chronological rule are this project's design.
struct A4Occurrence {int slot=-1,phase=0;float score=-1e30f;uint64_t born=0;};
struct A4Match {A4Occurrence occurrence{};LowRank key{};float distance=0;double log_weight=0;};
struct A4Scratch {
    std::array<A4Match,64> matches{};
    std::array<A4Occurrence,2> fallback{};
    std::array<int,2> best{{-1,-1}};
    Vec x{};
};
inline double a4_softplus(double x){return std::max(x,0.)+std::log1p(std::exp(-std::abs(x)));}
inline bool a4_stable_before(const A4Occurrence& a,const A4Occurrence& b){
    return a.slot<b.slot||(a.slot==b.slot&&a.phase<b.phase);
}
inline bool a4_weight_before(const A4Match& a,const A4Match& b){
    return a.log_weight>b.log_weight||(a.log_weight==b.log_weight&&
        a4_stable_before(a.occurrence,b.occurrence));
}
// All projections are per-call scratch; no persistent cache/checkpoint field.
inline void a4_select(const QueryHead& p,const RecallStream& records,QueryCache& c,
                      int lesion,bool exhaustive=false,int* projected=nullptr){
    A4Scratch s;int count=0,used=0;
    float d_min=std::numeric_limits<float>::infinity();
    for(int k=0;k<records.count;++k){
        s.x=source_vector(records.units[k].stable);if(lesion==3)s.x={};
        for(int ph=0;ph<4;++ph){
            A4Occurrence a{k,ph,affine(p.gate[ph],s.x),records.units[k].born};
            int at=a.score>s.fallback[0].score?0:(a.score>s.fallback[1].score?1:2);
            if(at<2){if(at==0)s.fallback[1]=s.fallback[0];s.fallback[at]=a;}
            if(a.score<=0)continue;
            auto& m=s.matches[count++];m.occurrence=a;
            // Every recognized candidate, even older than an overwhelming match,
            // contributes to d_min. Preserve float32 A2/A3 distance accumulation.
            for(int r=0;r<QRANK;++r){m.key[r]=std::tanh(affine(p.key[ph*QRANK+r],s.x));
                float d=m.key[r]-c.query[r];m.distance+=d*d;}
            d_min=std::min(d_min,m.distance);
        }
    }
    if(projected)*projected=count;
    std::sort(s.matches.begin(),s.matches.begin()+count,[](const auto& a,const auto& b){
        return a.occurrence.born!=b.occurrence.born?a.occurrence.born>b.occurrence.born:
            a4_stable_before(a.occurrence,b.occurrence);
    });
    double remaining=0;
    for(int i=0;i<count;++i){
        auto& m=s.matches[i];
        // Literal 1.96 = (2 * .7)^2, not a fitted distance percentile.
        const double logit=4.*(double(d_min)+1.96-double(m.distance));
        m.log_weight=remaining-a4_softplus(-logit);
        remaining-=a4_softplus(logit);
        auto better=[&](int k){return k>=used||a4_weight_before(m,s.matches[s.best[k]]);};
        int at=better(0)?0:(better(1)?1:2);
        if(at<2){if(at==0)s.best[1]=s.best[0];s.best[at]=i;used=std::min(2,used+1);}
        // Exact bound only skips scalar weights AFTER all keys/minimum exist.
        // Strict inequality preserves stable exact ties. Exhaustive test bypasses it.
        if(!exhaustive&&used==2&&remaining<s.matches[s.best[1]].log_weight)break;
    }
    if(count){
        for(int i=0;i<used;++i){const auto& m=s.matches[s.best[i]];const auto& a=m.occurrence;
            c.slots[i]=a.slot;c.phases[i]=a.phase;c.scores[i]=a.score;c.keys[i]=m.key;
            c.raw[i]=source_vector(records.units[a.slot].stable);if(lesion==3)c.raw[i]={};}
        if(used==1)c.attention=1;
        else c.attention=float(std::exp(-a4_softplus(s.matches[s.best[1]].log_weight-s.matches[s.best[0]].log_weight)));
    }else{
        // Complete A2/A3 historical fallback, including the empty-state result.
        for(int i=0;i<2;++i){const auto& a=s.fallback[i];
            if(a.slot>=0){c.slots[i]=a.slot;c.phases[i]=a.phase;c.scores[i]=a.score;
                c.raw[i]=source_vector(records.units[a.slot].stable);if(lesion==3)c.raw[i]={};}
            for(int r=0;r<QRANK;++r)c.keys[i][r]=std::tanh(affine(p.key[c.phases[i]*QRANK+r],c.raw[i]));
        }
        float difference=0;
        for(int r=0;r<QRANK;++r){float a=c.query[r]-c.keys[0][r],b=c.query[r]-c.keys[1][r];difference+=b*b-a*a;}
        c.attention=sigmoid(4*difference);
    }
}
