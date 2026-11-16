// EXP03/05-A3: the exact fixed rule in reports/exp0305a2/A3_REGISTRATION.md.
// Paired addressing: Miller et al., https://aclanthology.org/D16-1147/.
// MQAR: Arora et al., https://proceedings.iclr.cc/paper_files/paper/2024/hash/
// 448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html.
// Chronological probability matching and its constants are this project's rule.
struct A3Occurrence {int slot=-1,phase=0;float score=-1e30f;uint64_t born=0;};
struct A3Match {A3Occurrence occurrence{};LowRank key{};double log_weight=0;};
struct A3Scratch {
    std::array<A3Occurrence,64> occurrences{};
    std::array<A3Occurrence,2> fallback{};
    std::array<A3Match,2> best{};
    Vec x{};LowRank key{};
};
inline double a3_softplus(double x){return std::max(x,0.)+std::log1p(std::exp(-std::abs(x)));}
inline bool a3_stable_before(const A3Occurrence& a,const A3Occurrence& b){
    return a.slot<b.slot||(a.slot==b.slot&&a.phase<b.phase);
}
// Scratch is per call; no persistent cache or new checkpoint field.
inline void a3_select(const QueryHead& p,const RecallStream& records,QueryCache& c,
                      int lesion,bool exhaustive=false,int* projected=nullptr){
    A3Scratch s;int count=0,used=0,projections=0;
    for(int k=0;k<records.count;++k){
        s.x=source_vector(records.units[k].stable);if(lesion==3)s.x={};
        for(int ph=0;ph<4;++ph){
            A3Occurrence a{k,ph,affine(p.gate[ph],s.x),records.units[k].born};
            int at=a.score>s.fallback[0].score?0:(a.score>s.fallback[1].score?1:2);
            if(at<2){if(at==0)s.fallback[1]=s.fallback[0];s.fallback[at]=a;}
            if(a.score>0)s.occurrences[count++]=a;
        }
    }
    std::sort(s.occurrences.begin(),s.occurrences.begin()+count,[](const auto& a,const auto& b){
        return a.born!=b.born?a.born>b.born:a3_stable_before(a,b);
    });
    double remaining=0;
    for(int i=0;i<count;++i){
        const auto& a=s.occurrences[i];s.x=source_vector(records.units[a.slot].stable);if(lesion==3)s.x={};
        float distance=0;
        for(int r=0;r<QRANK;++r){s.key[r]=std::tanh(affine(p.key[a.phase*QRANK+r],s.x));
            float d=s.key[r]-c.query[r];distance+=d*d;}
        ++projections;
        // Float32 distance, then literal-double scalar and log-domain arithmetic.
        const double logit=4.*(7.84-double(distance));
        const double log_weight=remaining-a3_softplus(-logit);
        remaining-=a3_softplus(logit);
        auto better=[&](int k){return k>=used||log_weight>s.best[k].log_weight||
            (log_weight==s.best[k].log_weight&&a3_stable_before(a,s.best[k].occurrence));};
        int at=better(0)?0:(better(1)?1:2);
        if(at<2){if(at==0)s.best[1]=s.best[0];s.best[at]={a,s.key,log_weight};used=std::min(2,used+1);}
        // Every later log weight is bounded above by this remaining log mass.
        // Strict inequality preserves exact ties and their stable ordering.
        if(!exhaustive&&used==2&&remaining<s.best[1].log_weight)break;
    }
    if(projected)*projected=projections;
    if(count){
        for(int i=0;i<used;++i){const auto& a=s.best[i].occurrence;
            c.slots[i]=a.slot;c.phases[i]=a.phase;c.scores[i]=a.score;c.keys[i]=s.best[i].key;
            c.raw[i]=source_vector(records.units[a.slot].stable);if(lesion==3)c.raw[i]={};}
        if(used==1)c.attention=1;
        else {const double d=s.best[1].log_weight-s.best[0].log_weight;
            c.attention=float(std::exp(-a3_softplus(d)));}
    }else{
        // Complete A2 historical presence fallback, including empty-state behavior.
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
