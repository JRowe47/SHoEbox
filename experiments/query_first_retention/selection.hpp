// EXP03/05-A2, registered query-dependent selection BEFORE truncation.
// Miller et al. (EMNLP 2016), https://aclanthology.org/D16-1147/:
// paired addressing precedes value mixing. This precise routing rule is ours.
// All projections use existing continuous features; no symbol tests or parser.
inline void a2_select(const QueryHead& p,const RecallStream& records,QueryCache& c,int lesion){
    struct Candidate {int slot=-1,phase=0;float score=-1e30f,distance=0;uint64_t born=0;};
    std::array<Candidate,2> ranked{},fallback{};
    auto better=[](const Candidate& a,const Candidate& b){
        if(b.slot<0)return true;
        if(a.distance!=b.distance)return a.distance<b.distance;
        if(a.born!=b.born)return a.born>b.born;
        return a.slot<b.slot||(a.slot==b.slot&&a.phase<b.phase);
    };
    for(int k=0;k<records.count;++k){
        Vec x=source_vector(records.units[k].stable);if(lesion==3)x={};
        for(int ph=0;ph<4;++ph){
            Candidate a{k,ph,affine(p.gate[ph],x),0,records.units[k].born};
            int at=a.score>fallback[0].score?0:(a.score>fallback[1].score?1:2);
            if(at<2){if(at==0)fallback[1]=fallback[0];fallback[at]=a;}
            if(a.score<=0)continue;
            for(int r=0;r<QRANK;++r){float d=std::tanh(affine(p.key[ph*QRANK+r],x))-c.query[r];a.distance+=d*d;}
            at=better(a,ranked[0])?0:(better(a,ranked[1])?1:2);
            if(at<2){if(at==0)ranked[1]=ranked[0];ranked[at]=a;}
        }
    }
    int used=(ranked[0].slot>=0)+(ranked[1].slot>=0);
    for(const auto& a:fallback){
        if(used==2)break;if(a.slot<0)continue;
        bool duplicate=false;for(int i=0;i<used;++i)duplicate|=a.slot==ranked[i].slot&&a.phase==ranked[i].phase;
        if(!duplicate)ranked[used++]=a;
    }
    for(int i=0;i<used;++i){const auto& a=ranked[i];c.slots[i]=a.slot;c.phases[i]=a.phase;c.scores[i]=a.score;
        c.raw[i]=source_vector(records.units[a.slot].stable);if(lesion==3)c.raw[i]={};}
}
