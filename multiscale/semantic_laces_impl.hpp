void Model::lace_route(int scale,Event& e,int slot){
    auto& l=levels[scale];auto& b=l.laces;
    if(lace_intervention==1)return;
    SparseCode code;code.path=b.stream.children;
    for(int k=0;k<GROUP;++k)lace_top4(e.children[k],b.content_mean,&code.bits[k*2]);
    lace_top4(e.before,b.context_mean,&code.bits[8]);
    int chosen=-1;float best=scale==1?.49999f:.42499f;
    for(int k=0;k<LACE_SLOTS;++k){const auto& r=b.records[k];
        if(!r.identity||(r.status!=2&&r.incumbent!=e.expert))continue;
        float score=lace_similarity(code,r.code);
        if(score>best){best=score;chosen=k;}
    }
    if(training){
        ++b.clock;++b.natural_use[e.expert];b.last_use[e.expert]=b.clock;
        auto& credit=b.stream.events[slot];credit=LaceEvent{};
        if(chosen<0){
            ++b.misses;uint64_t oldest=UINT64_MAX;
            for(int k=0;k<LACE_SLOTS;++k){const auto& r=b.records[k];if(r.status)continue;
                bool pending=false;for(int t=0;t<l.s.count;++t)pending|=b.stream.events[t].record==k;
                if(pending)continue;
                if(!r.identity){chosen=k;break;}
                uint64_t retention=r.last+std::min(uint64_t(64),r.hits)*256;
                if(retention<oldest){oldest=retention;chosen=k;}
            }
            if(chosen>=0){auto& r=b.records[chosen];r=SemanticLace{};r.code=code;
                r.identity=b.next_identity++;r.incumbent=e.expert;}
        }else ++b.matches;
        if(chosen>=0){auto& r=b.records[chosen];++r.hits;r.last=b.clock;credit.record=chosen;
            if(lace_policy>=2&&r.hits%16==0)r.code.bits=code.bits;}
        // Slow centering removes the shared direction without replacing learned
        // content by raw bytes or a random hash. Addresses retain stable IDs.
        for(int j=0;j<D;++j){float v=0;for(int k=0;k<GROUP;++k)v+=e.children[k][j]*.25f;
            b.content_mean[j]+=.001f*(v-b.content_mean[j]);
            b.context_mean[j]+=.001f*(e.before[j]-b.context_mean[j]);}
    }
    if(chosen<0)return;
    auto& r=b.records[chosen];if(!r.status)return;
    // Eight-visit blocks give both arms consecutive local learning credit.
    // Frozen evaluation uses the incumbent until the trial has won.
    bool trial=r.status==2||(training&&(((r.hits/8)%4==1)||((r.hits/8)%4==2)));
    e.expert=trial?r.specialist:r.incumbent;e.runner=e.expert;e.gate_derivative=0;
    if(training){b.stream.events[slot].arm=trial;
        if(trial){if(r.status==2)++b.retained_routes;else ++b.trial_routes;}}
}

void Model::lace_feedback(int scale){
    auto& l=levels[scale];auto& b=l.laces;
    for(int t=0;t<l.s.count;++t){auto& e=b.stream.events[t];
        if(e.record<0||!e.targets)continue;
        auto& r=b.records[e.record];double loss=e.loss/e.targets;
        r.loss=r.loss==0?loss:r.loss+.03*(loss-r.loss);
        auto& mean=b.expert_loss[r.incumbent];mean=mean==0?loss:mean+.003*(loss-mean);
        if(r.status==1){if(e.arm){r.trial_sum+=loss;r.trial_sq+=loss*loss;++r.trial_n;}else{r.control_sum+=loss;r.control_sq+=loss*loss;++r.control_n;}}
    }
}

void Model::lace_maintain(){
    // Invoked after every eligibility tree has retired, never with references
    // to pre-copy expert weights outstanding. No extra neural forward pass.
    for(size_t scale=1;scale<levels.size();++scale){auto& l=levels[scale];auto& b=l.laces;
        int trials=0;
        for(auto& r:b.records)if(r.status==1){
            if(lace_policy>=2&&b.clock-r.last>2048){r.status=0;r.specialist=-1;++b.rejections;continue;}
            if(r.control_n>=128&&r.trial_n>=128){
                double gain=r.control_sum/r.control_n-r.trial_sum/r.trial_n;
                double variance=std::max(0.,r.control_sq/r.control_n-std::pow(r.control_sum/r.control_n,2))/r.control_n
                    +std::max(0.,r.trial_sq/r.trial_n-std::pow(r.trial_sum/r.trial_n,2))/r.trial_n;
                if(gain>.02+std::sqrt(variance)){r.status=2;++b.promotions;}
                else if(r.control_n>=512&&r.trial_n>=512){
                    r.status=0;r.specialist=-1;++b.rejections;
                    // Keep the slot reserved from ordinary routing. A later
                    // residual collision may reuse it by copying a new parent.
                }
            }
            trials+=r.status==1;
        }
        if(lace_training_bytes<200000||trials>=2)continue;
        int candidate=-1;double priority=0;
        for(int k=0;k<LACE_SLOTS;++k){const auto& r=b.records[k];
            if(!r.identity||r.status||r.attempts||r.hits<LACE_MIN_HITS||b.clock-r.last>2048)continue;
            double residual=r.loss-b.expert_loss[r.incumbent];
            if(residual<=.15)continue;
            int distinct=0;for(const auto& other:b.records)
                distinct+=other.identity&&other.incumbent==r.incumbent&&other.hits>=8;
            if(distinct<2)continue;
            double score=residual*std::sqrt(double(std::min(uint64_t(256),r.hits)));
            if(score>priority){priority=score;candidate=k;}
        }
        if(candidate<0)continue;
        int spare=-1;
        for(int k=0;k<int(l.experts.size());++k){
            bool owned=false;for(const auto& r:b.records)owned|=r.status&&(r.specialist==k||(lace_policy>=2&&r.incumbent==k));
            if(k!=b.records[candidate].incumbent&&!owned&&((b.reserved[k])||!b.natural_use[k]||b.clock-b.last_use[k]>std::max(uint64_t(256),b.clock/20))){spare=k;break;}
        }
        if(spare<0){++b.no_spare;continue;}
        auto& r=b.records[candidate];
        l.experts[spare]=l.experts[r.incumbent];b.reserved[spare]=1;
        r.specialist=spare;r.status=1;r.control_n=r.trial_n=0;r.control_sum=r.trial_sum=r.control_sq=r.trial_sq=0;++r.attempts;
        ++b.allocations;rebuild_index(l);
    }
}
