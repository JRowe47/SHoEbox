#pragma once
void Model::routing_prepare(int scale,Event& e,int slot){
    auto& l=levels[scale];auto& r=l.routing;auto& t=r.events[slot];t=RoutingTrace{};
    t.expert_input=e.input;
    Vec prior_mean=r.mean,extra{};
    if(routing_arm==3&&scale>0){
        t.signature=routing_signature(l.s.stable);
        Vec context=e.before;normalize(context);
        for(int j=0;j<D;++j){
            // Fixed projections into the existing 64-D compute interface.
            extra[j]=.5f*t.signature[j]+.125f*context[(3*j+17)%D];
            t.expert_input[j]+=.25f*t.signature[j]+.125f*context[(3*j+17)%D];
        }
    }
    // Means and balancing statistics only observe training inputs.
    if(training){
        float alpha=1.f/std::min<uint64_t>(1024,r.seen+1);
        for(int j=0;j<D;++j)r.mean[j]+=alpha*(e.input[j]-r.mean[j]);
        if(!r.seeded){
            r.seed_cues[r.seed_count]=e.input;
            for(int j=0;j<D;++j)r.seed_cues[r.seed_count][j]+=extra[j];
            ++r.seed_count;
            if(r.seed_count==int(r.seed_cues.size())){
                // Deterministic farthest-point initialization from actual cues.
                auto cues=r.seed_cues;
                for(auto& q:cues){for(int j=0;j<D;++j)q[j]-=r.mean[j];normalize(q);}
                std::vector<float> nearest(cues.size(),-2.f);size_t pick=0;
                for(size_t k=0;k<l.experts.size();++k){
                    l.experts[k].key=cues[pick];
                    if(energy(l.experts[k].key)<1e-10f)l.experts[k].key[k%D]=1;
                    for(size_t i=0;i<cues.size();++i)nearest[i]=std::max(nearest[i],dot(cues[i].data(),l.experts[k].key.data()));
                    pick=std::min_element(nearest.begin(),nearest.end())-nearest.begin();
                }
                r.seeded=1;rebuild_index(l);
            }
        }
    }
    for(int j=0;j<D;++j)t.cue[j]=e.input[j]-prior_mean[j]+extra[j];
    t.norm=std::sqrt(energy(t.cue)+1e-4f);for(auto& x:t.cue)x/=t.norm;
    // Normalized cue and unit prototypes bound semantic scores to [-1,1].
    // Past-use bias changes selection, not gate magnitude/gradient.
    int best=0,second=0;float a=-1e30f,b=-1e30f;
    for(size_t k=0;k<l.experts.size();++k){
        ++route_comparisons;float score=dot(l.experts[k].key.data(),t.cue.data())+(routing_arm>=2?r.bias[k]:0.f);
        if(score>a){b=a;second=best;a=score;best=int(k);}else if(score>b){b=score;second=int(k);}
    }
    e.expert=best;e.runner=second;
    for(int j=0;j<D;++j)t.key_difference[j]=l.experts[best].key[j]-l.experts[second].key[j];
    float margin=dot(l.experts[best].key.data(),t.cue.data())-dot(l.experts[second].key.data(),t.cue.data());
    float gate=std::tanh(margin);e.gate=1+.25f*gate;e.gate_derivative=.25f*(1-gate*gate);
    if(training){
        ++r.seen;++r.selections[best];
        if(routing_arm>=2){
            ++r.window[best];++r.window_count;
            if(r.window_count==256){
                // A level advances by its own minimum selections/expert.
                // Strong correction fades after every expert has 256 examples.
                auto minimum=*std::min_element(r.selections.begin(),r.selections.end());
                float warm=std::clamp(1.f-float(minimum)/256.f,0.f,1.f);
                float expected=256.f/r.bias.size(),average=0;
                for(size_t k=0;k<r.bias.size();++k){
                    if(balance_policy==0){ // Exact v8 pilot continuation.
                        float rate=.002f+.038f*warm;
                        r.bias[k]+=rate*std::clamp((expected-float(r.window[k]))/expected,-1.f,1.f);
                    }else{
                        // Retire warmup's bias on a 256-selections/expert
                        // horizon. Afterward only a <10%-of-uniform usage
                        // deficit earns a small positive starvation nudge.
                        float decay=std::exp(-(1-warm)/float(r.bias.size()));
                        float uniform=std::clamp((expected-float(r.window[k]))/expected,-1.f,1.f);
                        float starvation=std::max(0.f,1-float(r.window[k])/(.1f*expected));
                        r.bias[k]=decay*r.bias[k]+.04f*warm*uniform+.002f*(1-warm)*starvation;
                    }
                    r.bias[k]=std::clamp(r.bias[k],-.5f,.5f);average+=r.bias[k];r.window[k]=0;
                }
                average/=r.bias.size();for(auto& x:r.bias)x-=average;r.window_count=0;
            }
        }
    }
}

void Model::routing_target(int scale,const Vec& signature){
    auto& l=levels[scale];auto& s=l.s;if(!s.count)return;
    auto& t=l.routing.events[s.count-1];
    if(t.target)return;
    for(int j=0;j<D;++j){float d=t.forecast[j]-signature[j];t.prediction_grad[j]=d;l.routing.predict_loss+=.5*double(d)*d;}
    t.target=1;++l.routing.predict_targets;
}
