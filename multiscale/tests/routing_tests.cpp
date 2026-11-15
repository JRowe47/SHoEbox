#include "../native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
int checks=0;
void check(bool v,const char* message){++checks;if(!v)throw std::runtime_error(message);}
void close(float a,float b,float tolerance,const char* message){check(std::abs(a-b)<tolerance,message);}
bool balance_roundtrip(const std::string& raw,int policy){
    auto loaded=load(raw.data(),raw.size());
    return loaded.balance_policy==policy&&save(loaded)==raw;
}
int main(){
    // Tiled arithmetic preserves scalar reduction order for every expert.
    float tiled[D*16]{},scores[16]{};Vec cue{};
    for(int j=0;j<D;++j){cue[j]=float((j*31)%37-18)/19;
        for(int k=0;k<16;++k)tiled[j*16+k]=float((j*17+k*13)%41-20)/23;}
    routing_dot16(tiled,cue.data(),scores);
    for(int k=0;k<16;++k){Vec key{};for(int j=0;j<D;++j)key[j]=tiled[j*16+k];
        check(scores[k]==dot(key.data(),cue.data()),"tiled dot changed float reduction order");}
    Model cached(1,1500000,7,false,false);cached.routing_arm=3;cached.training=false;
    Model scalar=cached;scalar.levels[0].routing_keys.clear();
    Event cached_event,scalar_event;cached_event.input=scalar_event.input=cue;
    cached.routing_prepare(0,cached_event,0);scalar.routing_prepare(0,scalar_event,0);
    check(cached_event.expert==scalar_event.expert&&cached_event.runner==scalar_event.runner&&cached_event.gate==scalar_event.gate,
          "cached exact search changed top two or gate");
    check(save(cached)==save(scalar),"derived routing cache entered checkpoint state");
    // Translation invariance with a frozen mean; normalized gate finite VJP.
    Model m(4,1000000,1,false,false);m.semantic_laces=false;m.routing_arm=1;m.training=false;
    auto& l=m.levels[2];Event e;e.input[0]=.5f;e.input[1]=-.2f;e.input[3]=.3f;
    m.routing_prepare(2,e,0);auto trace=l.routing.events[0];int id=e.expert;
    Event shifted=e;for(int i=0;i<D;++i){shifted.input[i]+=2;l.routing.mean[i]+=2;}
    m.routing_prepare(2,shifted,0);check(shifted.expert==id,"common component changed centered route");
    close(shifted.gate,e.gate,1e-5f,"gate not translation invariant");l.routing.mean={};
    Vec delta=trace.key_difference;for(auto& x:delta)x*=e.gate_derivative;
    Vec grad=routing_vjp(trace,delta);
    for(int j:{0,1,3,7}){
        auto a=e,b=e;a.input[j]+=.0002f;b.input[j]-=.0002f;
        m.routing_prepare(2,a,0);m.routing_prepare(2,b,0);
        close((a.gate-b.gate)/.0004f,grad[j],.001f,"gate input VJP");
    }
    check(e.gate_derivative>.017f,"calibrated derivative bound");
    // Bias participates in selection but is excluded from semantic gate.
    m.routing_arm=2;l.routing.bias.assign(l.experts.size(),0);l.routing.bias[3]=5;
    m.routing_prepare(2,e,0);check(e.expert==3,"selection bias ignored");
    check(e.gate_derivative>.017f,"bias saturated semantic gate");
    // Once warmup is over, ordinary nonuniform usage is not equalized.
    // With no starving expert, only the old balancing bias should decay.
    m.training=true;l.routing.seeded=1;l.routing.seen=100000;
    std::fill(l.routing.selections.begin(),l.routing.selections.end(),300);
    std::fill(l.routing.window.begin(),l.routing.window.end(),2);
    l.routing.window[0]=100;l.routing.window_count=255;
    std::fill(l.routing.bias.begin(),l.routing.bias.end(),0);
    l.routing.bias[0]=.2f;l.routing.bias[1]=-.2f;
    m.routing_prepare(2,e,0);
    close(l.routing.bias[0]-l.routing.bias[1],.4f*std::exp(-1.f/l.experts.size()),1e-5f,"postwarmup pressure still equalizes nonstarved experts");
    // A short absence is not starvation once coverage warmup has completed.
    // This includes banks larger than the entire 256-selection window.
    for(int budget:{1000000,5000000}){
        auto warm_storage=std::make_unique<Model>(4,budget,7,false,false);
        auto& warm=*warm_storage;warm.semantic_laces=false;warm.routing_arm=3;warm.balance_policy=2;
        auto& r=warm.levels[0].routing;r.seeded=1;r.seen=1000000;
        std::fill(r.selections.begin(),r.selections.end(),300);
        r.window_count=255;auto legacy_storage=std::make_unique<Model>(warm);
        auto& legacy=*legacy_storage;legacy.balance_policy=1;
        Event we,le;we.input=le.input=cue;
        warm.routing_prepare(0,we,0);legacy.routing_prepare(0,le,0);
        check(std::all_of(r.bias.begin(),r.bias.end(),[](float x){return x==0;}),"mature absence manufactured usage pressure");
        check(std::any_of(legacy.levels[0].routing.bias.begin(),legacy.levels[0].routing.bias.end(),[](float x){return x!=0;}),"historical v9 absence rule changed");
        r.bias[0]=.2f;r.bias[1]=-.2f;r.window_count=255;
        warm.routing_prepare(0,we,0);
        close(r.bias[0]-r.bias[1],.4f*std::exp(-1.f/r.bias.size()),1e-5f,"mature coverage bias did not decay");
        std::fill(r.bias.begin(),r.bias.end(),0);std::fill(r.selections.begin(),r.selections.end(),0);
        r.window_count=255;warm.routing_prepare(0,we,0);
        check(*std::min_element(r.bias.begin(),r.bias.end())<0&&*std::max_element(r.bias.begin(),r.bias.end())>0,"coverage warmup was removed");
        check(balance_roundtrip(save(warm),2),"v10 policy did not round trip");
        check(balance_roundtrip(save(legacy),1),"v9 policy silently migrated");
    }
    // Fixed code is nonzero, sparse, ordered, and independent of routing.
    std::array<Vec,GROUP> children;for(int j=0;j<GROUP;++j)children[j]=stable_bytes()[65+j];
    Vec code=routing_signature(stable_group(children));int active=0;for(float x:code)active+=x!=0;
    check(active>0&&active<=16,"sparse support");close(energy(code),1,1e-5f,"signature norm");
    std::swap(children[0],children[1]);check(code!=routing_signature(stable_group(children)),"child order lost");
    check(energy(routing_signature(Vec{}))>.99,"zero code protection");
    // Targets arrive only with the next observed same-scale event; resets clear
    // eligibility but preserve means, utilization and learned parameters.
    Model d(4,1000000,7,false,false);d.semantic_laces=false;d.routing_arm=3;
    uint8_t x[2003],y[2003];for(int i=0;i<2003;++i){x[i]=(i*13)%251;y[i]=(i*7)%251;}
    d.run(x,y,4);check(d.levels[1].routing.predict_targets==0,"future target used early");
    d.run(x+4,y+4,4);check(d.levels[1].routing.predict_targets==1,"next event target missing");
    auto head=d.levels[1].routing.predict;auto w=d.levels[1].experts;
    d.run(x+8,y+8,995);d.finish();check(d.levels[1].routing.predict!=head,"forecast head not trained");
    auto count=d.levels[1].routing.seen;auto mean=d.levels[1].routing.mean;
    auto predictor=d.levels[1].routing.predict;
    d.reset();check(d.levels[1].routing.seen==count&&d.levels[1].routing.mean==mean,"reset erased persistent routing");
    check(d.levels[1].routing.predict==predictor,"reset erased predictor");
    auto targets=d.levels[1].routing.predict_targets;d.run(x,y,4);
    check(d.levels[1].routing.predict_targets==targets,"target crossed reset");
    // Checkpoint state contains every new pending field and stream counter.
    d.run(x+4,y+4,999);auto raw=save(d);auto q=load(raw.data(),raw.size());
    check(d.run(x,y,1003)==q.run(x,y,1003),"continuation loss changed");check(save(d)==save(q),"continuation state changed");
    Model frozen=d;frozen.finish();frozen.inference(true);auto prior=frozen.levels[1].routing;
    frozen.run(x,y,1003);const auto& after=frozen.levels[1].routing;
    check(after.mean==prior.mean&&after.bias==prior.bias&&after.selections==prior.selections,"frozen statistics changed");
    check(after.predict==prior.predict&&after.predict_targets==prior.predict_targets,"frozen predictor changed");
    // Explicitly reject a swapping intervention during training.
    d.swap_level=2;bool rejected=false;try{d.run(x,y,1);}catch(const std::logic_error&){rejected=true;}
    check(rejected,"training accepted expert swap");
    // With byte-head, original auxiliary and reconstruction credit disabled,
    // the new prediction objective must still train selected expert matrices.
    Model only(4,1000000,19,false,false);only.semantic_laces=false;only.routing_arm=3;
    only.learning_rate=only.head_rate=only.query_rate=only.route_rate=only.aux_weight=only.reconstruction_weight=0;
    for(auto& level:only.levels){for(auto& v:level.retrieval.read)v=0;}
    for(auto& v:only.decoder)v={};
    auto experts_before=only.levels[2].experts;
    only.run(x,y,128);only.finish();bool changed=false;
    for(size_t k=0;k<experts_before.size();++k)changed|=experts_before[k].w!=only.levels[2].experts[k].w;
    check(changed,"predictive SDR objective does not reach selected experts");
    std::cout<<checks<<" routing checks passed\n";
}
