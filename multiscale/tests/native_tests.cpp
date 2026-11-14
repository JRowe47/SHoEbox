#include "../native.hpp"
#include <iostream>
#include <memory>

using namespace shoe;
int checks=0;
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);++checks;}
double context_objective(const Event& e,const Memory& m,const Small& dr,const Memory& dm){
    // Independent double-precision forward oracle, avoiding float subtraction
    // noise when differencing a scalar objective with 128 derivative probes.
    double q[K]{},v[K]{},norm=1,result=0;
    for(int a=0;a<K;++a){
        for(int b=0;b<K;++b){int j=a*8+b;double s=((a+b)%3==0?-1.:1.)/std::sqrt(8.);
            q[a]+=s*e.content[j];v[a]+=s*e.content[(j+19)%D];}
        q[a]=std::tanh(q[a]);v[a]=std::tanh(v[a]);norm+=q[a]*q[a];
    }
    for(int b=0;b<K;++b){double read=0;for(int a=0;a<K;++a)read+=m[a*K+b]*q[a];result+=read*dr[b];
        for(int a=0;a<K;++a)result+=(.98*m[a*K+b]+.5*q[a]*(v[b]-.98*read)/norm)*dm[a*K+b];}
    return result;
}
void gradients(){
    Event e{};Memory memory{},incoming{};Small read_grad{};
    for(int i=0;i<D;++i){e.content[i]=.3f*std::sin(float(i));memory[i]=.1f*std::cos(float(i));incoming[i]=.2f*std::sin(float(i+2));}
    for(int j=0;j<K;++j)read_grad[j]=.15f*std::cos(float(j));
    Memory m=memory;context_forward(e,m);Memory dm=incoming;Vec dc{};context_backward(e,read_grad,dm,dc);
    const float eps=.001f;
    for(int j=0;j<D;++j){Event plus=e,minus=e;plus.content[j]+=eps;minus.content[j]-=eps;
        float num=(context_objective(plus,memory,read_grad,incoming)-context_objective(minus,memory,read_grad,incoming))/(2*eps);
        require(std::abs(num-dc[j])<2e-5f,"MinKron content VJP mismatch");
        Memory a=memory,b=memory;a[j]+=eps;b[j]-=eps;
        num=(context_objective(e,a,read_grad,incoming)-context_objective(e,b,read_grad,incoming))/(2*eps);
        require(std::abs(num-dm[j])<2e-5f,"MinKron carried-matrix VJP mismatch");
    }
}
// A complete eight-event byte objective, with fixed readout and hard routes,
// provides a numerical check of the exact local block derivative.
float block_loss(Model& m){
    uint8_t x[8]{31,7,31,16,8,2,16,31},y[8]{7,31,16,8,2,16,31,4};
    m.head_rate=0;m.route_rate=0;m.aux_weight=0;return float(m.run(x,y,8));
}
void block_gradient(){
    auto base=std::make_unique<Model>(1,100000,7,false,false);
    auto analytic=std::make_unique<Model>(*base);analytic->learning_rate=.01f;block_loss(*analytic);
    int expert=analytic->levels[0].s.events[0].expert;
    auto before=std::make_unique<Model>(*analytic);analytic->finish();
    struct Probe{int kind,index;};
    auto param=[&](Model& m,Probe p)->float&{
        if(p.kind==0)return m.levels[0].p.recurrent[p.index];
        if(p.kind==1)return m.levels[0].p.read[p.index];
        if(p.kind==2)return m.levels[0].experts[expert].w[p.index];
        return m.embedding[31][p.index];
    };
    for(Probe p:std::array<Probe,8>{{{0,3},{0,65},{1,9},{1,33},{2,7},{2,126},{3,12},{3,40}}}){
        const float eps=.002f;auto a=std::make_unique<Model>(*base),b=std::make_unique<Model>(*base);
        param(*a,p)+=eps;param(*b,p)-=eps;
        float numerical=(block_loss(*a)-block_loss(*b))/(2*eps);
        float actual=(param(*before,p)-param(*analytic,p))/.01f;
        require(std::abs(numerical-actual)<.0006f,"eight-event exact gradient mismatch");
    }
}
void slow_gradient(){
    auto base=std::make_unique<Model>(2,150000,7,false,false);
    std::vector<uint8_t> x(32),y(32);for(int t=0;t<32;++t){x[t]=uint8_t(t*7+13);y[t]=uint8_t(t*19+1);}
    auto analytic=std::make_unique<Model>(*base);analytic->learning_rate=0;analytic->head_rate=0;
    analytic->route_rate=0;analytic->aux_weight=0;analytic->run(x.data(),y.data(),32);
    int expert=analytic->levels[1].s.events[0].expert;
    auto before=std::make_unique<Model>(*analytic);analytic->learning_rate=.01f;analytic->finish();
    struct Probe{int kind,index;};
    auto param=[&](Model& m,Probe p)->float&{auto& l=m.levels[1];
        if(p.kind==0)return l.p.recurrent[p.index];if(p.kind==1)return l.p.role[p.index];
        if(p.kind==2)return l.experts[expert].w[p.index];if(p.kind==3)return l.p.gain[p.index];
        if(p.kind==4)return l.p.bias[p.index];if(p.kind==5)return l.p.offset[p.index];
        return l.p.length[p.index];
    };
    for(Probe p:std::array<Probe,10>{{{0,3},{0,65},{1,9},{1,133},{2,7},{3,126},{4,40},{5,12},{6,2},{6,23}}}){
        const float eps=.003f;auto a=std::make_unique<Model>(*base),b=std::make_unique<Model>(*base);
        param(*a,p)+=eps;param(*b,p)-=eps;a->training=b->training=false;
        // The first slow output appears at byte four: 29 served predictions.
        float numerical=float((32./29)*(a->run(x.data(),y.data(),32)-b->run(x.data(),y.data(),32))/(2*eps));
        float actual=(param(*before,p)-param(*analytic,p))/.01f;
        require(std::abs(numerical-actual)<.0008f,"slow composer/projector VJP mismatch");
    }
}
void contracts(){
    auto m=std::make_unique<Model>();std::vector<uint8_t> x(1031),y(1031);
    for(int i=0;i<1031;++i){x[i]=uint8_t(i*31);y[i]=uint8_t((i+1)*31);}
    m->run(x.data(),y.data(),519);
    require(m->levels[0].s.executions==519&&m->levels[1].s.executions==129&&m->levels[2].s.executions==32&&m->levels[3].s.executions==8,"native clocks");
    auto serialized=save(*m);auto clone=std::make_unique<Model>(load(serialized.data(),serialized.size()));
    double a=m->run(x.data()+519,y.data()+519,512),b=clone->run(x.data()+519,y.data()+519,512);
    require(a==b&&save(*m)==save(*clone),"pending-ring checkpoint continuation");
    auto frozen=std::make_unique<Model>(*m);frozen->inference(false);auto weights=frozen->embedding;
    frozen->run(x.data(),y.data(),1031);require(frozen->embedding==weights,"inference froze weights");
    for(auto& l:frozen->levels)require(l.s.count==0,"inference derivative ring absent");
    std::array<float,256> logs{};frozen->log_probs(logs.data());double total=0;for(float v:logs)total+=std::exp(v);
    require(std::abs(total-1)<1e-6,"byte decoder normalized");
    // Canonical content depends on children, not the recurrent state.
    std::array<Vec,GROUP> child{};std::array<float,GROUP> length{1,1,1,1};
    for(int k=0;k<GROUP;++k)for(int j=0;j<D;++j)child[k][j]=.3f*std::sin(float(k*17+j));
    auto c1=std::make_unique<Model>(2),c2=std::make_unique<Model>(*c1);c1->training=c2->training=false;
    c2->levels[1].s.state.fill(.8f);c2->levels[1].s.memory.fill(.1f);
    c1->forward_event(1,child,length,0);c2->forward_event(1,child,length,0);
    require(c1->levels[1].s.content==c2->levels[1].s.content,"content/context separation");
    require(c1->levels[1].s.state!=c2->levels[1].s.state,"recurrent context contributes");
    auto old=c1->levels[1].s.content;std::swap(child[0],child[1]);c1->forward_event(1,child,length,0);
    require(old!=c1->levels[1].s.content,"ordered composition");
    for(size_t n:{size_t(0),serialized.size()/2,serialized.size()-1}){
        bool caught=false;try{auto bad=load(serialized.data(),n);(void)bad;}catch(const std::exception&){caught=true;}
        require(caught,"truncated checkpoint rejected");
    }
    // Only selected banks (and their top-two routing keys) may change.
    auto sparse=std::make_unique<Model>(1,1000000);auto original=sparse->levels[0].experts;
    sparse->run(x.data(),y.data(),8);std::array<bool,1500> touched{};
    for(const auto& e:sparse->levels[0].s.events)touched[e.expert]=true;sparse->finish();
    for(size_t i=0;i<original.size();++i)if(!touched[i])require(original[i].w==sparse->levels[0].experts[i].w,"unselected weight traffic");
    auto large=std::make_unique<Model>(4,5000000);large->run(x.data(),y.data(),1024);
    uint64_t events=0;for(auto& l:large->levels)events+=l.s.executions;
    require(large->route_comparisons<=events*48,"indexed routing comparison bound");
    require(large->parameter_count()>=4995000&&large->parameter_count()<=5000000,"5M fitted parameter accounting");
}
void associations(){
    auto s=std::make_unique<AssociationStore>();const uint8_t key[]{'n','e','w'},payload[]{0,255,42,12};
    require(s->lookup(key,3)==0,"unobserved association absent");uint64_t handle=s->bind(key,3,payload,4);
    require(s->lookup(key,3)==handle,"exact association lookup");auto* record=s->resolve(handle);
    require(record&&record->payload_size==4&&std::memcmp(record->payload.data(),payload,4)==0,"exact binary payload");
    auto bucket=AssociationStore::hash_bytes(key,3)%BUCKETS;int collisions=0;
    for(uint32_t i=0;collisions<4;++i)if(AssociationStore::hash_bytes(reinterpret_cast<uint8_t*>(&i),4)%BUCKETS==bucket){s->bind(reinterpret_cast<uint8_t*>(&i),4,payload,4);++collisions;}
    require(!s->resolve(handle),"evicted handle rejected");
    Vec q{};q.fill(.2f);auto before=s->candidates;s->query(q);require(s->candidates-before==16,"semantic candidate bound");
    uint8_t long_value[257]{};bool rejected=false;try{s->bind(key,3,long_value,257);}catch(const std::exception&){rejected=true;}
    require(rejected,"oversized payload rejected");
}
int main(){try{gradients();block_gradient();slow_gradient();contracts();associations();std::cout<<"PASS "<<checks<<" native checks\n";return 0;}
catch(const std::exception& e){std::cerr<<"FAIL after "<<checks<<" checks: "<<e.what()<<"\n";return 1;}}
