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
    for(auto& l:base->levels)l.retrieval.read.fill(0); // Isolate the original recurrent-block objective.
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
    for(auto& l:base->levels)l.retrieval.read.fill(0); // Retained values are explicit stop-gradients.
    base->reconstruction_weight=0; // This oracle measures the byte objective only.
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
        if(p.kind==6)return l.p.length[p.index];
        if(p.kind==7)return l.address.read[p.index];
        if(p.kind==8)return l.address.query_content[p.index];
        if(p.kind==9)return l.address.query_state[p.index];
        return l.address.query_clock[p.index];
    };
    for(Probe p:std::array<Probe,16>{{{0,3},{0,65},{1,9},{1,133},{2,7},{3,126},{4,40},{5,12},{6,2},{6,23},
                                   {7,3},{7,140},{8,3},{8,1042},{9,3},{10,3}}}){
        const float eps=.003f;auto a=std::make_unique<Model>(*base),b=std::make_unique<Model>(*base);
        param(*a,p)+=eps;param(*b,p)-=eps;a->training=b->training=false;
        // The first slow output appears at byte four: 29 served predictions.
        float numerical=float((32./29)*(a->run(x.data(),y.data(),32)-b->run(x.data(),y.data(),32))/(2*eps));
        float actual=(param(*before,p)-param(*analytic,p))/.01f;
        require(std::abs(numerical-actual)<.0008f,"slow composer/projector VJP mismatch");
    }
}
// Independent double arithmetic for the new associative read/write derivative.
double address_objective(const Vec& c,const Vec& h,const Vec& stable,const AddressStream& s,
                         const AddressParameters& p,const AddressValue& dr,const AddressMatrix& dm){
    double current[K]{},previous[K]{},value[ADDRESS_V]{},key[ADDRESS_K]{},cn=1e-8,pn=1e-8;
    for(int i=0;i<K;++i){double learned=0;
        for(int j=0;j<K;++j){double sign=((i+j)%3==0?-1.:1.)*.3535533905932738;
            current[i]+=sign*stable[i*K+j];previous[i]+=sign*s.previous_stable[i*K+j];learned+=sign*c[i*K+j];}
        value[i]=std::tanh(learned);value[K+i]=current[i];cn+=current[i]*current[i];pn+=previous[i]*previous[i];}
    cn=1/std::sqrt(cn);pn=1/std::sqrt(pn);
    for(int i=0;i<K;++i){key[i]=current[i]*cn*.577350269;key[K+i]=previous[i]*pn*.577350269;}
    for(int i=0;i<TIME_D;++i)key[CONTENT_K+i]=s.clock.phase[i]*.577350269;
    double result=0,norm=1,residual[ADDRESS_V]{};
    for(int b=0;b<ADDRESS_V;++b)residual[b]=value[b];
    for(int a=0;a<ADDRESS_K;++a){norm+=key[a]*key[a];
        for(int b=0;b<ADDRESS_V;++b)residual[b]-=key[a]*retention()[a]*double(s.memory[a*ADDRESS_V+b]);}
    for(int a=0;a<ADDRESS_K;++a){double z=0;
        for(int i=0;i<D;++i)z+=p.query_content[a*D+i]*double(c[i])+p.query_state[a*D+i]*double(h[i]);
        if(a<CONTENT_K)z+=.5*current[a%K]*cn;
        else for(int i=0;i<TIME_D;++i)z+=p.query_clock[(a-CONTENT_K)*TIME_D+i]*s.clock.phase[i];
        double q=std::tanh(z);
        for(int b=0;b<ADDRESS_V;++b){int j=a*ADDRESS_V+b;
            result+=q*s.memory[j]*dr[b]+(retention()[a]*double(s.memory[j])+.5*key[a]*residual[b]/norm)*dm[j];}}
    return result;
}
void address_gradients(){
    Vec c{},h{},stable=stable_bytes()[73];AddressStream s;AddressParameters p{},g{};AddressValue dr{};AddressMatrix dm{};
    s.previous_stable=stable_bytes()[29];for(int k=0;k<19;++k)s.clock.advance();
    for(int i=0;i<D;++i){c[i]=.2f*std::sin(float(i));h[i]=.15f*std::cos(float(i));}
    for(int i=0;i<ADDRESS_K*ADDRESS_V;++i){s.memory[i]=.1f*std::cos(float(i));dm[i]=.07f*std::sin(float(i));}
    for(int i=0;i<ADDRESS_V;++i)dr[i]=.1f*std::cos(float(i));
    for(int i=0;i<ADDRESS_K*D;++i){p.query_content[i]=.02f*std::sin(float(i));p.query_state[i]=.03f*std::cos(float(i));}
    for(int i=0;i<TIME_D*TIME_D;++i)p.query_clock[i]=.03f*std::sin(float(i));
    auto next=s;AddressEvent e;address_forward(e,c,h,stable,next,p);auto carried=dm;Vec dc{},dh{};
    address_backward(e,c,h,p,dr,carried,dc,dh,g);
    constexpr float eps=.001f;
    for(int i=0;i<D;++i){auto plus=c,minus=c;plus[i]+=eps;minus[i]-=eps;
        double n=(address_objective(plus,h,stable,s,p,dr,dm)-address_objective(minus,h,stable,s,p,dr,dm))/(2*eps);
        require(std::abs(n-dc[i])<2e-5,"address content VJP mismatch");
        plus=h;minus=h;plus[i]+=eps;minus[i]-=eps;
        n=(address_objective(c,plus,stable,s,p,dr,dm)-address_objective(c,minus,stable,s,p,dr,dm))/(2*eps);
        require(std::abs(n-dh[i])<2e-5,"address previous-state VJP mismatch");}
    for(int i=0;i<ADDRESS_K*ADDRESS_V;++i){auto a=s,b=s;a.memory[i]+=eps;b.memory[i]-=eps;
        double n=(address_objective(c,h,stable,a,p,dr,dm)-address_objective(c,h,stable,b,p,dr,dm))/(2*eps);
        require(std::abs(n-carried[i])<2e-5,"address carried-matrix VJP mismatch");}
    for(int kind=0;kind<3;++kind)for(int index:{0,3,29,140}){
        auto a=p,b=p;auto field=[&](AddressParameters& z)->float&{return kind==0?z.query_content[index]:kind==1?z.query_state[index]:z.query_clock[index];};
        field(a)+=eps;field(b)-=eps;
        double n=(address_objective(c,h,stable,s,a,dr,dm)-address_objective(c,h,stable,s,b,dr,dm))/(2*eps);
        require(std::abs(n-field(g))<2e-5,"learned address query VJP mismatch");}
    AddressStream empty;AddressEvent first;address_forward(first,c,h,stable,empty,p);
    require(first.read==AddressValue{},"address reads before writing");
    Vec changed=c;changed.fill(.8f);auto other=s;AddressEvent altered;address_forward(altered,changed,h,stable,other,p);
    require(e.key==altered.key,"write coordinates independent of learned encoder");
    require(e.value!=altered.value,"addressed values include learned content");
}
void bounded_repeated_memory(){
    AddressStream stream;AddressParameters p{};AddressEvent e;Vec content{},before{};
    for(int i=0;i<D;++i)content[i]=.3f*std::sin(float(i));
    for(int i=0;i<TIME_D;++i)p.query_clock[i*TIME_D+i]=.25f;
    double max_memory=0,max_read=0;
    for(int t=0;t<10000;++t){
        address_forward(e,content,before,stable_bytes()[120],stream,p);
        double memory=0,read=0;for(float x:stream.memory)memory+=double(x)*x;
        for(float x:e.read)read+=double(x)*x;
        max_memory=std::max(max_memory,std::sqrt(memory));max_read=std::max(max_read,std::sqrt(read));
    }
    require(max_memory<10,"repeated content cannot accumulate unbounded coherent writes");
    require(max_read<10,"repeated-content read must remain bounded");
    require(stream.clock.ticks==10000,"residual write retains the native clock");
    // Removing a channel removes both its write and its read across the prefix.
    for(int mode:{2,3}){AddressStream s;
        for(int t=0;t<20;++t)address_forward(e,content,before,stable_bytes()[t],s,p,mode);
        for(int a=0;a<ADDRESS_K;++a)if((mode==2&&a>=CONTENT_K)||(mode==3&&a<CONTENT_K)){
            require(e.query[a]==0&&e.key[a]==0,"removed channel has no query or write key");
            for(int b=0;b<ADDRESS_V;++b)require(s.memory[a*ADDRESS_V+b]==0,"removed write channel remains empty");}
    }
    auto m=std::make_unique<Model>(2,200000);uint8_t key[]{1,2},value[]{4,5,6};
    auto handle=m->store.bind(key,2,value,3);auto copy=std::make_unique<Model>(*m);copy->inference(true);
    require(copy->store.resolve(handle)!=nullptr,"native cold context preserves persistent records");
    copy->reset(true);require(copy->store.resolve(handle)==nullptr,"explicit full reset clears persistent records");
    require(m->store.resolve(handle)!=nullptr,"inference reset leaves source records untouched");
}
void reconstruction_gradients(){
    Vec c{},target{};AddressParameters p{};for(int i=0;i<D;++i){c[i]=.2f*std::sin(float(i));target[i]=.3f*std::cos(float(i));}
    for(int i=0;i<D*D;++i)p.reconstruct[i]=.01f*std::sin(float(i));
    constexpr float eps=.002f;
    for(int role=0;role<GROUP;++role){AddressParameters g{};Vec dc{};reconstruct(c,target,role,p,1,dc,g);
        auto loss=[&](const Vec& v,const AddressParameters& params){AddressParameters dummy{};Vec d{};return reconstruct(v,target,role,params,0,d,dummy);};
        for(int i=0;i<D;++i){auto a=c,b=c;a[i]+=eps;b[i]-=eps;
            require(std::abs((loss(a,p)-loss(b,p))/(2*eps)-dc[i])<4e-6,"reconstruction content VJP mismatch");}
        for(int i:{0,17,140,4095}){auto a=p,b=p;a.reconstruct[i]+=eps;b.reconstruct[i]-=eps;
            require(std::abs((loss(c,a)-loss(c,b))/(2*eps)-g.reconstruct[i])<4e-6,"reconstruction head VJP mismatch");}
        auto learned=p;float start=loss(c,learned);
        for(int n=0;n<100;++n){AddressParameters grad{};Vec d{};reconstruct(c,target,role,learned,1,d,grad);
            step(learned.reconstruct,grad.reconstruct,1);step(learned.reconstruct_bias,grad.reconstruct_bias,1);}
        require(loss(c,learned)<start*.1f,"constituent decoder learns its target");}
}
void clock_contracts(){
    Clock clock;for(int t=0;t<1000000;++t)clock.advance();require(clock.ticks==1000000,"persistent native clock count");
    for(int i=0;i<ROTORS;++i){double theta=1000000*2*std::acos(-1.)/(4*std::pow(64.,double(i)/(ROTORS-1)));
        require(std::abs(clock.phase[2*i]-std::cos(theta)/std::sqrt(double(ROTORS)))<1e-9,"fixed oscillator phase");
        require(std::abs(clock.phase[2*i+1]-std::sin(theta)/std::sqrt(double(ROTORS)))<1e-9,"fixed oscillator quadrature");}
    std::array<Vec,GROUP> child{stable_bytes()[1],stable_bytes()[2],stable_bytes()[3],stable_bytes()[4]};
    auto original=stable_group(child);std::swap(child[0],child[1]);require(original!=stable_group(child),"stable address preserves constituent order");
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
    auto minimum=std::make_unique<Model>(4,205503);minimum->run(x.data(),y.data(),512);
    for(const auto& l:minimum->levels)require(l.experts.size()==2,"minimum budget reserves every expert bank");
    auto bytes=save(*minimum);auto resumed=std::make_unique<Model>(load(bytes.data(),bytes.size()));
    require(save(*resumed)==bytes,"minimum budget checkpoint topology");
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
// An independent double-precision attention oracle. Retained keys and payloads
// are detached observed constants; query parameters and the present state learn.
double recall_objective(const RecallStream& s,const RecallParameters& p,const Vec& input,int byte,const RecallValue& dv){
    double q[RECALL_K]{},scores[RECALL_SLOTS]{},max=-1e30,total=0,result=0;
    for(int j=0;j<RECALL_K;++j){double z=p.query_byte[byte*RECALL_K+j];
        for(int i=0;i<D;++i)z+=double(p.query[j*D+i])*input[i];q[j]=std::tanh(z);}
    for(int k=0;k<s.count;++k){scores[k]=-.5*double(s.clock-s.units[k].born);
        for(int j=0;j<RECALL_K;++j)scores[k]+=4*q[j]*s.units[k].key[j];max=std::max(max,scores[k]);}
    for(int k=0;k<s.count;++k){scores[k]=std::exp(scores[k]-max);total+=scores[k];}
    for(int k=0;k<s.count;++k)for(int j=0;j<RECALL_V;++j)result+=scores[k]/total*s.units[k].value[j]*dv[j];
    return result;
}
void recall_gradients(){
    RecallStream s;RecallParameters p{};Vec input{};RecallValue dv{};
    for(int i=0;i<6;++i)s.admit(stable_bytes()[31*i],stable_bytes()[31*i]);
    for(int j=0;j<D;++j)input[j]=.2f*std::sin(float(j));
    for(int j=0;j<RECALL_V;++j)dv[j]=.2f*std::cos(float(j));
    for(int j=0;j<RECALL_K*D;++j)p.query[j]=.03f*std::sin(float(j));
    for(int j=0;j<RECALL_K;++j)p.query_byte[63*RECALL_K+j]=.1f*std::cos(float(j));
    RecallRead e;recall_forward(s,p,input,63,e);RecallGradient g{};Vec di{};
    recall_backward(s,p,e,dv,g,di);constexpr float eps=.002f;
    for(int i=0;i<D;++i){auto a=input,b=input;a[i]+=eps;b[i]-=eps;
        double n=(recall_objective(s,p,a,63,dv)-recall_objective(s,p,b,63,dv))/(2*eps);
        require(std::abs(n-di[i])<1e-5,"present query-state VJP mismatch");}
    for(int j=0;j<RECALL_K;++j){auto a=p,b=p;a.query_byte[63*RECALL_K+j]+=eps;b.query_byte[63*RECALL_K+j]-=eps;
        double n=(recall_objective(s,a,input,63,dv)-recall_objective(s,b,input,63,dv))/(2*eps);
        require(std::abs(n-g.query_byte[j])<1e-5,"observed-byte query VJP mismatch");}
    for(int j:{0,31,67,309,780,1023}){auto a=p,b=p;a.query[j]+=eps;b.query[j]-=eps;
        double n=(recall_objective(s,a,input,63,dv)-recall_objective(s,b,input,63,dv))/(2*eps);
        require(std::abs(n-g.query[j])<1e-5,"query projection VJP mismatch");}
    // Verify teach wires this derivative to the same shared tree loss used by
    // logits, using fixed retained records rather than replaying their writers.
    auto base=std::make_unique<Model>(2,200000);base->learning_rate=base->head_rate=base->route_rate=0;
    const std::string prefix="Different retained units: AB then many xxxxx then ?";
    base->run(reinterpret_cast<const uint8_t*>(prefix.data()),reinterpret_cast<const uint8_t*>(prefix.data()),prefix.size());
    auto analytic=std::make_unique<Model>(*base);analytic->learning_rate=.01f;analytic->teach(49);
    auto loss=[](Model& m){auto& l=m.levels[0];Vec input=l.read_now.input;int byte=l.read_now.byte;recall_forward(l.retained,l.retrieval,input,byte,l.read_now);
        float out[256];m.log_probs(out);return -out[49];};
    for(int j:{0,3,9,15}){auto a=std::make_unique<Model>(*base),b=std::make_unique<Model>(*base);
        int index=63*RECALL_K+j;a->levels[0].retrieval.query_byte[index]+=eps;b->levels[0].retrieval.query_byte[index]-=eps;
        float n=(loss(*a)-loss(*b))/(2*eps);
        float actual=(base->levels[0].retrieval.query_byte[index]-analytic->levels[0].retrieval.query_byte[index])/.01f;
        require(std::abs(n-actual)<.0008f,"answer loss did not teach the current query");}
    auto read_update=std::make_unique<Model>(*base);read_update->head_rate=.01f;read_update->teach(49);
    for(int j:{0,8,17,31}){auto a=std::make_unique<Model>(*base),b=std::make_unique<Model>(*base);
        a->levels[0].retrieval.read[j]+=eps;b->levels[0].retrieval.read[j]-=eps;
        float n=(loss(*a)-loss(*b))/(2*eps);
        const auto& value=base->levels[0].read_now.value;
        float rate=.01f/(1+dot(value.data(),value.data(),RECALL_V));
        float actual=(base->levels[0].retrieval.read[j]-read_update->levels[0].retrieval.read[j])/rate;
        require(std::abs(n-actual)<.0008f,"retained-value decoder VJP mismatch");}
}
void recall_contracts(){
    RecallStream s,reverse;
    s.admit(stable_bytes()[65],stable_bytes()[65]);s.admit(stable_bytes()[66],stable_bytes()[66]);
    reverse.admit(stable_bytes()[66],stable_bytes()[66]);reverse.admit(stable_bytes()[65],stable_bytes()[65]);
    require(s.units[0].stable!=reverse.units[0].stable,"retained identity preserves pair order");
    auto cue=s.units[0];
    for(int t=0;t<10000;++t)s.admit(stable_bytes()[120+t%3],stable_bytes()[120+t%3]);
    require(s.count==5&&s.units[0].stable==cue.stable&&s.units[0].value==cue.value,"repeated distractor vocabulary preserves an observed cue");

    auto clock=s.clock;s.admit(stable_bytes()[120],stable_bytes()[120]);s.admit(stable_bytes()[120],stable_bytes()[120]);
    clock=s.clock;for(int t=0;t<10000;++t)s.admit(stable_bytes()[120],stable_bytes()[120]);
    require(s.clock==clock,"identical pairs do not admit or age retained units");
    for(int t=0;t<40;++t)s.admit(stable_bytes()[t],stable_bytes()[t]);
    require(s.count==RECALL_SLOTS,"retained memory has a fixed slot bound");
    bool old=false;for(const auto& u:s.units){old|=u.stable==cue.stable;require(u.born<=s.clock,"occurrence timestamps remain causal");}
    require(!old,"capacity overflow evicts the least recently observed distinct pair");
    auto m=std::make_unique<Model>(4,250000);m->inference(true);
    for(int i=0;i<194;++i)m->observe(uint8_t(i<2?65+i:120));
    auto slow=m->levels[3].s.executions;auto units=m->levels[3].retained;
    for(auto& l:m->levels){l.retrieval.query.fill(0);l.retrieval.query_byte.fill(0);l.retrieval.query_byte[63*RECALL_K]=1;}
    m->observe(63);
    require(m->levels[3].s.executions==slow&&m->levels[3].retained.clock==units.clock,"off-boundary question does not execute a slow expert or admit a slow pair");
    for(auto& l:m->levels){require(l.read_now.byte==63,"every scale receives the present question");
        require(l.read_now.query[0]>0,"current question selects retained units before next completion");
        require(l.s.count==0,"frozen recall does not populate derivative rings");}
    auto state=save(*m);auto clone=std::make_unique<Model>(load(state.data(),state.size()));
    require(save(*clone)==state,"warm frozen recall checkpoint round trip");
    m->reset();for(auto& l:m->levels)require(l.retained.count==0&&!l.retained.have_previous,"cold context clears causal retained units");
    for(int mode=1;mode<=4;++mode){RecallParameters p{};RecallRead e;recall_forward(s,p,Vec{},63,e,mode);
        if(mode==1)require(e.value==RecallValue{},"no-read intervention removes delivery");
        if(mode==3)for(int j=RECALL_K;j<RECALL_V;++j)require(e.value[j]==0,"fixed-value intervention");
        if(mode==4)for(int j=0;j<RECALL_K;++j)require(e.value[j]==0,"learned-value intervention");}
}
int main(){try{gradients();address_gradients();bounded_repeated_memory();reconstruction_gradients();clock_contracts();block_gradient();slow_gradient();contracts();associations();recall_gradients();recall_contracts();std::cout<<"PASS "<<checks<<" native checks\n";return 0;}
catch(const std::exception& e){std::cerr<<"FAIL after "<<checks<<" checks: "<<e.what()<<"\n";return 1;}}
