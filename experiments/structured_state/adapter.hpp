#pragma once
#include "native.hpp"
#include <memory>

namespace shoe {
// Research-only supervised state probe. No fields are added to production Model.
constexpr int TASK_FEATURES=2*D,TASK_REGISTERS=4,TASK_CLASSES=8;
using TaskFeatures=std::array<float,TASK_FEATURES>;
struct TaskHead {
    std::array<float,TASK_REGISTERS*TASK_CLASSES*TASK_FEATURES> weight{};
    std::array<float,TASK_REGISTERS*TASK_CLASSES> bias{};
};
struct TaskTape {
    std::array<std::array<std::array<Vec,BLOCK>,2>,2> gradient{};
    std::array<std::array<uint64_t,2>,2> count{};
};
struct TaskAdapter {
    std::unique_ptr<Model> model;
    TaskHead head{};
    TaskTape tape{};
    int mode=0; // 0 final-only; 1 auxiliary head-only; 2 auxiliary state credit
    float auxiliary=.25f,head_rate=.2f;
    std::array<uint64_t,2> labels{};
    TaskAdapter(int parameters,uint32_t seed,int arm,uint64_t budget):
        model(std::make_unique<Model>(4,parameters,seed,false,false)),mode(arm){
        if(arm<0||arm>2||!budget)throw std::invalid_argument("invalid task policy");
        auto& m=*model;m.semantic_laces=false;m.routing_arm=2;m.balance_policy=2;
        auto& r=m.route_learning;r.configured=r.enabled=true;
        r.lower=.1f;r.upper=1;r.start=1;r.target=.1f;r.amplitude=0;r.ramp_targets=budget;
        std::mt19937 rng(seed^0x53544154u);std::normal_distribution<float> n(0,.01f);
        for(auto& w:head.weight)w=n(rng);
    }
    TaskAdapter(const TaskAdapter& other):model(std::make_unique<Model>(*other.model)),
        head(other.head),tape(other.tape),mode(other.mode),auxiliary(other.auxiliary),
        head_rate(other.head_rate),labels(other.labels){}
    TaskFeatures features()const{
        TaskFeatures x{};for(int l=0;l<2;++l)std::copy(model->levels[l+2].s.state.begin(),model->levels[l+2].s.state.end(),x.begin()+l*D);return x;
    }
    std::array<float,32> probabilities(const TaskFeatures& x)const{
        std::array<float,32> p{};
        for(int r=0;r<4;++r){float maximum=-1e30f,total=0;
            for(int k=0;k<8;++k){int a=r*8+k;p[a]=head.bias[a]+dot(head.weight.data()+a*TASK_FEATURES,x.data(),TASK_FEATURES);maximum=std::max(maximum,p[a]);}
            for(int k=0;k<8;++k)total+=(p[r*8+k]=std::exp(p[r*8+k]-maximum));
            for(int k=0;k<8;++k)p[r*8+k]/=total;
        }return p;
    }
    double supervise(const int* target,bool intermediate){
        if(!model->training)throw std::logic_error("cannot teach frozen task adapter");
        int n=0;for(int r=0;r<4;++r){if(target[r]<-1||target[r]>=8)throw std::invalid_argument("class must be -1 or 0..7");n+=target[r]>=0;}
        if(!n)throw std::invalid_argument("at least one label required");
        if(!model->bytes||model->bytes%64)throw std::logic_error("task labels require a completed 64-byte boundary");
        for(int l=2;l<4;++l)if(!model->levels[l].s.count)throw std::logic_error("task event already retired");
        if(intermediate&&mode==0)return 0;
        const auto x=features();auto p=probabilities(x);TaskFeatures dx{};double loss=0;
        const float weight=intermediate?auxiliary:1.f;
        const float rate=head_rate*model->route_learning.shared()*weight/(1+dot(x.data(),x.data(),TASK_FEATURES));
        for(int r=0;r<4;++r)if(target[r]>=0){
            loss-=std::log(std::max(p[r*8+target[r]],1e-30f))/n;
            for(int k=0;k<8;++k){int a=r*8+k;float error=(p[a]-(k==target[r]))/n;
                for(int j=0;j<TASK_FEATURES;++j){dx[j]+=error*head.weight[a*TASK_FEATURES+j];head.weight[a*TASK_FEATURES+j]-=rate*error*x[j];}
                head.bias[a]-=head_rate*model->route_learning.shared()*weight*.05f*error;
            }
        }
        int channel=intermediate?1:0;labels[channel]+=n;
        if(!intermediate||mode==2)for(int l=0;l<2;++l){int t=model->levels[l+2].s.count-1;
            for(int j=0;j<D;++j)tape.gradient[channel][l][t][j]+=dx[l*D+j];++tape.count[channel][l];}
        return loss;
    }
};
inline thread_local TaskAdapter* active_task=nullptr;
struct TaskScope {
    TaskAdapter* before;
    explicit TaskScope(TaskAdapter& a):before(active_task){active_task=&a;}
    ~TaskScope(){active_task=before;}
};
inline void task_credit(Model* m,int scale,int slot,Vec& dh){
    if(!active_task||active_task->model.get()!=m||scale<2)return;
    auto& a=*active_task;int l=scale-2;
    for(int ch=0;ch<2;++ch)if(a.tape.count[ch][l]){
        float w=(ch?a.auxiliary:1.f)/a.tape.count[ch][l];
        for(int j=0;j<D;++j)dh[j]+=w*a.tape.gradient[ch][l][slot][j];
    }
}
inline void task_retired(Model* m,int scale){
    if(!active_task||active_task->model.get()!=m||scale<2)return;
    for(int ch=0;ch<2;++ch){active_task->tape.gradient[ch][scale-2]={};active_task->tape.count[ch][scale-2]=0;}
}
inline std::string task_save(const TaskAdapter& a){
    Writer w;uint64_t magic=0x31304b5341544853ULL;uint32_t version=1;w.pod(magic);w.pod(version);
    auto core=save(*a.model);uint64_t size=core.size();w.pod(size);w.out+=core;
    w.pod(a.mode);w.pod(a.auxiliary);w.pod(a.head_rate);w.pod(a.head);w.pod(a.tape);w.pod(a.labels);return w.out;
}
inline std::unique_ptr<TaskAdapter> task_load(const char* p,size_t size){
    Reader r{p,size};uint64_t magic,n;uint32_t version;r.pod(magic);r.pod(version);r.pod(n);
    if(magic!=0x31304b5341544853ULL||version!=1||n>r.n)throw std::runtime_error("invalid task checkpoint");
    auto a=std::make_unique<TaskAdapter>(1000000,7,0,1);a->model=std::make_unique<Model>(load(r.p,n));r.p+=n;r.n-=n;
    r.pod(a->mode);r.pod(a->auxiliary);r.pod(a->head_rate);r.pod(a->head);r.pod(a->tape);r.pod(a->labels);
    if(r.n||a->mode<0||a->mode>2||a->model->levels.size()!=4)throw std::runtime_error("invalid task policy");
    for(float x:{a->auxiliary,a->head_rate})if(!std::isfinite(x)||x<0)throw std::runtime_error("invalid task rate");
    for(float x:a->head.weight)if(!std::isfinite(x))throw std::runtime_error("invalid head weight");
    for(float x:a->head.bias)if(!std::isfinite(x))throw std::runtime_error("invalid head bias");
    for(int ch=0;ch<2;++ch)for(int l=0;l<2;++l){
        if(a->tape.count[ch][l]>1000000)throw std::runtime_error("invalid task credit count");
        for(int t=0;t<BLOCK;++t)for(float x:a->tape.gradient[ch][l][t])if(!std::isfinite(x)||(t>=a->model->levels[l+2].s.count&&x!=0))throw std::runtime_error("invalid task credit");
    }return a;
}
} // namespace shoe
