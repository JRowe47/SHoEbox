#include "native.hpp"
#include <iostream>
#include <memory>
using namespace shoe;
extern "C" int diag_validate_gradient(void*,double*);
int checks=0;
void check(bool value,const char* msg){++checks;if(!value)throw std::runtime_error(msg);}
int main(){
    Vec state{},mean{},gradient{};
    for(int i=0;i<D;++i){state[i]=std::sin(.7f*i)*.2f;mean[i]=std::cos(.4f*i)*.13f;gradient[i]=std::sin(.3f*i)*.7f;}
    for(int mode:{1,2}){
        auto trace=centered_readout(state,mean,mode);auto analytic=centered_vjp(trace,gradient,mode);
        for(int j=0;j<D;++j){auto plus=state,minus=state;float eps=.002f;plus[j]+=eps;minus[j]-=eps;
            auto p=centered_readout(plus,mean,mode),m=centered_readout(minus,mean,mode);
            float numeric=(dot(p.value.data(),gradient.data())-dot(m.value.data(),gradient.data()))/(2*eps);
            check(std::abs(numeric-analytic[j])<.001f,"centered readout VJP");}
        auto model=std::make_unique<Model>(4,1000000,7,false,false);
        model->routing_arm=2;model->balance_policy=2;model->semantic_laces=false;model->center_mode=mode;
        uint8_t x[1024],y[1024];for(int i=0;i<1024;++i){x[i]=(17*i+i/7)%251;y[i]=(11*i+i/9)%251;}
        model->run(x,y,509);auto checked=std::make_unique<Model>(*model);double out[8]{};
        check(diag_validate_gradient(checked.get(),out)==0,"inspect centered gradient");
        for(int i=0;i<4;++i)check(out[2*i]<2e-6,"centered native gradient decomposition");
        auto raw=save(*model);auto restored=std::make_unique<Model>(load(raw.data(),raw.size()));
        check(raw==save(*restored),"centered checkpoint roundtrip");
        check(model->run(x,y,1024)==restored->run(x,y,1024)&&save(*model)==save(*restored),"centered pending continuation");
        model->inference(true);auto before=model->levels[3].center;model->run(x,y,1024);
        check(before.mean==model->levels[3].center.mean&&before.seen==model->levels[3].center.seen,"frozen means learned from evaluation");
    }
    std::cout<<checks<<" centered readout checks passed\n";
}
