#include <iostream>
using namespace shoe;
int checks=0;
void check(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
void frozen_rates(TaskAdapter& a){auto& m=*a.model;m.learning_rate=m.slow_rate=m.query_rate=m.head_rate=m.route_rate=0;m.route_learning.enabled=false;}
void prefix(TaskAdapter& a){TaskScope scope(a);std::array<uint8_t,129> data{};for(int i=0;i<129;++i)data[i]=uint8_t(32+(i*13)%80);a.model->run(data.data(),data.data()+1,128);}
double objective(TaskAdapter& a){auto p=a.probabilities(a.features());double loss=0;for(int r=0;r<4;++r)loss-=std::log(double(p[r*8+r+1]))/4;return loss;}
int main(){
    try{
        auto base=std::make_unique<TaskAdapter>(1000000,7,2,3000000);frozen_rates(*base);prefix(*base);
        auto detached=std::make_unique<TaskAdapter>(*base),trained=std::make_unique<TaskAdapter>(*base);
        detached->mode=1;int labels[4]{1,2,3,4};detached->supervise(labels,true);trained->supervise(labels,true);
        check(std::memcmp(&detached->head,&trained->head,sizeof(TaskHead))==0,"head-only weights differ");
        for(auto* a:{detached.get(),trained.get()}){a->model->slow_rate=.08f;TaskScope scope(*a);a->model->finish();}
        for(int l:{2,3}){
            auto& before=detached->model->levels[l].p.recurrent;auto& after=trained->model->levels[l].p.recurrent;
            std::vector<int> ids(D*D);std::iota(ids.begin(),ids.end(),0);
            std::partial_sort(ids.begin(),ids.begin()+8,ids.end(),[&](int a,int b){return std::abs(before[a]-after[a])>std::abs(before[b]-after[b]);});
            for(int k=0;k<8;++k){int j=ids[k];double analytic=(before[j]-after[j])/.08/.25;double values[2]{};
                for(int sign=0;sign<2;++sign){auto probe=std::make_unique<TaskAdapter>(1000000,7,2,3000000);frozen_rates(*probe);probe->model->levels[l].p.recurrent[j]+=(sign?1:-1)*.01f;prefix(*probe);values[sign]=objective(*probe);}
                double numeric=(values[1]-values[0])/.02;
                if(std::abs(analytic-numeric)>2e-5+std::abs(numeric)*.08){std::cerr<<l<<" "<<j<<" "<<analytic<<" "<<numeric<<"\n";throw std::runtime_error("task recurrent finite difference");}++checks;
            }
        }
        auto continuation=task_load(task_save(*base).data(),task_save(*base).size());
        check(task_save(*base)==task_save(*continuation),"checkpoint mismatch");
        for(auto* a:{base.get(),continuation.get()}){TaskScope scope(*a);a->supervise(labels,true);a->model->finish();a->model->reset();prefix(*a);}
        check(task_save(*base)==task_save(*continuation),"continuation mismatch");
        std::cout<<checks<<" structured-state native checks passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
