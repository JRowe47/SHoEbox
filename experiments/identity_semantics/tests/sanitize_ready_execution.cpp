#include <cassert>
#include <fstream>
#include <iostream>
#include "ready_execution_fixture.cpp"
#include "retention_oracle.inc"
#include "semantic_retention_oracle.inc"
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t size;f.read(reinterpret_cast<char*>(&size),8);assert(size>=0&&size<10000000);
  std::vector<T> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);assert(!inline_diagnostics_enabled());assert(execution_accumulator_fixture());
  std::ifstream f(argv[1],std::ios::binary);
  auto params=read_array<float>(f),emb=read_array<float>(f),fb=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);
  auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  for(int scenario=0;scenario<7;scenario++){
    auto weights=params;void*ptr=create(weights.data(),emb.data(),fb.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);
    auto&m=*static_cast<Model*>(ptr);
    if(scenario==0)assert(execution_adjoint_fixture(ptr));
    else if(scenario==1)assert(execution_scheduler_fixture(ptr));
    else if(scenario==2)assert(execution_reverse_fixture(ptr));
    else {
      if(scenario>3){assert(event_configure(ptr,scenario==6?2:1,1,4,2,.05f,.1f,.25f));}
      if(scenario>=5){assert(bank_pipeline_configure(ptr,1));assert(bank_pipeline_workers(ptr,scenario==6?2:1,-1,1));}
      float lcfg[12]{3,4,16,.1,.25,.5,1,1,2,64,1,1};configure_laces(ptr,lcfg);
      std::vector<unsigned char>x(1025),y(1025);const char*text="The river crosses the valley. ";
      for(int i=0;i<1025;i++){x[i]=text[i%30];y[i]=text[(i+1)%30];}
      float out=0;assert(std::isfinite(run(ptr,x.data(),y.data(),1024,6,.04,.04/3,.002,0,&out)));
      bank_pipeline_finish(ptr);reset_context(ptr);
      assert(std::isfinite(run(ptr,x.data(),y.data(),129,6,.04,.04/3,.002,0,&out)));
      uint64_t inventory[32];execution_storage_status(ptr,inventory);assert(inventory[0]>0);
    }
    destroy(ptr);
  }
  std::cout<<"ready execution, accumulators, packed derivatives, plan invalidation and reference/event/worker lifecycle ASan/UBSan passed\n";
}
