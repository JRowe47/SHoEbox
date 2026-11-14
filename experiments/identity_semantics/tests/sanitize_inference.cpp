#include <cassert>
#include <fstream>
#include <iostream>
#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t size;f.read(reinterpret_cast<char*>(&size),8);assert(size>=0&&size<10000000);
  std::vector<T> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);std::ifstream f(argv[1],std::ios::binary);
  auto params=read_array<float>(f),emb=read_array<float>(f),fb=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  for(int scenario=0;scenario<4;scenario++){
    auto weights=params;void*ptr=create(weights.data(),emb.data(),fb.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);
    auto&m=*static_cast<Model*>(ptr);
    std::vector<float> extra(6*D+4*E*D);float state_cfg[24]{1,1,.25f,.125f,.03125f,1,1,64,0,.05f,.5f,0,128,512,100000,4096,8192,.1f,4096,64,0,.5f,0,.01f};
    configure_state(ptr,extra.data(),state_cfg);
    if(scenario)assert(event_configure(ptr,scenario==3?2:1,1,4,2,.05f,.1f,.25f));
    if(scenario>=2){assert(bank_epoch_configure(ptr,16,1,0));assert(bank_pipeline_workers(ptr,2,-1,1));}
    if(scenario==3)assert(frame_configure(ptr,1));
    assert(prediction_configure(ptr,scenario>0,16,scenario?8:1,1));
    float lcfg[12]{3,4,16,.1,.25,.5,1,1,2,64,1,1};configure_laces(ptr,lcfg);
    std::vector<unsigned char>x(8193);for(size_t i=0;i<x.size();i++)x[i]="The river crosses the valley. "[i%29];
    float out=0;assert(std::isfinite(run(ptr,x.data(),x.data(),259,6,.04,.04/3,.002,0,&out)));
    assert(inference_enter(ptr));uint64_t info[16];inference_inspect(ptr,info);assert(!info[2]&&!info[3]&&!info[4]&&!info[6]);
    auto fixed=weights;assert(std::isnan(run(ptr,x.data(),x.data(),1,6,.04,.04/3,.002,0,&out)));
    assert(std::isfinite(run(ptr,x.data(),x.data(),8193,0,0,0,0,0,&out)));assert(weights==fixed);
    std::vector<unsigned char> saved(inference_state_size());assert(inference_export(ptr,saved.data()));
    assert(!inference_import(ptr,saved.data(),saved.size()-1));assert(inference_import(ptr,saved.data(),saved.size()));
    std::vector<unsigned char> again(saved.size());inference_export(ptr,again.data());assert(again==saved);
    assert(composition_finish(ptr,0,0,0,0));reset_context(ptr);assert(m.forward_only);
    assert(!inference_resume(ptr,0));assert(inference_resume(ptr,1));assert(!m.forward_only);
    assert(std::isfinite(run(ptr,x.data(),x.data(),129,6,.04,.04/3,.002,0,&out)));
    reset_context(ptr);destroy(ptr);
  }
  std::cout<<"forward-only reference/event/epochs/frames, 8193-step continuation, import/reset/resume: ASan/UBSan passed\n";
}
