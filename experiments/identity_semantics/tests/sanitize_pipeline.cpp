// Production-layout lifecycle; no observation build and no inline harness.
#include <cassert>
#include <fstream>
#include <iostream>
#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t size;f.read(reinterpret_cast<char*>(&size),8);assert(size>=0&&size<10000000);
  std::vector<T> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);assert(!inline_diagnostics_enabled());std::ifstream f(argv[1],std::ios::binary);
  auto params=read_array<float>(f),emb=read_array<float>(f),fb=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  for(int workers:{1,2}){
    auto weights=params;void*ptr=create(weights.data(),emb.data(),fb.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);
    auto&m=*static_cast<Model*>(ptr);assert(event_configure(ptr,1,1,4,2,.05f,.1f,.25f));
    assert(bank_pipeline_configure(ptr,1));assert(bank_pipeline_workers(ptr,workers,-1,1));
    float lcfg[12]{3,4,16,.1,.25,.5,1,1,2,64,1,1};configure_laces(ptr,lcfg);
    std::vector<unsigned char>x(1025),y(1025);for(int i=0;i<1025;i++){x[i]="The river crosses the valley. "[i%30];y[i]=x[i];}
    float out=0;assert(std::isfinite(run(ptr,x.data(),y.data(),1024,6,.04,.04/3,.002,0,&out)));
    uint64_t info[24],rows[N*E*5];bank_pipeline_inspect(ptr,info,rows);assert(info[11]>=info[12]);
    auto*first=m.bank_pipeline.pending.get();auto*second=m.bank_pipeline.building.get();assert(first!=second);
    for(int n=0;n<33;n++){
      run(ptr,x.data(),y.data(),1,6,.04,.04/3,.002,0,&out);bank_pipeline_inspect(ptr,info,rows);
      assert((m.bank_pipeline.pending.get()==first)||(m.bank_pipeline.pending.get()==second));
    }
    std::vector<unsigned char>saved(bank_pipeline_state_size(ptr));bank_pipeline_export(ptr,saved.data());
    assert(bank_pipeline_import(ptr,saved.data(),saved.size()));assert(!bank_pipeline_import(ptr,saved.data(),saved.size()-1));
    auto before=saved;std::vector<unsigned char>reexport(bank_pipeline_state_size(ptr));bank_pipeline_export(ptr,reexport.data());assert(before==reexport);
    // Fresh importer jobs contain no transient factor arrays. Resume then recycle.
    run(ptr,x.data(),y.data(),129,6,.04,.04/3,.002,0,&out);bank_pipeline_finish(ptr);assert(!m.bank_pipeline.pending);
    run(ptr,x.data(),y.data(),17,6,.04,.04/3,.002,0,&out);destroy(ptr);
  }
  std::cout<<"production compact pipeline ASan/UBSan passed\n";
}
