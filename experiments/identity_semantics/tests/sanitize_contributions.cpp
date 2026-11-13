#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "../runtime/hierarchy_lab/throughput_probe/contributions.cpp"
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t size;f.read(reinterpret_cast<char*>(&size),8);assert(size>=0&&size<10000000);
  std::vector<T> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);std::ifstream f(argv[1],std::ios::binary);assert(f.good());
  auto parameters=read_array<float>(f),embedding=read_array<float>(f),feedback=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  auto make=[&](){return create(parameters.data(),embedding.data(),feedback.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);};
  void*source=make();std::vector<unsigned char>x(96),y(96);for(int i=0;i<96;i++)x[i]=y[i]=i;
  float out=0;assert(std::isfinite(run(source,x.data(),y.data(),x.size(),0,0,0,0,0,&out)));
  for(int kind=CHECK_NONE;kind<=CHECK_EXPERT;kind++)for(int r=0;r<N;r++){
    if(kind==CHECK_SEMANTIC&&r<2)continue;
    void*p=make();evaluation_copy_memory(p,source);freeze_writes(p,1);
    assert(contribution_configure(p,kind,kind==CHECK_NONE?-1:r));
    uint64_t before[5],after[5];evaluation_memory_info(p,before);
    assert(std::isnan(run(p,x.data(),y.data(),x.size(),6,.01,.003,.002,0,&out)));
    assert(std::isfinite(run(p,x.data(),y.data(),x.size(),0,0,0,0,0,&out)));
    assert(!composition_finish(p,6,.01,.003,.002));
    assert(composition_finish(p,0,0,0,0));
    evaluation_memory_info(p,after);assert(before[2]==after[2]);
    uint64_t counts[N][5],hashes[N];contribution_metrics(p,&counts[0][0],hashes);
    assert(counts[0][1]==x.size());assert(!contribution_configure(p,1,0));
    destroy(p);
  }
  destroy(source);std::cout<<"ASan/UBSan passed: every intervention/region, warm frozen memory, evaluation guards\n";
}
