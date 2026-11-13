#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "../runtime/hierarchy_lab/throughput_probe/composition.cpp"
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t size;f.read(reinterpret_cast<char*>(&size),8);assert(size>=0&&size<10000000);
  std::vector<T> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);std::ifstream f(argv[1],std::ios::binary);assert(f.good());
  auto parameters=read_array<float>(f),embedding=read_array<float>(f),feedback=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  void*p=create(parameters.data(),embedding.data(),feedback.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);
  float lcfg[12]{3,4,16,.1,.25,.5,0,1,2,64,4,1};configure_laces(p,lcfg);
  auto&m=*static_cast<Model*>(p);m.cfg[12]=256;m.cfg[13]=4;m.scfg[10]=.5;
  std::vector<unsigned char>x(10000),y(10000);for(int i=0;i<10000;i++){x[i]=i%19<12?"abcdefghijkl"[i%12]:i%256;y[i]=x[i];}
  float out=0;double loss=run(p,x.data(),y.data(),x.size(),6,.012,.004,.002,0,&out);assert(std::isfinite(loss));
  assert(composition_finish(p,6,.012,.004,.002));
  std::vector<unsigned char>saved(composition_state_size());composition_export(p,saved.data());
  assert(composition_import(p,saved.data(),saved.size(),0));
  reset_context(p);loss=run(p,x.data(),y.data(),x.size(),0,0,0,0,0,&out);assert(std::isfinite(loss));
  destroy(p);std::cout<<"ASan/UBSan passed; train/eval/drain/reset/export/import; loss "<<loss<<"\n";
}
