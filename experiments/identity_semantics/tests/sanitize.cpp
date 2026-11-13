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
#ifdef SHOEBOX_LEARNED_COMPLETION
  // Exercise the enabled reference paths, including learned bottom state and
  // the auxiliary objectives, rather than only the older comparison policy.
  lcfg[6]=1;lcfg[10]=1;configure_laces(p,lcfg);
  float cfg[24]{3,1,256,1,1,1,1,4,.6f,1,1,1,2048,4,.5f,.98f,1,1,1,.85f,.02f,0,.4f,.005f};
  configure(p,cfg);
  float scfg[24]{1,4,.5f,.125f,.03125f,4,1,64,1,.05f,.5f,1,512,2048,100000,8192,32768,.05f,16384,64,.1f,.5f,0,.01f};
  std::vector<float>extra(2*3*D+4*E*D,0);
  for(int r=0;r<2;r++)for(int j=0;j<D;j++){
    extra[(r*3)*D+j]=std::atanh(.1f/.5f);
    float write=r?.03125f:.125f;extra[(r*3+2)*D+j]=std::log(write/(1-write));
  }
  memcpy(extra.data()+2*3*D,m.keys,2*E*D*sizeof(float));
  configure_state(p,extra.data(),scfg);
  float utility[8]{1,1,1,1,1,.001f,0,0};utility_configure(p,utility);
#endif
#ifdef SHOEBOX_LEARNED_COMPLETION
  constexpr int length=4096; // Plus the forced >16K live-DAG lifecycle below.
#else
  constexpr int length=10000;
#endif
  std::vector<unsigned char>x(length),y(length);for(int i=0;i<length;i++){x[i]=i%19<12?"abcdefghijkl"[i%12]:i%256;y[i]=x[i];}
  float out=0;double loss=run(p,x.data(),y.data(),x.size(),6,.012,.004,.002,0,&out);assert(std::isfinite(loss));
  assert(composition_finish(p,6,.012,.004,.002));
  std::vector<unsigned char>saved(composition_state_size(p));composition_export(p,saved.data());
  assert(composition_import(p,saved.data(),saved.size(),0));
#ifdef SHOEBOX_LEARNED_COMPLETION
  reset_context(p);
  ContentEvent child{};child.ref.region=3;semantic_encode(m,3,child);
  constituent_encode(m,3,child,nullptr,0,false);m.composition.stream.history[3][0]=child;
  for(int i=0;i<SEM_RING+31;i++){
    ContentEvent next{};next.ref.region=3;semantic_encode(m,3,next);
    constituent_encode(m,3,next,&child,1,false);child=next;
    m.composition.stream.history[3][0]=child;
  }
  assert(constituent_capacity(m)>SEM_RING);
  float gradient[D];for(float&g:gradient)g=.1f;
  constituent_update(m,child.semantic,gradient,.01f,3);
  saved.resize(composition_state_size(p));composition_export(p,saved.data());
  reset_context(p);assert(composition_import(p,saved.data(),saved.size(),0));
  assert(constituent_record(m,m.composition.stream.history[3][0].semantic));
  constituent_update(m,m.composition.stream.history[3][0].semantic,gradient,.01f,3);
#endif
  reset_context(p);loss=run(p,x.data(),y.data(),x.size(),0,0,0,0,0,&out);assert(std::isfinite(loss));
  destroy(p);std::cout<<"ASan/UBSan passed; train/eval/drain/reset/export/import (including semantic pool growth in completion); loss "<<loss<<"\n";
}
