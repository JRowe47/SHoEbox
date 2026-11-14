#include <fstream>
#include <iostream>
#include <cassert>
#include "factor_fixture.cpp"
template<class T> std::vector<T> read_array(std::ifstream&f){
  int64_t n;f.read(reinterpret_cast<char*>(&n),8);assert(n>=0&&n<10000000);
  std::vector<T> v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));assert(f.good());return v;
}
int main(int argc,char**argv){
  assert(argc==2);std::ifstream f(argv[1],std::ios::binary);assert(f.good());
  auto parameters=read_array<float>(f),embedding=read_array<float>(f),feedback=read_array<float>(f),slots=read_array<float>(f);
  auto starts=read_array<int>(f),perm=read_array<int>(f);auto signs=read_array<float>(f);auto fperm=read_array<int>(f);auto fsign=read_array<float>(f);auto cadence=read_array<int>(f);
  void*p=create(parameters.data(),embedding.data(),feedback.data(),slots.data(),starts.data(),perm.data(),signs.data(),fperm.data(),fsign.data(),cadence.data(),1);
  auto&m=*static_cast<Model*>(p);m.lcfg[3]=.1f;
  float vjp[32*D];optimization_vjp(p,16,vjp);assert(!memcmp(vjp,vjp+16*D,16*D*sizeof(float)));
  for(int workers:{2,4,1,2}){
    assert(content_workers_configure(p,workers,nullptr));
    optimization_gradients(p,1,0);optimization_gradients(p,1,1);
  }
  assert(content_workers_configure(p,1,nullptr));
  for(int sparse:{0,1}){
    content_factor_reset_status(p);factor_fixture(p,1,3,sparse,0);
  }
  content_factor_reset_status(p);factor_cancellation(p);
  optimization_credit_order(p,1,0);optimization_credit_order(p,1,1);
  reset_context(p);destroy(p);
  std::cout<<"ASan/UBSan passed: ready VJPs, repeated clipped factors, masks, scopes, cancellation, recycled handles, parallel rows and delayed cycles\n";
}
