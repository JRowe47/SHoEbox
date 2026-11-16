#include <iostream>
int main(){
 using namespace shoe;int checks=0;
 auto require=[&](bool ok,const char* msg){++checks;if(!ok)throw std::runtime_error(msg);};
 // Check every affine coordinate for local MSE and logistic presence gradients.
 std::mt19937 rng(91);std::uniform_real_distribution<float> draw(-.2f,.2f);
 for(int trial=0;trial<16;++trial){Affine p{};Vec x{};for(auto& z:p)z=draw(rng);for(auto& z:x)z=draw(rng);float target=.7f;
  for(int type=0;type<2;++type){float z=affine(p,x);float e=type?(sigmoid(z)-1):(std::tanh(z)-target)*(1-std::tanh(z)*std::tanh(z));
   auto objective=[&](const Affine& q){double y=double(q[D]);for(int j=0;j<D;++j)y+=double(q[j])*x[j];return type?std::log1p(std::exp(-y)):.5*std::pow(std::tanh(y)-target,2);};
   for(int j=0;j<=D;++j){Affine plus=p,minus=p;plus[j]+=.001f;minus[j]-=.001f;double numerical=(objective(plus)-objective(minus))/(plus[j]-minus[j]);
    double analytic=e*(j==D?1:x[j]);require(std::abs(numerical-analytic)<2e-5,"local derivative");}
  }
  auto q=p;reconstruct(q,x,target,.1f);float e=(std::tanh(affine(p,x))-target)*(1-std::pow(std::tanh(affine(p,x)),2));
  for(int j=0;j<=D;++j)require(std::abs((p[j]-q[j])-.1f/(1+energy(x))*e*(j==D?1:x[j]))<1e-7,"local step");
 }
 QueryAdapter a(500000,7,true,1,false);require(a.model->minimal_core,"minimal");require(QPARAMS==9900,"head accounting");
 for(int t=0;t<300;++t){a.model->observe(uint8_t(t));
  auto before=save(*a.model);a.prepare();require(before==save(*a.model),"readout mutates core");
  auto clone=query_load(query_save(a).data(),query_save(a).size());
  require(query_save(a)==query_save(*clone),"checkpoint roundtrip");
  QueryScope scope(a);float lp[256];a.model->log_probs(lp);double mass=0;for(float p:lp)mass+=std::exp(p);require(std::abs(mass-1)<2e-6,"distribution");
  Labels local{{t%256,-1,-1,-1,-1}};a.local(local);a.model->teach(uint8_t(t+1));
  require(save(*a.model)==before,"frozen local update changes core");
 }
 // Every output and bias coefficient has the native logistic derivative.
 a.prepare();for(int n=0;n<255;++n){float error=.23f;a.errors={};a.errors[n]=error;auto old=a.head;auto h=a.cache.h;std::array<RecallValue,4> unused{};a.backward(unused);
  float lr=a.rate/(1+dot(h.data(),h.data(),VRANK));
  for(int r=0;r<VRANK;++r)require(std::abs(a.head.output[n*VRANK+r]-(old.output[n*VRANK+r]-lr*error*h[r]))<1e-7,"native output derivative");
  require(std::abs(a.head.bias[n]-(old.bias[n]-a.rate*.05f*error))<1e-7,"native bias derivative");
 }
 a.training=false;a.model->inference(false);auto frozen=query_save(a);a.local({{1,1,1,1,1}});require(frozen==query_save(a),"frozen local learning");
 std::cout<<checks<<" constituent checks passed\n";
}
