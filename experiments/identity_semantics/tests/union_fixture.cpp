#include "optimization_fixture.cpp"
#include <cassert>
extern "C" void union_fixture(void*p,int policy,int mixed,int frozen){
  auto&m=*static_cast<Model*>(p);m.reverse_policy=policy;m.content_records.resize(8);
  m.lcfg[3]=.1f;m.lcfg[9]=16;m.frozen_region[3]=frozen;
  if(mixed)for(int r=2;r<N;r++){m.teaching_scale[r][0]=.3f+.1f*(r%4);m.teaching_scale[r][1]=.8f-.1f*(r%5);}
  for(int i=0;i<8;i++){
    auto&node=constituent_slot(m,i);node=ConstituentRecord{};node.ref={uint64_t(i+1),2+i%3,i};node.encoder=1;
    node.gain_derivative=.4f;
    if(i>0)node.edges[node.count++]={constituent_slot(m,i-1).ref,{},-100,.7f};
    if(i>2)node.edges[node.count++]={constituent_slot(m,i-3).ref,{},-101,-.2f};
    auto&v=m.content_records[i];v=ContentExpertRecord{};v.ref=node.ref;v.expert=2;v.runner=3;
    v.gate=.8f;v.gate_derivative=.2f;v.input_mask=0x8181818181818181ull;
    for(int j=0;j<D;j++){
      node.derivative[j]=.7f;node.payload_derivative[j]=.8f;node.payload[j]=.1f*std::sin(float(j));
      v.input[j]=.1f*std::cos(float(i*13+j));v.residual[j]=.2f*std::sin(float(i+j));
      v.route_derivative[j]=.01f*std::cos(float(i*3+j));
    }
    for(int k=0;k<8;k++)node.features[k]=.2f*std::cos(float(i+k));
  }
  {ContentCreditScope outer(m);
    for(int q=0;q<18;q++){
      float gradient[D];for(int j=0;j<D;j++)gradient[j]=.03f*std::sin(float(q*3+j));
      int i=7-q%3;float rate=q%5==0?-.001f:.002f;
      constituent_update(m,constituent_slot(m,i).ref,gradient,rate,2+q%3,mixed?q%2:0);
    }
  }
  assert(m.reverse_roots.empty()&&m.content_dirty.empty()&&!m.reverse_flushing);
}
extern "C" void union_clipping(void*p,int policy){
  auto&m=*static_cast<Model*>(p);m.reverse_policy=policy;m.content_records.resize(1);m.lcfg[3]=0;
  auto&node=constituent_slot(m,0);node=ConstituentRecord{};node.ref={1,2,0};node.encoder=1;
  for(int j=0;j<D;j++)node.derivative[j]=1;
  auto&v=m.content_records[0];v=ContentExpertRecord{};v.ref=node.ref;v.expert=2;v.runner=3;v.input[0]=1;v.gate=1;v.input_mask=~uint64_t(0);
  float old=m.bias[(2*E+2)*D],gradient[D]{};
  {ContentCreditScope scope(m);gradient[0]=4;constituent_update(m,node.ref,gradient,1,2);
    gradient[0]=-1;constituent_update(m,node.ref,gradient,1,2);}
  assert(std::abs(m.bias[(2*E+2)*D]-(old-(policy==1?.5f:1.5f)))<1e-6f);
}
