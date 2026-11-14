#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
extern "C" {
float prediction_projection_vjp_fixture(int rank){
  float x[D],g[D],f[D],back[D],y[D],p[D];
  for(int j=0;j<D;j++){x[j]=.3f*std::sin(j+1.f);g[j]=.1f*std::cos(j+2.f);}
  prediction_pool(x,f,rank);prediction_expand(f,y,rank);
  prediction_pool(g,p,rank);prediction_expand(p,back,rank);
  float error=0;
  for(int j=0;j<D;j++){
    float original=x[j],values[2];
    for(int side=0;side<2;side++){
      x[j]=original+(side?-.001f:.001f);prediction_pool(x,f,rank);prediction_expand(f,y,rank);
      values[side]=dot64(y,g);
    }
    x[j]=original;error=std::max(error,std::abs((values[0]-values[1])/.002f-back[j]));
  }
  return error;
}
float prediction_forecast_vjp_fixture(void*p){
  auto&m=*static_cast<Model*>(p);auto&z=m.prediction;int r=6,rank=z.rank;
  float x[D],err[D],analytical[D],pooled[D];
  for(int j=0;j<D;j++){x[j]=.1f*std::sin(j+1.f);err[j]=.2f*std::cos(j+2.f);}
  prediction_gradient(m,r,err,analytical);
  const float*w=z.forecast.data()+prediction_region(r)*D*rank;float worst=0;
  for(int j=0;j<D;j++){
    float original=x[j],loss[2]{};
    for(int side=0;side<2;side++){
      x[j]=original+(side?-.001f:.001f);prediction_pool(x,pooled,rank);
      for(int i=0;i<D;i++)for(int k=0;k<rank;k++)loss[side]+=err[i]*w[i*rank+k]*pooled[k];
    }
    x[j]=original;worst=std::max(worst,std::abs((loss[0]-loss[1])/.002f-analytical[j]));
  }
  return worst;
}
int prediction_teacher_fixture(void*p){
  auto&m=*static_cast<Model*>(p);auto&z=m.prediction;int r=6;
  auto&v=m.laces.records[r][0];v=LaceRecord{};v.tick=1;v.expert=0;m.s.tick=1;
  for(int j=0;j<D;j++)v.h[j]=.2f*std::sin(j+1.f);
  float q[D]{},target[D];for(int j=0;j<D;j++)target[j]=.3f*std::cos(j+1.f);
  relay_schedule(m,r,q);auto view=z.stream.pending[r];auto weights=z.forecast;
  m.s.tick=10000; // Idle time cannot replace the pending producer or target.
  if(z.stream.pending[r].tick!=1||m.composition.stream.teaching[r].tick!=1)return 0;
  m.local_neural_credit=false;m.lcfg[0]=0;
  relay_teach(m,r,target,1,.01f,true);
  if(m.composition.stream.teaching[r].tick||z.forecast_version[r]!=1||weights==z.forecast)return 0;
  if(memcmp(&view,&z.stream.pending[r],sizeof(view)))return 0;
  weights=z.forecast;relay_teach(m,r,target,1,.01f,true);
  return weights==z.forecast&&z.forecast_version[r]==1;
}
int prediction_slow_clock_fixture(void*p){
  auto&m=*static_cast<Model*>(p);auto&z=m.prediction;
  auto&v=m.laces.records[17][0];v=LaceRecord{};v.tick=1;v.expert=0;m.s.tick=1;
  m.composition.stream.byte_position=1;
  for(int j=0;j<D;j++)v.h[j]=.2f*std::sin(j+1.f);
  prediction_slow_publish(m);auto snapshot=z.stream.current[17];
  for(int t=2;t<=z.stride;t++){
    m.s.tick=t;m.composition.stream.byte_position=t;
    if(prediction_slow_ready(m))return 0;
    v.h[0]+=.1f;
    if(memcmp(&prediction_view(m,17,v),&snapshot,sizeof(snapshot)))return 0;
  }
  ++m.composition.stream.byte_position;return prediction_slow_ready(m);
}
}
extern "C" int prediction_content_order_fixture(void*p){
  auto&m=*static_cast<Model*>(p);int r=6;ContentEvent children[COMP_MAX];int n=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
    auto&v=children[n++];int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]);
    v.ref.region=src;v.ref.role=k;v.ref.seq=3-age;v.ref.id={uint64_t(100+src+age),uint64_t(200+src+age)};
    v.ref.begin=v.clock_begin=2-age;v.ref.end=v.clock_end=3-age;
    for(int j=0;j<D;j++)v.edit_values().code[j]=v.edit_values().structure[j]=.1f*std::sin(float(j+src*7+age*11));
  }
  auto original=binding_compose(m,r,children,n,false,7);
  for(int region=0;region<N;region++)for(int j=0;j<D;j++)m.s.mem[region][j]=.9f;
  for(int j=0;j<MK_DIM;j++)m.event.memory[r][0][j][j]=10;
  auto changed_context=binding_compose(m,r,children,n,false,7);
  if(memcmp(original.values().code,changed_context.values().code,D*sizeof(float)))return 0;
  // Swap the two primary payloads while leaving their source/role slots fixed.
  auto a=children[0].payload;children[0].payload=children[1].payload;children[1].payload=a;
  std::swap(children[0].ref.id,children[1].ref.id);
  auto reversed=binding_compose(m,r,children,n,false,7);
  if(!memcmp(original.values().code,reversed.values().code,D*sizeof(float)))return 0;
  if(original.ref.id.a==reversed.ref.id.a&&original.ref.id.b==reversed.ref.id.b)return 0;
  return original.present_mask==reversed.present_mask&&original.count==reversed.count;
}

extern "C" void prediction_zero_a1_fast(void*p){memset(static_cast<Model*>(p)->prediction.fast[2],0,D*sizeof(float));}
