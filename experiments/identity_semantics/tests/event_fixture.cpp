#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
#include <cassert>
extern "C" {
float event_context_vjp_check(void*p,int heads){
  auto&m=*static_cast<Model*>(p);int r=7;m.event.heads=heads;m.s.tick=1;
  float u[D],g[D],base[MK_HEADS][MK_DIM][MK_DIM];
  for(int i=0;i<D;i++){u[i]=.2f*std::sin(float(i+1));g[i]=.3f*std::cos(float(2*i+3));}
  for(int h=0;h<MK_HEADS;h++)for(int a=0;a<MK_DIM;a++)for(int b=0;b<MK_DIM;b++)base[h][a][b]=.13f*std::sin(float(1+h+3*a+5*b));
  auto forward=[&](const float*x,ContextEligibility&save,float*out){
    memcpy(m.event.memory[r],base,sizeof(base));context_forward(m,r,x,out,save);
    for(int j=0;j<D;j++)out[j]=std::tanh(x[j]+out[j]);
  };
  ContextEligibility saved{};float out[D];forward(u,saved,out);m.event.context[r][0]=saved;
  auto&v=m.laces.records[r][0];v=LaceRecord{};v.tick=1;memcpy(v.u,u,sizeof(u));memcpy(v.h,out,sizeof(out));
  float analytic[D];assert(context_input_gradient(m,r,v,g,analytic));float worst=0;
  for(int i=0;i<D;i++){
    float x[D],plus[D],minus[D];memcpy(x,u,sizeof(u));x[i]+=.001f;forward(x,saved,plus);
    x[i]-=.002f;forward(x,saved,minus);float numeric=0;
    for(int j=0;j<D;j++)numeric+=g[j]*(plus[j]-minus[j])/.002f;
    worst=std::max(worst,std::abs(analytic[i]-numeric));
  }
  return worst;
}
int event_frontier_check(void*p,int count){
  auto&m=*static_cast<Model*>(p);assert(m.event.local);m.event.training=false;
  ContentEvent child{};child.ref.region=3;semantic_encode(m,3,child);
  constituent_encode(m,3,child,nullptr,0,false);auto old=child.semantic;
  for(int i=0;i<count;i++){
    ContentEvent next{};next.ref.region=3;semantic_encode(m,3,next);
    constituent_encode(m,3,next,&child,1,false);child=next;m.composition.stream.history[3][0]=child;
  }
  assert(constituent_capacity(m)==SEM_RING);assert(!constituent_record(m,old));
  float g[D]{};g[0]=.2f;g[13]=-.3f;g[27]=.1f;
  constituent_update(m,child.semantic,g,.02f,3);event_finish_step(m);
  assert(m.event.metrics[3][3]>0&&m.event.metrics[3][10]>0);
  assert(m.event.metrics[3][9]==0&&m.composition.stream.completion_metrics[3][1]==0);
  return int(m.event.metrics[3][2]);
}
int event_content_independence(void*p){
  auto&m=*static_cast<Model*>(p);ContentEvent a{},b{};a.ref.region=b.ref.region=3;
  semantic_encode(m,3,a);semantic_encode(m,3,b);constituent_encode(m,3,a,nullptr,0,false);
  for(auto&h:m.event.memory[3])for(auto&row:h)for(float&v:row)v=9.f;
  constituent_encode(m,3,b,nullptr,0,false);
  return !memcmp(a.values().code,b.values().code,D*sizeof(float))&&a.ref.id.a==b.ref.id.a&&a.ref.id.b==b.ref.id.b;
}
}
