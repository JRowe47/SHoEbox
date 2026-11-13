#include "neural_fixture.cpp"
extern "C" {
void local_fixture_setup(void*p,int depth,int repeated,int relay){
  auto&m=*static_cast<Model*>(p);reset_context(p);m.s.tick=99;
  m.lcfg[0]=depth;m.lcfg[1]=1;m.lcfg[5]=.5f;m.lcfg[6]=0;m.lcfg[9]=D;
  m.utility.config[4]=0;
  fixture_execute(m,5,4);fixture_execute(m,3,2);fixture_execute(m,4,3);m.s.tick++;
  current_lace(m,3)->nparents=1;current_lace(m,3)->parents[0]={5,-1,100,.25f,1};
  fixture_execute(m,7,1);
  auto&transparent=*current_lace(m,7);transparent.expert=-1;transparent.nparents=1;
  transparent.parents[0]={3,-1,101,1,1};
  fixture_execute(m,6,5);auto&previous=*current_lace(m,6);
  previous.nparents=1;previous.parents[0]={5,-1,100,.7f,1};
  fixture_execute(m,6,2);auto&v=*current_lace(m,6);
  for(int k=0;k<repeated;k++)v.parents[v.nparents++]={relay?7:3,-1,relay?104:101,.3f/repeated,1};
  int slot=m.starts[6];v.parents[v.nparents++]={4,slot,102,.2f,1};
  auto&input=*lace_input(m,6,v);input.previous_tick=105;input.previous_index=1;input.retention=.5f;
  input.message.feedback_count=1;input.message.feedback[0]={4,slot,102,.6f,1};input.message.alpha=.5f;
  for(int j=0;j<D;j++){
    input.message.feedback_derivative[j]=.8f;
    input.message.route_derivative[j]=.04f*(j%2?1:-1);
  }
  // Self-consistent saved expert activation for independent finite differences.
  auto*w=m.W+(6*E+2)*D*D;auto*b=m.bias+(6*E+2)*D;
  for(int j=0;j<D;j++){
    v.residual[j]=dot64(w+j*D,v.u);
    v.h[j]=std::tanh(v.u[j]+b[j]+v.gate*v.residual[j]);
  }
  ContentEvent event{};event.ref.region=6;event.ref.id={19,37};
  semantic_encode(m,6,event);constituent_encode(m,6,event,nullptr,0,false);
  input.semantic=event.semantic;input.symbol_scale=.5f;input.message.qa=1;
  float q[D]{};relay_schedule(m,6,q);
}
void local_fixture_teach(void*p,float rate,int relay){
  auto&m=*static_cast<Model*>(p);float q[D];for(int j=0;j<D;j++)q[j]=.04f*(j%7-3);
  relay_teach(m,6,q,6,rate,relay);
}
void local_fixture_snapshot(void*p,int r,int64_t tick,float*out){
  auto&m=*static_cast<Model*>(p);auto*v=find_lace(m,r,tick);
  if(!v){for(int j=0;j<3*D+1;j++)out[j]=NAN;return;}
  memcpy(out,v->u,D*sizeof(float));memcpy(out+D,v->h,D*sizeof(float));
  memcpy(out+2*D,v->residual,D*sizeof(float));out[3*D]=v->gate;
}
void local_fixture_pending(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);out[0]=out[1]=0;
  for(int r=0;r<N;r++)for(auto*v:lace_records(m,r)){
    auto*input=lace_input(m,r,*v);
    for(int d=0;d<LACE_DEPTH;d++){out[0]+=v->uses[d];if(input)out[1]+=input->uses[d];}
  }
}
void local_fixture_freeze(void*p,int r){static_cast<Model*>(p)->frozen_region[r]=1;}
void local_fixture_task_credit(void*p){
  auto&m=*static_cast<Model*>(p);auto*v=find_lace(m,3,101);v->uses[0]=2;neural_pending(m,3,v->tick,0);
  for(int j=0;j<D;j++)v->credit[0][j]=.01f*(j+1);
}
void local_fixture_task_status(void*p,float*out){
  auto&m=*static_cast<Model*>(p);auto*v=find_lace(m,3,101);
  if(!v){for(int j=0;j<LACE_DEPTH*(D+1);j++)out[j]=NAN;return;}
  memcpy(out,v->credit,sizeof(v->credit));
  for(int d=0;d<LACE_DEPTH;d++)out[D*LACE_DEPTH+d]=v->uses[d];
}
void local_fixture_live_teach(void*p,int r){
  auto&m=*static_cast<Model*>(p);LaceRecord*v=nullptr;
  auto records=lace_records(m,r);
  for(auto it=records.rbegin();it!=records.rend();++it)if((*it)->expert>=0&&(*it)->nparents){v=*it;break;}
  if(!v)return;
  auto&t=m.composition.stream.teaching[r];t=PendingTeaching{};t.tick=v->tick;t.index=lace_cache_index(m,r,*v);
  for(int i=0;i<D;i++){
    t.prediction[i]=m.pbias[r*D+i];
    for(int j=0;j<D;j++)t.prediction[i]+=m.P[(r*D+i)*D+j]*v->h[j];
  }
  float q[D];for(int j=0;j<D;j++)q[j]=.04f*(j%7-3);
  relay_teach(m,r,q,6,.02f,false);
}
}
