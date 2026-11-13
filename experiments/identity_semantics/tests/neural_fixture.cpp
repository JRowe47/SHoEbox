#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"

// Deterministic executions through the production cache eviction, bottom
// publication, event publication, relay, collection and credit routines.
inline void fixture_execute(Model&m,int r,int expert){
  m.s.tick++;
  int index=(m.laces.cursor[r]+1)&(LACE_RING-1);
  neural_archive_slot(m,r,index);
  m.s.chosen[r]=expert;
  for(int j=0;j<D;j++){
    m.s.u[r][j]=.04f*(1+j%7);m.s.h[r][j]=.03f*(1+j%9);m.s.mem[r][j]=m.s.u[r][j];
  }
  if(r<2)m.prepare_read(r);
  publish_state(m,r,1.1f,(expert+1)%E);
  m.laces.cursor[r]=index;auto&v=m.laces.records[r][index];v=LaceRecord{};
  v.tick=m.s.tick;v.expert=expert;v.runner=(expert+1)%E;v.gate=1.1f;v.gate_derivative=.3f;
  v.input_mask=~uint64_t(0);memcpy(v.u,m.s.u[r],sizeof(v.u));memcpy(v.h,m.s.h[r],sizeof(v.h));
  for(int j=0;j<D;j++)v.residual[j]=.02f*(1+j%5);
  auto&input=m.composition.stream.input[r][index];input=InputCredit{};input.tick=v.tick;
  for(int j=0;j<D;j++)m.probe.cues[r][index][j]=.06f*(1+j%3);
  m.probe.ticks[r][index]=v.tick;
  ContentEvent event{};event.ref.region=r;event.ref.begin=v.tick-1;event.ref.end=v.tick;
  event.clock_begin=v.tick-1;event.clock_end=v.tick;event.count=event.local_arity=2;
  content_publish(m,r,event);
}
extern "C" {
void neural_fixture_setup(void*p,int source,int owner){
  auto&m=*static_cast<Model*>(p);reset_context(p);m.s.tick=100;
  fixture_execute(m,source,2);
  auto held=m.composition.stream.history[source][0];
  if(owner==0){ // held publication through a transparent relay
    auto relayed=content_compose(m,14,&held,1,1,false);content_passthrough(m,14,relayed,held);
  }else if(owner==1){ // pending completed frame with repeated source roles
    auto&event=m.composition.stream.pending[14][0];event=held;event.origin=ChildEvent{};
    event.context_count=2;
    for(int k=0;k<2;k++)event.context_edges[k]={source,-1,held.origin.tick,.5f,held.origin.index};
    m.composition.stream.length[14]=1;
  }else if(owner==2){ // outstanding credit, independent of any publication
    auto*v=current_lace(m,source);v->uses[0]=1;
    for(int j=0;j<D;j++)v->credit[0][j]=.1f;
  }else if(owner==3){ // pending local forecast, not a public content owner
    float cue[D]{};relay_schedule(m,source,cue);
  }else if(owner==4||owner==5){ // saved feedback or retained-input dependency
    fixture_execute(m,17,3);auto&input=*lace_input(m,17,*current_lace(m,17));
    if(owner==4){input.message.feedback_count=1;input.message.feedback[0]={source,m.starts[17],held.origin.tick,1.f,held.origin.index};}
    else {input.previous_tick=held.origin.tick;input.previous_index=held.origin.index;input.retention=.5f;}
    auto consumer=m.composition.stream.history[17][0];content_passthrough(m,14,consumer,consumer);
  }
  // The owner above is intentionally the sole long-term root.
  m.composition.stream.history[source][0]=ContentEvent{};
  m.composition.stream.history[source][1]=ContentEvent{};
}
void neural_fixture_advance(void*p,int source,int count){
  auto&m=*static_cast<Model*>(p);
  for(int k=0;k<count;k++){fixture_execute(m,source,4);if(m.s.tick%64==0)neural_collect(m);}
  neural_collect(m);
}
void neural_fixture_status(void*p,int source,int64_t tick,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto*v=find_lace(m,source,tick,1);
  out[0]=v!=nullptr;out[1]=m.neural_archive[source].count(tick);
  out[2]=v?uint64_t(v->expert):999;out[3]=v&&lace_cue(m,source,*v)!=nullptr;
  out[4]=v&&lace_input(m,source,*v)&&lace_input(m,source,*v)->tick==tick;
  out[5]=source>=2||temporal_record(m,source,tick)!=nullptr;
}
void neural_fixture_update(void*p,int source,int objective){
  auto&m=*static_cast<Model*>(p);float g[D];for(int j=0;j<D;j++)g[j]=.1f*(1+j%3);
  if(objective==0){ // language derivative through two roles, then a delayed flush
    fixture_execute(m,17,3);auto&v=*current_lace(m,17);v.nparents=2;
    for(int k=0;k<2;k++)v.parents[k]={source,-1,101,.5f,1};
    distribute_credit(m,17,v,g,0);flush_credit(m,source,.04f);
  }else if(objective==1){ // saved local issuer; this fixture has no child dependencies
    auto&t=m.composition.stream.teaching[source];t.tick=101;t.index=1;
    for(int j=0;j<D;j++)t.prediction[j]=.3f;
    relay_teach(m,source,g,6,.04f,true);
  }else if(objective==2){
    fixture_execute(m,17,3);auto&v=*current_lace(m,17);auto&input=*lace_input(m,17,v);
    input.message.feedback_count=1;input.message.feedback[0]={source,m.starts[17],101,1.f,1};
    input.message.alpha=.5f;
    for(int j=0;j<D;j++)input.message.feedback_derivative[j]=.8f;
    message_feedback_credit(m,17,v,g,g,0);flush_credit(m,source,.04f);
  }else flush_credit(m,source,.04f);
}
void neural_fixture_release(void*p){
  auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;
  memset(s.history,0,sizeof(s.history));memset(s.pending,0,sizeof(s.pending));memset(s.length,0,sizeof(s.length));
  memset(s.teaching,0,sizeof(s.teaching));memset(s.memory_tick,0,sizeof(s.memory_tick));
  for(auto&region:m.laces.records)for(auto&v:region){v.nparents=0;memset(v.uses,0,sizeof(v.uses));}
  for(auto&region:s.input)for(auto&input:region)input=InputCredit{};
  for(auto&archive:m.neural_archive)for(auto&entry:archive){memset(entry.second.lace.uses,0,sizeof(entry.second.lace.uses));memset(entry.second.input.uses,0,sizeof(entry.second.input.uses));}
  neural_collect(m);
}
void neural_fixture_depth(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);reset_context(p);m.s.tick=100;m.lcfg[0]=1;
  fixture_execute(m,2,2);fixture_execute(m,3,2);fixture_execute(m,4,2);
  current_lace(m,3)->nparents=1;current_lace(m,3)->parents[0]={2,-1,101,1,1};
  current_lace(m,4)->nparents=1;current_lace(m,4)->parents[0]={3,-1,102,1,1};
  auto held=m.composition.stream.history[4][0];content_passthrough(m,14,held,held);
  for(int r=2;r<=4;r++)for(int k=0;k<192;k++)fixture_execute(m,r,4);
  neural_collect(m);
  out[0]=find_lace(m,4,103,1)!=nullptr;out[1]=find_lace(m,3,102,1)!=nullptr;out[2]=find_lace(m,2,101,1)!=nullptr;
  // The out-of-budget grandparent must not be accessed by depth-1 propagation.
  auto*child=find_lace(m,3,102,1);child->uses[0]=1;child->credit[0][0]=.1f;flush_credit(m,3,.01f);
  out[3]=m.laces.metrics[3][3];
}
}

extern "C" int neural_fixture_revive(void*p,int source,int expert){return revive_expert(*static_cast<Model*>(p),source,expert,5);}
extern "C" {
void neural_fixture_semantic_input(void*p){
  auto&m=*static_cast<Model*>(p);auto&input=*lace_input(m,3,*current_lace(m,3));
  ContentEvent event{};event.ref.region=3;event.ref.id={19,37};
  for(int j=0;j<D;j++)event.structure[j]=.05f*(1+j%3);
  semantic_encode(m,3,event);constituent_encode(m,3,event,nullptr,0,false);
  input.semantic=event.semantic;input.symbol_scale=.5f;input.message.qa=1;input.message.alpha=.5f;
}
void neural_fixture_recycle_semantics(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto*v=find_lace(m,3,101,1);auto ref=lace_input(m,3,*v)->semantic;
  for(int i=0;i<SEM_RING+8;i++){
    ContentEvent event{};event.ref.region=7;event.ref.id={uint64_t(i+100),73};
    semantic_encode(m,7,event);constituent_encode(m,7,event,nullptr,0,false);
  }
  out[0]=constituent_record(m,ref)!=nullptr;out[1]=constituent_capacity(m);
}
}
