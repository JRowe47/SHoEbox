#define SHOEBOX_LEARNED_COMPLETION 1
#include "messages_fixture.cpp"
inline ContentEvent completion_fixture_leaf(Model&m,int r,int id,int64_t begin=0,int64_t end=2){
  ContentEvent e{};e.ref.id={content_mix(id+1),content_mix(id+313)};
  e.ref.region=r;e.ref.seq=id+1;e.ref.begin=begin;e.ref.end=end;
  e.clock_begin=begin;e.clock_end=end;
  e.ref.recognized=1;e.count=e.local_arity=2;e.kind=1;
  for(int j=0;j<D;j++)e.structure[j]=.2f*std::sin(float(id*7+j));
  if(r>=2){semantic_encode(m,r,e);constituent_encode(m,r,e,nullptr,0,false);}
  else memcpy(e.code,e.structure,sizeof(e.code));
  e.origin=e.ref;return e;
}
inline ContentEvent completion_fixture_bind(Model&m,int r,int src,const ContentEvent&child){
  ContentEvent parts[COMP_MAX];int n=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
    int source=int(m.slots[4*k]);parts[n]=source==src?child:completion_fixture_leaf(m,source,source+80);
    parts[n].ref.role=k;n++;
  }
  return binding_compose(m,r,parts,n,false,0);
}
extern "C" {
void completion_fixture_grow(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;
  auto child=completion_fixture_leaf(m,3,99);SemanticRef first=child.semantic;
  s.history[3][0]=child;
  for(int i=0;i<SEM_RING+31;i++){
    ContentEvent next{};next.ref.region=3;next.ref.seq=i+2;next.ref.begin=0;next.ref.end=i+2;
    semantic_encode(m,3,next);constituent_encode(m,3,next,&child,1,false);
    child=next;s.history[3][0]=child;
  }
  out[0]=constituent_capacity(m);out[1]=constituent_record(m,first)!=nullptr;
  float gradient[D];for(float&g:gradient)g=.1f;
  auto before=s.completion_metrics[3][0];constituent_update(m,child.semantic,gradient,.01f,3);
  out[2]=s.completion_metrics[3][0]-before;out[3]=s.completion_metrics[3][4];
}
void completion_fixture_update_grown(void*p){
  auto&m=*static_cast<Model*>(p);float gradient[D];for(float&g:gradient)g=.1f;
  constituent_update(m,m.composition.stream.history[3][0].semantic,gradient,.01f,3);
}
void completion_fixture_chain(void*p,int coordinate,float pulse,int relay,float*out,uint64_t*ids){
  auto&m=*static_cast<Model*>(p);auto child=completion_fixture_leaf(m,3,21);
  if(relay){auto copy=content_compose(m,2,&child,1,1,false);child=copy;}
  if(coordinate>=0)child.code[coordinate]+=pulse;
  auto parent=completion_fixture_bind(m,7,3,child);
  auto upper=completion_fixture_bind(m,10,7,parent);
  m.composition.stream.history[10][0]=upper;
  memcpy(out,upper.code,D*sizeof(float));memcpy(out+D,upper.structure,D*sizeof(float));
  ids[0]=upper.ref.id.a;ids[1]=upper.ref.id.b;
}
void completion_fixture_update(void*p,const float*g,float rate){
  auto&m=*static_cast<Model*>(p);constituent_update(m,m.composition.stream.history[10][0].semantic,g,rate,10);
}
void completion_fixture_gate(void*p,int r,float delta){static_cast<Model*>(p)->composition.payload_gates[r-2]+=delta;}
void completion_fixture_source(void*p,int r,int id,int64_t begin,int64_t end){
  auto&m=*static_cast<Model*>(p);auto e=completion_fixture_leaf(m,r,id,begin,end);
  content_publish(m,r,e);
}
void completion_fixture_seed_frame(void*p,int r){
  auto&m=*static_cast<Model*>(p);ContentEvent children[COMP_MAX];int n=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
    children[n]=m.composition.stream.history[int(m.slots[4*k])][int(m.slots[4*k+1])];children[n++].ref.role=k;
  }
  for(int i=0;i<4;i++)binding_compose(m,r,children,n,true,0);
}
void completion_fixture_prepare(void*p,int r,int draining,int learn,int64_t*out){
  auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;ContentEvent event{},parts[COMP_MAX];int n=0;
  s.draining=draining;bool ready=content_prepare(m,r,learn,event,parts,n);s.draining=0;
  if(ready)content_publish(m,r,event);
  out[0]=ready;out[1]=s.length[r];out[2]=n;out[3]=event.ref.recognized;out[4]=s.seen[r];
  auto&frame=ready?parts[0]:s.pending[r][std::max(0,s.length[r]-1)];
  out[5]=frame.count;out[6]=frame.clock_begin;out[7]=frame.clock_end;
  for(int i=0;i<frame.count;i++){out[8+3*i]=frame.children[i].region;out[9+3*i]=frame.children[i].seq;out[10+3*i]=frame.children[i].role;}
}
void completion_fixture_erase_patterns(void*p){auto&m=*static_cast<Model*>(p);memset(m.composition.patterns,0,sizeof(m.composition.patterns));}
}

extern "C" {
void completion_fixture_clock_case(void*p,int overlapping,double*out){
 auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;
 auto source=[&](int r,int id,int64_t begin,int64_t end,int64_t clock_begin){
  ContentEvent e{};e.ref.id={content_mix(id+13),content_mix(id+71)};e.ref.region=r;e.ref.begin=begin;e.ref.end=end;
  e.clock_begin=clock_begin;e.clock_end=end;e.count=e.local_arity=2;e.kind=1;e.ref.recognized=1;
  for(int j=0;j<D;j++)e.code[j]=e.structure[j]=.1f*std::sin(float(id+j));content_publish(m,r,e);
 };
 auto seed=[&](){ContentEvent parts[COMP_MAX];int n=0;
  for(int k=m.starts[14];k<m.starts[15];k++)if(!m.slots[4*k+2]){parts[n]=s.history[int(m.slots[4*k])][int(m.slots[4*k+1])];parts[n++].ref.role=k;}
  ContentEvent e;for(int t=0;t<4;t++)e=binding_compose(m,14,parts,n,true,0);return e;
 };
 source(11,1,0,8,0);source(11,2,8,16,8);source(12,3,0,16,0);
 source(10,4,0,8,0);source(10,5,8,16,8);seed();ContentEvent e{},parts[8];int n;
 completion_prepare(m,14,false,e,parts,n);
 auto previous=s.pending[14][0];
 int shift=overlapping==2?8:overlapping==3?-8:0;
 source(10,6,overlapping?0:16,24+shift,16+shift);source(10,7,overlapping?8:24,32+shift,24+shift);auto next=seed();
 auto pair=content_pair(previous.ref.id,next.ref.id,0x70616972);
 for(int t=0;t<10;t++){content_count(m.composition.totals[14],previous.ref.id,true,s.metrics[14][6]);content_count(m.composition.pairs[14],pair,true,s.metrics[14][6]);}
 out[0]=content_transition(m,14,previous.ref.id,next.ref.id,false);out[1]=previous.clock_end;
 out[2]=s.history[10][1].ref.begin;out[3]=s.history[10][1].clock_begin;
 out[4]=completion_prepare(m,14,false,e,parts,n);out[5]=n;out[6]=s.length[14];
}
}
