#include "completion_fixture.cpp"
inline ContentEvent pattern_fixture_child(int source,int token,int64_t begin,int64_t end){
  ContentEvent child{};child.ref.region=source;child.ref.id={content_mix(token+1),content_mix(token+713)};
  child.ref.seq=begin+10000;child.ref.begin=child.clock_begin=begin;child.ref.end=child.clock_end=end;
  child.ref.recognized=1;child.origin=child.ref;
  for(int j=0;j<D;j++)child.code[j]=child.structure[j]=.2f*((content_mix(uint64_t(token)*D+j)&1)?1.f:-1.f);
  return child;
}
inline ContentEvent pattern_fixture_frame(Model&m,int context,int swap,int phase,int64_t begin,bool learn){
  ContentEvent children[COMP_MAX];int n=0,r=6;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
    int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]);
    int token=src==2?(age^swap?11:29)+phase*100:context+300;
    auto child=pattern_fixture_child(src,token,src==2?begin+(1-age)*2:begin-context-10,src==2?begin+(2-age)*2:begin-context+1);
    child.ref.role=k;children[n++]=child;
  }
  return binding_compose(m,r,children,n,learn,7);
}
extern "C" {
void pattern_fixture_frame(void*p,int context,int swap,int phase,int64_t begin,int learn,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto event=pattern_fixture_frame(m,context,swap,phase,begin,learn);
  content_publish(m,6,event);auto key=pattern_key(m,event);
  out[0]=key.a;out[1]=event.ref.id.a;out[2]=event.ref.id.b;out[3]=event.ref.recognized;
  out[4]=event.present_mask;out[5]=event.count;out[6]=event.clock_begin;out[7]=event.clock_end;
}
float pattern_fixture_transition(void*p,int learn,int disturb){
  auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;auto next=s.history[6][0];
  auto*saved=pattern_assignment(m,next);
  if(disturb&&saved)for(float&v:saved->prediction)v=-v;
  float result=pattern_transition(m,6,s.history[6][1],next,learn);
  if(disturb&&saved)for(float&v:saved->prediction)v=-v;
  return result;
}
void pattern_fixture_prepare_pair(void*p,int context,int gap,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;reset_context(p);
  // The learned dictionary survives a document boundary.
  auto source=[&](int phase,int64_t begin){
    auto side=pattern_fixture_child(3,context+300,begin-context-10,begin-context+1);content_publish(m,3,side);
    for(int age=1;age>=0;age--){
      auto child=pattern_fixture_child(2,(age?11:29)+phase*100,begin+(1-age)*2,begin+(2-age)*2);content_publish(m,2,child);
    }
  };
  ContentEvent event{},parts[COMP_MAX];int n=0;
  source(0,0);bool first=completion_prepare(m,6,false,event,parts,n);
  source(1,4+gap);bool second=completion_prepare(m,6,false,event,parts,n);
  out[0]=first;out[1]=second;out[2]=s.length[6];out[3]=n;
  s.draining=1;bool drained=completion_prepare(m,6,false,event,parts,n);s.draining=0;
  out[4]=drained;out[5]=n;out[6]=event.clock_begin;out[7]=event.clock_end;
}
void pattern_fixture_relay(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto child=m.composition.stream.history[6][0];
  auto relay=content_compose(m,7,&child,1,1,false);auto a=pattern_key(m,child),b=pattern_key(m,relay);
  out[0]=a.a;out[1]=b.a;out[2]=a.b;out[3]=b.b;
}
void pattern_fixture_freeze(void*p,int region){static_cast<Model*>(p)->frozen_region[region]=1;}
void pattern_fixture_eviction(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto first=pattern_fixture_frame(m,0,0,0,0,true);
  out[0]=pattern_key(m,first).a;
  for(int i=1;i<=PATTERN_CAP+5;i++)pattern_fixture_frame(m,i,0,i,4*i,true);
  out[1]=pattern_key(m,first).a;out[2]=pattern_find(m,6,out[0])!=nullptr;
  auto again=pattern_fixture_frame(m,0,0,0,0,true);out[3]=pattern_key(m,again).a;
}
void pattern_fixture_regions(void*p,int context,uint64_t*out){
  auto&m=*static_cast<Model*>(p);
  for(int r=3;r<=14;r++){
    ContentEvent children[COMP_MAX];int n=0,driver=int(m.slots[4*m.starts[r]]);
    for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
      int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]),repeats=0;
      for(int t=m.starts[r];t<m.starts[r+1];t++)repeats+=!m.slots[4*t+2]&&int(m.slots[4*t])==src;
      bool core=src==driver||repeats>1;int token=src*31+age+(core?0:context*100);
      auto child=pattern_fixture_child(src,token,core?(1-age)*2: -context-10,core?(2-age)*2:1-context);
      child.ref.role=k;children[n++]=child;
    }
    auto event=binding_compose(m,r,children,n,true,0);auto key=pattern_key(m,event);
    out[(r-3)*4]=key.a;out[(r-3)*4+1]=event.ref.recognized;
    out[(r-3)*4+2]=event.ref.id.a;out[(r-3)*4+3]=event.count;
  }
}
void pattern_fixture_content(void*p,int reverse,uint64_t*out){
  auto&m=*static_cast<Model*>(p);ContentEvent parts[2]{completion_fixture_leaf(m,3,21),completion_fixture_leaf(m,3,29)};
  if(reverse)for(float&v:parts[0].code)v=-v;
  auto event=content_compose(m,7,parts,2,2,true);auto key=pattern_key(m,event);
  out[0]=key.a;out[1]=event.ref.id.a;out[2]=event.ref.id.b;
}
}
