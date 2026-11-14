#include "../runtime/hierarchy_lab/throughput_probe/contributions.cpp"
extern "C" {
void check_fixture_zero_expert(void*p,int r){
  auto&m=*static_cast<Model*>(p);
  memset(m.W+r*E*D*D,0,E*D*D*sizeof(float));memset(m.bias+r*E*D,0,E*D*sizeof(float));
}
void check_fixture_memory(void*p,int r,float value){
  auto&m=*static_cast<Model*>(p);for(float&x:m.s.mem[r])x=value;
}
void check_fixture_vectors(void*p,int r,float*out){
  auto&m=*static_cast<Model*>(p);auto&e=m.composition.stream.history[r][0];
  memcpy(out,m.s.mem[r],D*sizeof(float));memcpy(out+D,m.s.h[r],D*sizeof(float));
  memcpy(out+2*D,e.values().structure,D*sizeof(float));memcpy(out+3*D,e.context_data().context,D*sizeof(float));
  memcpy(out+4*D,m.learned.memory[r<2?r:0],D*sizeof(float));
}
void check_fixture_relay(void*p,int source,int receiver){
  auto&m=*static_cast<Model*>(p);auto child=m.composition.stream.history[source][0];
  auto event=content_compose(m,receiver,&child,1,1,false);
  content_passthrough(m,receiver,event,child);
}
void check_fixture_symbol(void*p,int r){
  auto&m=*static_cast<Model*>(p);for(auto&row:m.composition.symbols[r-2])for(float&v:row)v=.25f;
}
void check_fixture_atom(void*p,int r,int byte){
  auto&m=*static_cast<Model*>(p);m.s.tick=1;m.composition.stream.byte_position=1;
  for(float&v:m.s.h[r])v=.75f;content_atom(m,r,byte);
}
}
