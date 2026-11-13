#include "../runtime/hierarchy_lab/throughput_probe/completion.cpp"
inline ContentEvent paired_fixture_event(Model&m,int a,int token){
  ContentEvent e{};e.ref.region=15+a;e.ref.seq=m.composition.stream.history[15+a][0].ref.seq+1;
  e.ref.tick=m.s.tick;e.ref.begin=e.clock_begin=m.s.tick-1;e.ref.end=e.clock_end=m.s.tick;
  e.ref.id={uint64_t(token+100),uint64_t(a+1)};e.ref.recognized=1;
  for(int j=0;j<D;j++)e.code[j]=.125f*((content_mix(uint64_t(token+17)*D+j)&1)?1.f:-1.f);
  return e;
}
extern "C" {
void paired_fixture_step(void*p,int mask,int content,int context,int commit_now){
  auto&m=*static_cast<Model*>(p);m.s.tick++;
  for(int a=0;a<2;a++)if(mask&(1<<a)){
    auto e=paired_fixture_event(m,a,a?context:content);
    paired_read(m,a,e.code,e);m.composition.stream.history[15+a][0]=e;
    paired_stage(m,a,e.code,e);
  }
  if(commit_now)paired_commit(m);
}
void paired_fixture_commit(void*p){paired_commit(*static_cast<Model*>(p));}
void paired_fixture_returns(void*p,uint64_t*ids,float*values){
  auto&m=*static_cast<Model*>(p);ids[0]=m.s.return_birth[0];ids[1]=m.s.return_birth[1];
  ids[2]=m.paired.stream.read_episode;ids[3]=uint64_t(m.paired.stream.read_slot+1);
  memcpy(values,m.s.returned,sizeof(m.s.returned));
}
size_t paired_fixture_entries(void*p,uint64_t*meta,float*values){
  auto&m=*static_cast<Model*>(p);size_t n=0;
  for(size_t i=0;i<m.paired.slots.size();i++){
    const auto&e=m.paired.slots[i];if(!e.id)continue;auto*out=meta+10*n;
    out[0]=i;out[1]=e.id;out[2]=e.born;out[3]=e.updated;
    for(int a=0;a<2;a++){
      out[4+3*a]=e.members[a].stream;out[5+3*a]=e.members[a].occurrence.seq;out[6+3*a]=e.members[a].occurrence.tick;
      memcpy(values+(n*2+a)*D,e.members[a].value,D*sizeof(float));
    }
    n++;
  }
  return n;
}
void paired_fixture_credit(void*p,int r,int slot,uint64_t id,const float*returned,float*out){
  auto&m=*static_cast<Model*>(p);LaceRecord record{};record.tick=m.s.tick;record.expert=0;record.gate=1;
  record.recall_slot=slot;record.recall_birth=id;float g[D];
  for(int j=0;j<D;j++){record.h[j]=.1f;record.u[j]=.2f;g[j]=.3f;record.returned[j]=returned[j];}
  float before[D*D];memcpy(before,m.P+r*D*D,sizeof(before));
  float old=slot>=0?m.paired.slots[slot].utility:0;
  expert_update(m,r,g,.04f,&record,0,false);
  out[0]=0;for(int j=0;j<D*D;j++)out[0]+=std::abs(before[j]-m.P[r*D*D+j]);
  out[1]=slot>=0?m.paired.slots[slot].utility-old:0;
}
size_t paired_fixture_priority_offset(){return sizeof(PairedHeader)+sizeof(uint64_t)+offsetof(PairedEpisode,priority);}
}
