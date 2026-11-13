struct PairedHeader {
  uint64_t version=1,serial=0,capacity=0,count=0;
  PairedStream stream{};
};
inline bool paired_valid_observation(const PairedObservation&o,int side,uint64_t epoch,int64_t tick){
  if(!o.stream)return o.occurrence.seq==0;
  if(o.stream>epoch||o.occurrence.region!=15+side||o.occurrence.seq<=0||
     o.occurrence.tick<=0||o.occurrence.tick>tick||o.occurrence.begin<0||
     o.occurrence.end<o.occurrence.begin||o.clock_begin<0||o.clock_end<o.clock_begin)return false;
  for(float v:o.value)if(!std::isfinite(v))return false;
  float norm=0;for(float v:o.key){if(!std::isfinite(v))return false;norm+=v*v;}
  return norm<=1.00001f;
}
extern "C" {
int paired_memory_configure(void*p,int enabled){
  auto&m=*static_cast<Model*>(p);
  if((enabled!=0&&enabled!=1)||m.composition.stream.semantic_serial||m.composition.stream.byte_position||
     paired_occupied(m)||m.s.used||m.paired.stream.latest[0].stream||m.paired.stream.latest[1].stream)return 0;
  for(int r=0;r<N;r++)if(m.s.valid[r])return 0; // Legacy neural returns must also be retired.
  m.paired.enabled=enabled;return 1;
}
size_t paired_memory_state_size(void*p){
  return sizeof(PairedHeader)+paired_occupied(*static_cast<Model*>(p))*(sizeof(uint64_t)+sizeof(PairedEpisode));
}
void paired_memory_export(void*p,void*out){
  auto&m=*static_cast<Model*>(p);auto&z=m.paired;
  PairedHeader header{};header.serial=z.serial;header.capacity=z.slots.size();header.count=paired_occupied(m);header.stream=z.stream;
  auto*cursor=static_cast<unsigned char*>(out);
  auto write=[&](const void*value,size_t size){memcpy(cursor,value,size);cursor+=size;};
  write(&header,sizeof(header));
  for(uint64_t i=0;i<z.slots.size();i++)if(z.slots[i].id){write(&i,sizeof(i));write(&z.slots[i],sizeof(PairedEpisode));}
}
int paired_memory_import(void*p,const void*data,size_t size){
  auto&m=*static_cast<Model*>(p);PairedHeader header{};
  if(size<sizeof(header))return 0;memcpy(&header,data,sizeof(header));
  const auto&s=header.stream;
  if(header.version!=1||header.serial>uint64_t(INT64_MAX)||header.capacity>PCAP||header.count>header.capacity||
     (header.capacity&&header.capacity!=size_t(m.cfg[12])*size_t(m.cfg[13]))||
     size!=sizeof(header)+header.count*(sizeof(uint64_t)+sizeof(PairedEpisode))||!s.epoch||
     s.read_tick<0||s.read_tick>m.s.tick||s.read_episode>header.serial||s.read_slot<-1||
     (s.read_slot>=0&&uint64_t(s.read_slot)>=header.capacity)||bool(s.read_episode)!=(s.read_slot>=0)||
     !std::isfinite(s.gate)||s.gate<0||!std::isfinite(s.similarity))return 0;
  for(int a=0;a<2;a++){
    if(!paired_valid_observation(s.latest[a],a,s.epoch,m.s.tick)||!paired_valid_observation(s.query[a],a,s.epoch,m.s.tick)||
       (s.latest[a].stream&&s.latest[a].stream!=s.epoch)||(s.query[a].stream&&s.query[a].stream!=s.epoch)||
       s.considered_stream[a]>s.epoch||s.considered_seq[a]<0||bool(s.considered_stream[a])!=bool(s.considered_seq[a])||
       (s.considered_stream[a]&&(s.considered_stream[a]!=s.latest[a].stream||s.considered_seq[a]>s.latest[a].occurrence.seq)))return 0;
  }
  auto next=std::make_unique<PairedMemory>();next->enabled=m.paired.enabled;
  next->serial=header.serial;next->stream=s;next->slots.resize(header.capacity);
  auto*cursor=static_cast<const unsigned char*>(data)+sizeof(header);uint64_t previous=0;
  std::map<uint64_t,bool>ids;
  for(uint64_t i=0;i<header.count;i++){
    uint64_t slot;PairedEpisode e{};memcpy(&slot,cursor,sizeof(slot));cursor+=sizeof(slot);
    memcpy(&e,cursor,sizeof(e));cursor+=sizeof(e);
    if(slot>=header.capacity||(i&&slot<=previous)||!e.id||e.id>header.serial||!ids.emplace(e.id,true).second||
       e.born<=0||e.born>e.updated||e.updated>m.s.tick||!std::isfinite(e.priority)||e.priority<0||e.priority>1||
       !std::isfinite(e.utility)||std::abs(e.utility)>1||!e.members[0].stream||e.members[0].stream!=e.members[1].stream)return 0;
    for(int a=0;a<2;a++)if(!paired_valid_observation(e.members[a],a,s.epoch,e.born))return 0;
    previous=slot;next->slots[slot]=e;
  }
  if(!m.paired.enabled&&(header.count||s.latest[0].stream||s.latest[1].stream||s.read_episode))return 0;
  // A cached return may legally refer to an episode replaced later in the same
  // step. Its immutable ID and detached values remain valid for delayed credit.
  m.paired=std::move(*next);return 1;
}
void paired_memory_status(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto&z=m.paired;memcpy(out,z.metrics,sizeof(z.metrics));
  out[16]=paired_occupied(m);out[17]=z.serial;out[18]=z.stream.epoch;out[19]=z.stream.read_episode;
  out[20]=z.stream.read_tick;
  for(int a=0;a<2;a++){
    const auto&o=z.stream.latest[a];out[21+3*a]=o.stream;out[22+3*a]=o.occurrence.seq;out[23+3*a]=o.occurrence.tick;
  }
}
}
