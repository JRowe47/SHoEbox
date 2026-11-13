inline bool paired_enabled(const Model&m){return m.paired.enabled&&m.composition.enabled;}
inline size_t paired_occupied(const Model&m){size_t n=0;for(const auto&e:m.paired.slots)n+=e.id!=0;return n;}
inline void paired_reset_stream(Model&m){
  auto&z=m.paired;uint64_t epoch=z.stream.epoch+uint64_t(z.enabled);z.stream=PairedStream{};z.stream.epoch=epoch;
  memset(z.metrics,0,sizeof(z.metrics));
}
inline void paired_clear(Model&m){
  // Retained neural records may still carry old return IDs after clear_memory.
  // Preserve the serial allocator so replacement can never alias their credit.
  m.paired.slots.clear();paired_reset_stream(m);
  memset(m.s.returned,0,sizeof(m.s.returned));memset(m.s.return_birth,0,sizeof(m.s.return_birth));
  m.s.recall_slot[0]=m.s.recall_slot[1]=-1;
}
inline PairedObservation paired_observation(Model&m,int a,const float*q,const ContentEvent&event,bool published){
  PairedObservation out{};out.occurrence=event.ref;out.occurrence.region=15+a;
  out.occurrence.tick=m.s.tick;
  if(!published)out.occurrence.seq=m.composition.stream.history[15+a][0].ref.seq+1;
  out.stream=m.paired.stream.epoch;out.clock_begin=event.clock_begin;out.clock_end=event.clock_end;
  memcpy(out.value,q,sizeof(out.value));normalize_key(m,a,q,out.key);return out;
}
inline void paired_buckets(Model&m,const PairedObservation*pair,int*out){
  float joint[D];
  for(int j=0;j<D;j++)joint[j]=.7071067811865475f*(pair[0].key[j]+(j%2?-1.f:1.f)*pair[1].key[(j*13+7)%D]);
  buckets(joint,int(m.cfg[12]),out);
}
inline float paired_similarity(const PairedObservation*a,const PairedObservation*b){
  return .5f*(dot64(a[0].key,b[0].key)+dot64(a[1].key,b[1].key));
}
inline bool paired_same_occurrences(const PairedObservation*a,const PairedObservation*b){
  for(int side=0;side<2;side++)if(a[side].stream!=b[side].stream||a[side].occurrence.seq!=b[side].occurrence.seq)return false;
  return true;
}
inline int paired_lookup(Model&m,const PairedObservation*query,const int*bs,float&similarity){
  int best=-1,ways=int(m.cfg[13]);similarity=-1;
  for(int i=0;i<4;i++){
    bool duplicate=false;for(int j=0;j<i;j++)duplicate|=bs[j]==bs[i];if(duplicate)continue;
    for(int w=0;w<ways;w++){
      int k=bs[i]*ways+w;m.s.pc[1]++;
      if(size_t(k)>=m.paired.slots.size())continue;
      const auto&e=m.paired.slots[k];if(!e.id||paired_same_occurrences(query,e.members))continue;
      m.s.pc[2]++;m.paired.metrics[9]++;
      float score=paired_similarity(query,e.members);if(score>similarity){similarity=score;best=k;}
    }
  }
  return best;
}
inline void paired_read(Model&m,int a,const float*q,const ContentEvent&event){
  auto&z=m.paired;auto&s=m.s;auto&stream=z.stream;
  // The first EC request locks both return ports for this entire regional pass.
  // Later asynchronous publications become cues for the next joint transaction.
  if(stream.read_tick==s.tick){z.metrics[14]++;return;}
  stream.read_tick=s.tick;stream.read_episode=0;stream.read_slot=-1;stream.similarity=stream.gate=0;
  memset(s.returned,0,sizeof(s.returned));memset(s.return_birth,0,sizeof(s.return_birth));
  s.recall_slot[0]=s.recall_slot[1]=-1;
  stream.query[0]=stream.latest[0];stream.query[1]=stream.latest[1];
  stream.query[a]=paired_observation(m,a,q,event,false);
  z.metrics[8]++;
  if(!stream.query[0].stream||!stream.query[1].stream){z.metrics[13]++;return;}
  if(z.slots.empty()||s.ablation==2||s.tick%int(m.cfg[17])||(m.cfg[1]<=0&&!m.cfg[21]))return;
  int bs[4];paired_buckets(m,stream.query,bs);s.pc[0]++;
  float sim;int best=paired_lookup(m,stream.query,bs,sim);stream.similarity=sim;
  float gate=best<0?0:std::clamp((sim-m.cfg[8])/std::max(1e-6f,1-m.cfg[8]),0.f,1.f)*m.cfg[1];
  if(gate<=0)return;
  if(s.ablation==4){
    int count=int(z.slots.size()),wrong=(best+count/2)%count;
    for(int k=0;k<count&&!z.slots[wrong].id;k++)wrong=(wrong+1)%count;
    best=wrong;
  }
  const auto&episode=z.slots[best];
  if(s.ablation==3||m.cfg[21]||(m.block_before&&episode.born<=m.block_before))return;
  stream.read_episode=episode.id;stream.read_slot=best;stream.gate=gate;
  int64_t age=s.tick-episode.updated;s.age_sum+=age;s.max_age=std::max(s.max_age,age);s.similarity_sum+=sim;
  s.pc[10]++;s.pc[11]+=age>256;s.pc[12]+=age>4096;s.pc[13]+=age>16384;s.reads++;z.metrics[10]++;
  for(int side=0;side<2;side++){
    s.recall_slot[side]=best;s.return_birth[side]=int64_t(episode.id);
    for(int j=0;j<D;j++)s.returned[side][j]=gate*(episode.members[side].value[j]-(m.cfg[5]?s.vmean[side][j]:0));
  }
}
inline void paired_stage(Model&m,int a,const float*q,const ContentEvent&event){
  auto&z=m.paired;auto&previous=z.stream.latest[a];
  if(previous.stream==z.stream.epoch&&previous.occurrence.seq==event.ref.seq)return;
  previous=paired_observation(m,a,q,event,true);z.metrics[a]++;
  // Preserve the existing EC-only running centers and temporal address trace.
  for(int j=0;j<D;j++){
    m.s.key_trace[a][j]+=.25f*(q[j]-m.s.qmean[a][j]-m.s.key_trace[a][j]);
    m.s.vmean[a][j]+=.01f*(q[j]-m.s.vmean[a][j]);
  }
}
inline void paired_commit(Model&m){
  auto&z=m.paired;auto&s=m.s;auto&stream=z.stream;
  if(!paired_enabled(m)||s.freeze_writes||s.ablation==2||s.tick%int(m.cfg[9]))return;
  const auto*pair=stream.latest;
  if(!pair[0].stream||!pair[1].stream){z.metrics[15]++;return;}
  bool fresh=false;for(int a=0;a<2;a++)fresh|=stream.considered_stream[a]!=pair[a].stream||stream.considered_seq[a]!=pair[a].occurrence.seq;
  if(!fresh)return;
  for(int a=0;a<2;a++){stream.considered_stream[a]=pair[a].stream;stream.considered_seq[a]=pair[a].occurrence.seq;}
  z.metrics[2]++;s.pc[3]++;
  size_t capacity=size_t(m.cfg[12])*size_t(m.cfg[13]);
  if(z.slots.empty())z.slots.resize(capacity);
  if(z.slots.size()!=capacity)return; // Configuration changes cannot reinterpret a populated store.
  int bs[4];paired_buckets(m,pair,bs);float sim;int match=paired_lookup(m,pair,bs,sim);
  float surprise=0,value_norm=1e-6f;
  if(match>=0)for(int a=0;a<2;a++)for(int j=0;j<D;j++){
    float current=pair[a].value[j],stored=z.slots[match].members[a].value[j],d=current-stored;
    surprise+=d*d;value_norm+=current*current+stored*stored;
  }
  surprise/=value_norm;
  if(match>=0&&sim>=m.cfg[15]&&surprise<m.cfg[22]){
    // Reinforce admission metadata, never blend the members of an existing ID.
    z.slots[match].updated=s.tick;z.metrics[5]++;s.pc[4]++;return;
  }
  int victim=-1;float weakest=1e30f;int ways=int(m.cfg[13]);
  for(int tier=0;tier<2;tier++){
    if(match>=0&&1-sim<(tier?.05f:m.cfg[14])&&!(tier&&surprise>=m.cfg[22]))continue;
    for(int w=0;w<ways;w++){
      int k=bs[tier]*ways+w;auto&entry=z.slots[k];
      if(tier&&entry.id)entry.priority*=1-m.cfg[23]*(1-std::clamp(entry.utility,0.f,1.f));
      float keep=entry.id?entry.priority+.1f*entry.utility:-1.f;
      if(keep<weakest){weakest=keep;victim=k;}
    }
  }
  if(victim<0){z.metrics[6]++;s.pc[9]++;return;}
  float importance=1+m.cfg[16]*std::min(3.f,std::max(0.f,1-sim)+surprise);
  float u=(random32(s)+1.0)/4294967297.0,priority=std::pow(u,1.f/importance);
  if(priority<=weakest){z.metrics[7]++;s.pc[8]++;return;}
  if(z.serial==uint64_t(INT64_MAX))return; // Saved credit handles use signed int64 IDs.
  auto&entry=z.slots[victim];bool occupied=entry.id!=0;
  entry=PairedEpisode{};entry.id=++z.serial;entry.born=entry.updated=s.tick;entry.priority=priority;
  entry.members[0]=pair[0];entry.members[1]=pair[1];z.metrics[3]++;s.pc[5]++;s.writes++;
  if(occupied){z.metrics[4]++;s.pc[6]++;}else s.pc[7]++;
}
inline void paired_utility_credit(Model&m,int r,const LaceRecord*record,const float*g,const float*h,const float*returned){
  if(!paired_enabled(m)||m.s.freeze_writes||(r!=15&&r!=16)||!m.cfg[10])return;
  int side=r-15,slot=record?record->recall_slot:m.s.recall_slot[side];
  uint64_t id=uint64_t(record?record->recall_birth:m.s.return_birth[side]);
  if(slot<0||!id)return;
  if(size_t(slot)>=m.paired.slots.size()||m.paired.slots[slot].id!=id){m.paired.metrics[12]++;return;}
  float influence=0;
  for(int i=0;i<D;i++){
    float contribution=int(m.cfg[10])==2?m.P[(r*D+i)*D+i]*returned[i]:dot64(m.P+(r*D+i)*D,returned);
    influence-=g[i]*(1-h[i]*h[i])*contribution;
  }
  auto&utility=m.paired.slots[slot].utility;utility+=.01f*(std::clamp(influence,-1.f,1.f)-utility);
  m.paired.metrics[11]++;
}
