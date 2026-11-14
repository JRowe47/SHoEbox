// Explicit extension ABI 1. Historical checkpoint records stay unchanged.
// Import validates into a temporary EventRuntime and commits atomically.
extern "C" {
int event_configure(void*p,int heads,int local,int frontier,int replay,float credit_lr,float synthetic,float replay_scale){
  auto&m=*static_cast<Model*>(p);auto&e=m.event;
  if(heads<0||heads>2||(local!=0&&local!=1)||frontier<1||frontier>64||replay<0||replay>32||
     !std::isfinite(credit_lr)||credit_lr<0||credit_lr>1||!std::isfinite(synthetic)||synthetic<0||synthetic>2||
     !std::isfinite(replay_scale)||replay_scale<0||replay_scale>2||m.composition.stream.byte_position||m.composition.stream.semantic_serial||
     !m.reverse_roots.empty()||!m.reverse_bank_version.empty()||
     ((heads||local)&&(m.reverse_epoch_bytes!=1||m.reverse_policy!=2)))return 0;
  for(int r=0;r<N;r++)if(m.s.valid[r])return 0;
  e.heads=heads;e.local=local;e.frontier=frontier;e.replay_budget=replay;
  e.credit_lr=credit_lr;e.synthetic_scale=synthetic;e.replay_scale=replay_scale;
  if(!e.initialized)for(int r=2;r<N;r++)for(int i=0;i<D;i++)for(int k=0;k<MK_FEATURES;k++)
    e.parameters.readout[r][i][k]=.125f*std::sin(float(1+(r+1)*(i+3)*(k+7)));
  e.initialized=true;return 1;
}
void event_status(void*p,uint64_t*out,double*norms){
  auto&m=*static_cast<Model*>(p);memcpy(out,m.event.metrics,sizeof(m.event.metrics));
  memcpy(norms,m.event.teacher_norm,sizeof(m.event.teacher_norm));
}
void event_reset_status(void*p){auto&e=static_cast<Model*>(p)->event;memset(e.metrics,0,sizeof(e.metrics));memset(e.teacher_norm,0,sizeof(e.teacher_norm));}
size_t event_state_size(void*p,int fitted){
  auto&m=*static_cast<Model*>(p);auto&e=m.event;size_t n=12*sizeof(uint64_t)+sizeof(EventParameters);
  if(fitted)return n;
  n+=sizeof(e.memory)+sizeof(e.ordinal)+sizeof(e.replay);
  for(int r=2;r<N;r++){
    for(int k=0;k<LACE_RING;k++)if(e.context[r][k].tick)n+=2*sizeof(uint64_t)+sizeof(ContextEligibility);
    n+=e.archive[r].size()*(sizeof(uint64_t)+sizeof(ContextEligibility));
  }
  for(const auto&v:e.eligibility)if(v.key.ref.serial&&constituent_record(m,v.key.ref))n+=sizeof(EventEligibility);
  return n;
}
void event_export(void*p,void*out,int fitted){
  auto&m=*static_cast<Model*>(p);auto&e=m.event;
  uint64_t h[12]{1,uint64_t(fitted),uint64_t(e.heads),uint64_t(e.local),uint64_t(e.frontier),uint64_t(e.replay_budget),0,0,0,uint64_t(E),0,0};
  if(!fitted){
    for(int r=2;r<N;r++){for(int k=0;k<LACE_RING;k++)h[6]+=e.context[r][k].tick!=0;h[7]+=e.archive[r].size();}
    for(const auto&v:e.eligibility)h[8]+=v.key.ref.serial&&constituent_record(m,v.key.ref);
  }
  auto*dest=static_cast<unsigned char*>(out);
  auto put=[&](const void*data,size_t n){memcpy(dest,data,n);dest+=n;};
  put(h,sizeof(h));put(&e.parameters,sizeof(e.parameters));if(fitted)return;
  put(e.memory,sizeof(e.memory));put(e.ordinal,sizeof(e.ordinal));put(e.replay,sizeof(e.replay));
  for(int r=2;r<N;r++)for(int k=0;k<LACE_RING;k++)if(e.context[r][k].tick){
    uint64_t location[2]{uint64_t(r),uint64_t(k)};put(location,sizeof(location));put(&e.context[r][k],sizeof(ContextEligibility));
  }
  for(int r=2;r<N;r++)for(const auto&entry:e.archive[r]){uint64_t region=r;put(&region,sizeof(region));put(&entry.second,sizeof(ContextEligibility));}
  for(const auto&v:e.eligibility)if(v.key.ref.serial&&constituent_record(m,v.key.ref))put(&v,sizeof(v));
}
int event_import(void*p,const void*input,size_t size,int fitted){
  auto&m=*static_cast<Model*>(p);auto&e=m.event;
  if(size<12*sizeof(uint64_t)+sizeof(EventParameters)||m.content_credit_depth)return 0;
  auto*data=static_cast<const unsigned char*>(input);size_t remain=size;
  auto get=[&](void*out,size_t n){if(n>remain)return false;memcpy(out,data,n);data+=n;remain-=n;return true;};
  uint64_t h[12];get(h,sizeof(h));
  if(h[0]!=1||h[1]!=uint64_t(fitted)||h[2]!=uint64_t(e.heads)||h[3]!=uint64_t(e.local)||
     h[4]!=uint64_t(e.frontier)||h[5]!=uint64_t(e.replay_budget)||h[9]!=E||h[10]||h[11]||h[6]>N*LACE_RING||
     h[7]>size/sizeof(ContextEligibility)||h[8]>uint64_t(constituent_capacity(m))||
     (fitted&&(h[6]||h[7]||h[8])))return 0;
  auto fresh=std::make_unique<EventRuntime>();auto&z=*fresh;
  auto finite=[](const float*values,size_t n){for(size_t i=0;i<n;i++)if(!std::isfinite(values[i]))return false;return true;};
  get(&z.parameters,sizeof(z.parameters));
  if(!finite(reinterpret_cast<const float*>(&z.parameters),sizeof(z.parameters)/sizeof(float)))return 0;
  if(fitted){if(remain)return 0;e.parameters=z.parameters;return 1;}
  if(!get(z.memory,sizeof(z.memory))||!get(z.ordinal,sizeof(z.ordinal))||!get(z.replay,sizeof(z.replay))||
     !finite(&z.memory[0][0][0][0],N*MK_HEADS*MK_DIM*MK_DIM))return 0;
  auto context_valid=[&](const ContextEligibility&v){
    return v.tick>0&&v.tick<=m.s.tick&&finite(&v.before[0][0][0],MK_HEADS*MK_DIM*MK_DIM)&&
      finite(&v.query_derivative[0][0],MK_FEATURES)&&finite(v.read,MK_FEATURES);
  };
  for(uint64_t n=0;n<h[6];n++){
    uint64_t loc[2];ContextEligibility v{};if(!get(loc,sizeof(loc))||!get(&v,sizeof(v))||
      loc[0]<2||loc[0]>=N||loc[1]>=LACE_RING||!e.heads||!context_valid(v)||
      m.laces.records[loc[0]][loc[1]].tick!=v.tick||z.context[loc[0]][loc[1]].tick)return 0;
    z.context[loc[0]][loc[1]]=v;
  }
  for(uint64_t n=0;n<h[7];n++){
    uint64_t r;ContextEligibility v{};if(!get(&r,sizeof(r))||!get(&v,sizeof(v))||r<2||r>=N||!e.heads||!context_valid(v)||
      !m.neural_archive[r].count(v.tick)||!z.archive[r].emplace(v.tick,v).second)return 0;
  }
  auto key_valid=[&](const EventKey&key){
    if(!key.ref.serial)return key.ordinal==0;
    return key.ref.region>=2&&key.ref.region<N&&key.ref.index>=0&&key.ordinal>0&&key.ordinal<=z.ordinal[key.ref.region]&&finite(key.feature,MK_DIM);
  };
  z.eligibility.resize(e.local?constituent_capacity(m):0);
  for(uint64_t n=0;n<h[8];n++){
    EventEligibility v{};if(!get(&v,sizeof(v))||!e.local||!key_valid(v.key)||!constituent_record(m,v.key.ref)||
      z.eligibility[v.key.ref.index].key.ref.serial)return 0;
    const auto&node=*constituent_record(m,v.key.ref);
    if(memcmp(&node.ref,&v.key.ref,sizeof(SemanticRef)))return 0;
    for(int k=0;k<node.count;k++)if(!key_valid(v.children[k])||
      memcmp(&v.children[k].ref,&node.edges[k].source,sizeof(SemanticRef)))return 0;
    z.eligibility[v.key.ref.index]=v;
  }
  if(remain)return 0;
  // Every live active producer must have its matching extension, not merely
  // every supplied extension. This rejects truncated-but-reframed snapshots.
  if(e.heads)for(int r=2;r<N;r++){
    for(int k=0;k<LACE_RING;k++)if(m.laces.records[r][k].tick&&m.laces.records[r][k].expert>=0&&
      z.context[r][k].tick!=m.laces.records[r][k].tick)return 0;
    for(const auto&entry:m.neural_archive[r])if(entry.second.lace.expert>=0&&!z.archive[r].count(entry.first))return 0;
  }
  if(e.local)for(int i=0;i<constituent_capacity(m);i++)if(constituent_slot(m,i).ref.serial&&
    z.eligibility[i].key.ref.serial!=constituent_slot(m,i).ref.serial)return 0;
  for(int r=0;r<N;r++)for(int t=0;t<2;t++){
    const auto&slot=z.replay[r][t];const auto&q=slot.request;if(!q.ref.serial)continue;
    if(!e.local||q.ref.region!=r||q.teaching!=t||q.receiver<2||q.receiver>=N||!constituent_record(m,q.ref)||q.reserved||slot.reserved||
      !std::isfinite(slot.priority)||slot.priority<=0||!std::isfinite(q.rate)||!finite(q.gradient,D)||!finite(q.scales,N))return 0;
  }
  e.parameters=z.parameters;memcpy(e.memory,z.memory,sizeof(e.memory));memcpy(e.ordinal,z.ordinal,sizeof(e.ordinal));
  memcpy(e.context,z.context,sizeof(e.context));memcpy(e.replay,z.replay,sizeof(e.replay));
  for(int r=0;r<N;r++)e.archive[r]=std::move(z.archive[r]);e.eligibility=std::move(z.eligibility);
  return 1;
}
}
