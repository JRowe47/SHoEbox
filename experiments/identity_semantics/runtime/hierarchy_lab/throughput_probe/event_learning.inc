inline EventEligibility* event_eligibility(Model&m,SemanticRef ref){
  if(ref.index<0||size_t(ref.index)>=m.event.eligibility.size())return nullptr;
  auto&entry=m.event.eligibility[ref.index];
  return entry.key.ref.serial==ref.serial&&entry.key.ref.region==ref.region?&entry:nullptr;
}
inline bool event_recent(const Model&m,const EventKey&key){
  return key.ref.serial&&key.ref.region>=2&&key.ref.region<N&&key.ordinal&&
    m.event.ordinal[key.ref.region]>=key.ordinal&&m.event.ordinal[key.ref.region]-key.ordinal<uint64_t(m.event.frontier);
}
inline bool event_follow(Model&m,SemanticRef parent,int edge){
  if(!m.event.local)return true;
  auto*saved=event_eligibility(m,parent);
  if(!saved){m.event.metrics[parent.region][9]++;return false;}
  return !m.event.local_only&&event_recent(m,saved->children[edge]);
}
inline void event_predict(Model&m,const EventKey&key,int teaching,float*out){
  const auto&weights=m.event.parameters.credit[key.ref.region][teaching];
  for(int j=0;j<D;j++){
    float value=0;for(int a=0;a<MK_DIM;a++)value+=weights[j][a]*key.feature[a];
    out[j]=std::clamp(value,-2.f,2.f);
  }
}
inline void event_boundary(Model&m,const EventKey&key,const float*g,float rate,const Model::ReverseRoot&policy){
  int r=key.ref.region,t=policy.teaching;
  if(!key.ref.serial||r<2||r>=N||!key.ordinal){m.event.metrics[policy.receiver][9]++;return;}
  // An actual 64-component directional teacher crosses the consolidated
  // interface. Its predictor learns from the exact saved content key.
  m.event.metrics[r][3]++;m.event.metrics[r][10+t]++;
  float prediction[D],denominator=1;event_predict(m,key,t,prediction);
  for(float f:key.feature)denominator+=f*f;
  double priority=0;
  for(int j=0;j<D;j++){
    float target=std::clamp(g[j],-2.f,2.f),error=target-prediction[j];priority+=double(error)*error;
    m.event.teacher_norm[r][t]+=std::abs(double(g[j]));
    if(!m.frozen_region[r]&&policy.scales[r]!=0)for(int a=0;a<MK_DIM;a++)
      m.event.parameters.credit[r][t][j][a]+=m.event.credit_lr*error*key.feature[a]/denominator;
  }
  if(m.event.local_only||!m.event.replay_budget||!m.event.replay_scale||!constituent_record(m,key.ref))return;
  auto&slot=m.event.replay[r][t];float score=float(priority)*std::abs(rate*policy.scales[r]);
  if(score==0)return;
  // One hardest outstanding record per region and objective. Every displaced
  // signal already trained the compact predictor; displacement is counted.
  if(!slot.request.ref.serial||score>slot.priority){
    m.event.metrics[r][7]+=slot.request.ref.serial!=0;
    slot=EventReplay{};slot.request=policy;slot.request.ref=key.ref;slot.request.rate=rate*m.event.replay_scale;
    memcpy(slot.request.gradient,g,D*sizeof(float));slot.priority=score;m.event.metrics[r][5]++;
  }
}
inline void event_register(Model&m,const ConstituentRecord&node){
  if(!m.event.local)return;
  if(size_t(node.ref.index)>=m.event.eligibility.size())m.event.eligibility.resize(constituent_capacity(m));
  // Resize before taking a reference: this vector can grow with the slab.
  auto&saved=m.event.eligibility[node.ref.index];saved=EventEligibility{};
  saved.key.ref=node.ref;saved.key.ordinal=++m.event.ordinal[node.ref.region];
  const auto*content=content_expert_record(m,node.ref);
  for(int a=0;a<MK_DIM;a++){
    float value=0;for(int b=0;b<8;b++)value+=(content?content->input[a*8+b]:node.payload[a*8+b])*.3535533905932738f;
    saved.key.feature[a]=std::tanh(value);
  }
  for(int e=0;e<node.count;e++){
    auto*child=event_eligibility(m,node.edges[e].source);
    if(child)saved.children[e]=child->key;
    else m.event.metrics[node.ref.region][9]++;
  }
  if(m.event.training)m.event.new_events.push_back(node.ref);
}
// At byte boundaries, spend a fixed number of exact local replay VJPs, then
// predict directional credit once per newly encoded event. Neither operation
// recursively reopens an old DAG. Replay still teaches each child interface.
inline void event_finish_step(Model&m){
  if(!m.event.local)return;
  m.event.local_only=true;
  for(int n=0;n<m.event.replay_budget;n++){
    EventReplay*best=nullptr;int region=0;
    for(int r=2;r<N;r++)for(int t=0;t<2;t++){
      auto&candidate=m.event.replay[r][t];
      if(candidate.request.ref.serial&&(!best||candidate.priority>best->priority)){best=&candidate;region=r;}
    }
    if(!best)break;
    m.reverse_roots.push_back(best->request);*best=EventReplay{};
    m.event.metrics[region][6]++;constituent_union_flush(m,true);content_expert_flush(m);
  }
  for(auto ref:m.event.new_events){
    auto*saved=event_eligibility(m,ref);if(!saved){m.event.metrics[ref.region][9]++;continue;}
    for(int t=0;t<2;t++){
      Model::ReverseRoot request{};request.ref=ref;request.receiver=ref.region;request.teaching=t;
      request.rate=m.event.rates[t]*m.event.synthetic_scale*m.reverse_update_scale;
      if(request.rate==0||m.frozen_region[ref.region])continue;
      event_predict(m,saved->key,t,request.gradient);float magnitude=0;
      for(float x:request.gradient)magnitude+=x*x;if(magnitude<1e-20f)continue;
      for(int r=0;r<N;r++)request.scales[r]=m.teaching_scale[r][t];
      m.reverse_roots.push_back(request);m.event.metrics[ref.region][4]++;m.event.metrics[ref.region][14+t]++;
    }
  }
  // Synthetic predictions are terminal local updates; they must not become
  // teachers for the credit predictors that produced them.
  m.event.new_events.clear();m.event.synthetic=true;
  constituent_union_flush(m,true);content_expert_flush(m);m.event.local_only=false;m.event.synthetic=false;
}
inline void event_drain(Model&m){
  int budget=m.event.replay_budget;m.event.replay_budget=N*2;event_finish_step(m);m.event.replay_budget=budget;
}
inline void event_reset_stream(Model&m){
  event_drain(m);
  memset(m.event.memory,0,sizeof(m.event.memory));memset(m.event.ordinal,0,sizeof(m.event.ordinal));
  memset(m.event.context,0,sizeof(m.event.context));memset(m.event.replay,0,sizeof(m.event.replay));
  for(auto&archive:m.event.archive)archive.clear();m.event.eligibility.clear();m.event.new_events.clear();
  m.event.training=false;m.event.rates[0]=m.event.rates[1]=0;
}
