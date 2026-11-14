#include "semantic_ownership.inc"
// Exact IDs and structural descriptors never include floating learned values.
// Every code has its own saved producer and derivative graph; neural context
// is excluded. The learned gate is bounded in [0.5,1.5].
inline int constituent_allocate(Model&m,int r,const ContentEvent*parts,int n){
  auto&s=m.composition.stream;int capacity=constituent_capacity(m);
  int start=int(s.semantic_cursor%capacity);
  if(!m.semantic_free_valid){
    m.semantic_free.clear();m.semantic_free.reserve(capacity);
    for(int k=capacity-1;k>=0;k--){int index=(start+k)%capacity;
      if(!constituent_slot(m,index).ref.serial)m.semantic_free.push_back(index);
    }
    m.semantic_free_valid=true;
  }
  // Consume the previous collection's inventory in O(1), including after
  // import. Cached handles do not count as owners and never keep nodes alive.
  while(!m.semantic_free.empty()){
    int index=m.semantic_free.back();m.semantic_free.pop_back();
    if(!constituent_slot(m,index).ref.serial){s.semantic_cursor=index+1;return index;}
  }
  constituent_retain(m,parts,n);
  for(int k=0;k<capacity;k++)if(!constituent_retained(m,k)){
    if(m.event.local&&constituent_slot(m,k).ref.serial)SHOEBOX_DIAGNOSTIC(m.event.metrics[constituent_slot(m,k).ref.region][8]++;)
    constituent_slot(m,k).ref=SemanticRef{};
    if(size_t(k)<m.event.eligibility.size())m.event.eligibility[k]=EventEligibility{};
    if(size_t(k)<m.content_records.size())m.content_records[k].ref=SemanticRef{};
    if(size_t(k)<m.pattern.assignments.size())m.pattern.assignments[k]=PatternAssignment{};
  }
  for(int k=capacity-1;k>=0;k--){
    int index=(start+k)%capacity;
    if(!constituent_retained(m,index))m.semantic_free.push_back(index);
  }
  if(!m.semantic_free.empty()){
    int index=m.semantic_free.back();m.semantic_free.pop_back();s.semantic_cursor=index+1;return index;
  }
  // Every slot is live. Grow the reference pool instead of severing credit.
  constituent_resize(m,size_t(capacity)*2-SEM_RING);
  for(int index=2*capacity-1;index>capacity;--index)m.semantic_free.push_back(index);
  s.semantic_cursor=capacity+1;return capacity;
}
inline void constituent_learned_encode(Model&m,int r,ContentEvent&out,ConstituentRecord&node,const float*raw){
  auto&published=out.edit_values();
  float t=std::tanh(m.composition.payload_gates[r-2]);
  float gain=m.message_config[0]*(1.f+.5f*t);
  node.gain_derivative=m.message_config[0]*.5f*(1-t*t);
  bool muted=false;
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  muted=contribution_selected(m,CHECK_SEMANTIC,r);
#endif
  float expert[D]{};for(int j=0;j<D;j++)node.payload[j]=std::tanh(raw[j]);
  bool encode=m.content_experts;
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  if(contribution_apply(m,CHECK_CONTENT_EXPERT,r))encode=false;
#endif
  int selected=encode?content_expert_encode(m,r,node,published.structure,gain,expert):0;
  for(int j=0;j<D;j++){
    float payload=node.payload[j],value=0;
    for(int k=0;k<8;k++)value+=m.composition.symbols[r-2][j][k]*node.features[k];
    float pre=value+gain*payload;
    if(encode)pre+=expert[j]; // Keep historical floating-point order when disabled.
    float residual=muted?0:std::tanh(pre);
    published.code[j]=published.structure[j]+.25f*residual;
    node.derivative[j]=published.symbol_derivative[j]=muted?0:.25f*(1-residual*residual);
    node.payload[j]=payload;node.payload_derivative[j]=gain*(1-payload*payload);
  }
  if(m.forward_only){auto&v=out.edit_forward();v.encoder=node.encoder;v.expert=selected;}
  else event_register(m,node);
}
inline void constituent_encode(Model&m,int r,ContentEvent&out,const ContentEvent*parts,int n,bool binding,bool descriptor=false){
  auto&s=m.composition.stream;int index=m.forward_only?-1:constituent_allocate(m,r,parts,n);
  ConstituentRecord transient{};
  auto&node=m.forward_only?transient:constituent_slot(m,index);node=ConstituentRecord{};
  if(size_t(index)<m.content_records.size())m.content_records[index]=ContentExpertRecord{};
  if(size_t(index)<m.pattern.assignments.size())m.pattern.assignments[index]=PatternAssignment{};
  node.ref={++s.semantic_serial,r,index};out.semantic=node.ref;
  auto&published=out.edit_values();
  memcpy(node.features,published.symbol_features,sizeof(node.features));
  float squared=0,raw[D]{};
  if(binding)for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2])squared+=m.slots[4*k+3]*m.slots[4*k+3];
  float scale=1.f/std::sqrt(binding?std::max(1.f,squared):float(std::max(1,n)));
  for(int i=0;i<n;i++){
    const auto&child=parts[i];int slot=binding?child.ref.role:-100-i;
    float weight=scale*(binding?m.slots[4*slot+3]*m.regional_gain[child.ref.region]:1.f);
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
    if(contribution_apply(m,CHECK_PAYLOAD,child.ref.region))continue;
#endif
    const auto&cached=publication_role(m,slot,child);
    for(int j=0;j<D;j++)raw[j]+=weight*constituent_sign(m,slot,j)*cached.delta[j];
    if(!m.forward_only&&weight&&child.semantic.serial)node.edges[node.count++]={child.semantic,child.ref,slot,weight};
  }
  if(descriptor){
    // No local parameters or expert selection. Preserve the exact child DAG
    // and the VJP of the fixed signed role sum for subsequent completion credit.
    node.encoder=2;
    for(int j=0;j<D;j++){
      published.code[j]=published.structure[j]+raw[j];
      node.payload[j]=raw[j];node.derivative[j]=node.payload_derivative[j]=1.f;
    }
    if(m.forward_only){auto&v=out.edit_forward();v.encoder=2;memcpy(v.frame_payload,raw,sizeof(raw));}
    else event_register(m,node);return;
  }
  constituent_learned_encode(m,r,out,node,raw);
}
// A single completed frame keeps its exact binding identity/roles. Clone its
// immutable descriptor into a new learned producer and use the saved role sum;
// do not invent a one-child temporal wrapper or classify the same binding twice.
inline ContentEvent constituent_complete_frame(Model&m,int r,const ContentEvent&frame){
  if(m.forward_only){
    if(!frame.forward||frame.forward->encoder!=2)throw std::runtime_error("Missing forward frame descriptor");
    ContentEvent out=frame;ConstituentRecord node{};
    node.ref={++m.composition.stream.semantic_serial,r,-1};out.semantic=node.ref;out.ref.seq=0;
    memcpy(node.features,out.values().symbol_features,sizeof(node.features));
    constituent_learned_encode(m,r,out,node,frame.forward->frame_payload);
    if(out.forward->has_pattern)out.edit_forward().pattern.ref=node.ref;
    return out;
  }
  auto*source=constituent_record(m,frame.semantic);
  if(!source||source->encoder!=2)throw std::runtime_error("Missing lightweight frame producer");
  ConstituentRecord saved=*source;PatternAssignment assignment{};
  if(size_t(frame.semantic.index)<m.pattern.assignments.size())assignment=m.pattern.assignments[frame.semantic.index];
  int index=constituent_allocate(m,r,&frame,1);auto&node=constituent_slot(m,index);
  node=saved;node.ref={++m.composition.stream.semantic_serial,r,index};node.encoder=0;
  if(size_t(index)<m.content_records.size())m.content_records[index]=ContentExpertRecord{};
  if(size_t(index)<m.pattern.assignments.size())m.pattern.assignments[index]=PatternAssignment{};
  ContentEvent out=frame;out.semantic=node.ref;out.ref.seq=0;
  constituent_learned_encode(m,r,out,node,saved.payload);
  if(assignment.ref.serial==frame.semantic.serial){
    if(size_t(index)>=m.pattern.assignments.size())m.pattern.assignments.resize(constituent_capacity(m));
    assignment.ref=node.ref;m.pattern.assignments[index]=assignment;
  }
  return out;
}
