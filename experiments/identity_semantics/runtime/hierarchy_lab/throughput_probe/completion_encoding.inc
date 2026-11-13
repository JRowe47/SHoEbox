// Exact IDs and structural descriptors never include floating learned values.
// Every code has its own saved producer and derivative graph; neural context
// is excluded. The learned gate is bounded in [0.5,1.5].
inline int constituent_allocate(Model&m,int r,const ContentEvent*parts,int n){
  auto&s=m.composition.stream;int capacity=constituent_capacity(m);
  int start=int(s.semantic_cursor++%capacity);
  if(!constituent_slot(m,start).ref.serial)return start;
  // Mark from actual owners before recycling. Long held completed units must
  // not lose eligibility just because a faster child has published again.
  uint64_t mark=++m.semantic_work_serial;std::vector<SemanticRef> work;work.reserve(capacity);
  auto add=[&](SemanticRef ref){
    if(!constituent_record(m,ref)||constituent_mark(m,ref.index)==mark)return;
    constituent_mark(m,ref.index)=mark;work.push_back(ref);
  };
  for(int a=0;a<N;a++){
    for(int k=0;k<2;k++)add(s.history[a][k].semantic);
    for(int k=0;k<s.length[a];k++)add(s.pending[a][k].semantic);
    for(int k=0;k<LACE_RING;k++)add(s.input[a][k].semantic);
    for(const auto&entry:m.neural_archive[a])add(entry.second.input.semantic);
  }
  for(int k=0;k<n;k++)add(parts[k].semantic);
  add(m.semantic_inflight);
  for(size_t i=0;i<work.size();i++){
    auto&node=*constituent_record(m,work[i]);for(int k=0;k<node.count;k++)add(node.edges[k].source);
  }
  for(int k=0;k<capacity;k++)if(constituent_mark(m,k)!=mark){
    constituent_slot(m,k).ref=SemanticRef{};
    if(size_t(k)<m.content_records.size())m.content_records[k].ref=SemanticRef{};
    if(size_t(k)<m.pattern.assignments.size())m.pattern.assignments[k]=PatternAssignment{};
  }
  for(int k=0;k<capacity;k++){
    int index=(start+k)%capacity;
    if(constituent_mark(m,index)!=mark){s.semantic_cursor=index+1;return index;}
  }
  // Every slot is live. Grow the reference pool instead of severing credit.
  constituent_resize(m,size_t(capacity)*2-SEM_RING);
  s.semantic_cursor=capacity+1;return capacity;
}
inline void constituent_encode(Model&m,int r,ContentEvent&out,const ContentEvent*parts,int n,bool binding){
  auto&s=m.composition.stream;int index=constituent_allocate(m,r,parts,n);
  auto&node=constituent_slot(m,index);node=ConstituentRecord{};
  if(size_t(index)<m.content_records.size())m.content_records[index]=ContentExpertRecord{};
  if(size_t(index)<m.pattern.assignments.size())m.pattern.assignments[index]=PatternAssignment{};
  node.ref={++s.semantic_serial,r,index};out.semantic=node.ref;
  memcpy(node.features,out.symbol_features,sizeof(node.features));
  float squared=0,raw[D]{};
  if(binding)for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2])squared+=m.slots[4*k+3]*m.slots[4*k+3];
  float scale=1.f/std::sqrt(binding?std::max(1.f,squared):float(std::max(1,n)));
  for(int i=0;i<n;i++){
    const auto&child=parts[i];int slot=binding?child.ref.role:-100-i;
    float weight=scale*(binding?m.slots[4*slot+3]*m.regional_gain[child.ref.region]:1.f);
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
    if(contribution_apply(m,CHECK_PAYLOAD,child.ref.region))continue;
#endif
    for(int j=0;j<D;j++){int c=constituent_coord(m,slot,j);raw[j]+=weight*constituent_sign(m,slot,j)*(child.code[c]-child.structure[c]);}
    if(weight&&child.semantic.serial)node.edges[node.count++]={child.semantic,child.ref,slot,weight};
  }
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
  if(encode)content_expert_encode(m,r,node,out.structure,gain,expert);
  for(int j=0;j<D;j++){
    float payload=node.payload[j],value=0;
    for(int k=0;k<8;k++)value+=m.composition.symbols[r-2][j][k]*node.features[k];
    float pre=value+gain*payload;
    if(encode)pre+=expert[j]; // Keep historical floating-point order when disabled.
    float residual=muted?0:std::tanh(pre);
    out.code[j]=out.structure[j]+.25f*residual;
    node.derivative[j]=out.symbol_derivative[j]=muted?0:.25f*(1-residual*residual);
    node.payload[j]=payload;node.payload_derivative[j]=gain*(1-payload*payload);
  }
}
