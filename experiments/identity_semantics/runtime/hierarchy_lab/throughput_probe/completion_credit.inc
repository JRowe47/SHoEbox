inline int content_coord(int,int);
inline float content_sign(int,int);
inline int constituent_coord(const Model&m,int slot,int j){return slot<0?content_coord(j,-100-slot):m.perm[slot*D+j];}
inline float constituent_sign(const Model&m,int slot,int j){return slot<0?content_sign(j,-100-slot):m.signs[slot*D+j];}
inline ConstituentRecord* constituent_record(Model&m,SemanticRef ref){
  if(!ref.serial||ref.region<2||ref.region>=N||ref.index<0||ref.index>=SEM_RING)return nullptr;
  auto&node=m.composition.stream.semantic_records[ref.index];
  return node.ref.serial==ref.serial?&node:nullptr;
}
// A saved composition is a small DAG: multiple roles may use the same source.
// Accumulate those paths before updating its actual adapter. Reverse creation
// order is topological, including a sequence of locally encoded bound frames.
inline void constituent_update(Model&m,SemanticRef root,const float*gradient,float rate,int receiver){
  if(!root.serial||rate==0)return;
  auto&stream=m.composition.stream;
  uint64_t mark=++m.semantic_work_serial;
  SemanticRef work[SEM_WORK];int count=0;
  auto admit=[&](SemanticRef ref){
    if(!ref.serial)return;
    if(!constituent_record(m,ref)){stream.completion_metrics[receiver][1]++;return;}
    if(m.semantic_mark[ref.index]==mark)return;
    if(count==SEM_WORK){stream.completion_metrics[receiver][2]++;return;}
    m.semantic_mark[ref.index]=mark;
    memset(m.semantic_gradient[ref.index],0,D*sizeof(float));
    work[count++]=ref;
  };
  admit(root);if(!count)return;
  for(int i=0;i<count;i++){
    auto*node=constituent_record(m,work[i]);
    for(int e=0;e<node->count;e++)admit(node->edges[e].source);
  }
  std::sort(work,work+count,[](SemanticRef a,SemanticRef b){return a.serial>b.serial;});
  memcpy(m.semantic_gradient[root.index],gradient,D*sizeof(float));
  for(int i=0;i<count;i++){
    auto ref=work[i];auto&node=*constituent_record(m,ref);int r=ref.region;
    float*incoming=m.semantic_gradient[ref.index];float local[D];
    double norm2=0;float gate_gradient=0;
    for(int j=0;j<D;j++){
      local[j]=incoming[j]*node.derivative[j];
      norm2+=double(local[j])*local[j];gate_gradient+=local[j]*node.payload[j]*node.gain_derivative;
    }
    // Policy affects parameter updates, not the derivative of the saved code.
    if(!m.frozen_region[r]&&(r==receiver||m.teaching_scale[r][0]!=0)){
      float scale=rate*(r==receiver?1.f:m.teaching_scale[r][0]);
      for(int j=0;j<D;j++)for(int k=0;k<8;k++)
        m.composition.symbols[r-2][j][k]-=scale*local[j]*node.features[k];
      m.composition.payload_gates[r-2]-=scale*gate_gradient;
      m.composition.symbol_updates[r-2]++;
    }
    stream.completion_metrics[r][0]++;
    if(r!=receiver){stream.message_metrics[r][0]++;stream.message_metrics[r][1]+=std::sqrt(norm2);}
    for(int e=0;e<node.count;e++){
      const auto&edge=node.edges[e];auto child=edge.source;
      if(!child.serial||!constituent_record(m,child)||m.semantic_mark[child.index]!=mark)continue;
      float*out=m.semantic_gradient[child.index];
      for(int j=0;j<D;j++)out[constituent_coord(m,edge.slot,j)]+=edge.weight*constituent_sign(m,edge.slot,j)*node.payload_derivative[j]*local[j];
    }
  }
}
inline void completion_semantic_update(Model&m,int r,LaceRecord&v,InputCredit&input,const float*du,const float*dcue,float rate){
  float gradient[D];
  for(int j=0;j<D;j++)gradient[j]=std::clamp(input.message.qa*(input.message.alpha*du[j]+(dcue?dcue[j]:0.f)),-2.f,2.f);
  constituent_update(m,input.semantic,gradient,rate*m.message_config[3]/norm(v.u),r);
}
