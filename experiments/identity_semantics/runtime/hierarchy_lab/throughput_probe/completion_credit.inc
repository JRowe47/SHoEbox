inline int content_coord(int,int);
inline float content_sign(int,int);
inline int constituent_capacity(const Model&m){return SEM_RING+int(m.semantic_overflow.size());}
inline ConstituentRecord& constituent_slot(Model&m,int index){
  return index<SEM_RING?m.composition.stream.semantic_records[index]:m.semantic_overflow[index-SEM_RING];
}
inline uint64_t& constituent_mark(Model&m,int index){
  return index<SEM_RING?m.semantic_mark[index]:m.semantic_overflow_mark[index-SEM_RING];
}
inline float* constituent_gradient(Model&m,int index){
  return index<SEM_RING?m.semantic_gradient[index]:m.semantic_overflow_gradient[index-SEM_RING].data();
}
inline void constituent_resize(Model&m,size_t overflow){
  m.semantic_overflow.resize(overflow);m.semantic_overflow_mark.resize(overflow);
  m.semantic_overflow_gradient.resize(overflow);
  m.composition.stream.reserved=uint32_t(overflow);
}
inline int constituent_coord(const Model&m,int slot,int j){return slot<0?content_coord(j,-100-slot):m.perm[slot*D+j];}
inline float constituent_sign(const Model&m,int slot,int j){return slot<0?content_sign(j,-100-slot):m.signs[slot*D+j];}
inline const Model::ConstituentRole& constituent_role(Model&m,int slot){
  size_t index=slot<0?m.starts[N]+(-100-slot):slot;
  if(m.constituent_roles.size()<=index)m.constituent_roles.resize(index+1);
  auto&role=m.constituent_roles[index];
  if(!role.valid){
    for(int j=0;j<D;j++){role.coord[j]=constituent_coord(m,slot,j);role.sign[j]=constituent_sign(m,slot,j);}
    role.valid=true;
  }
  return role;
}
inline ConstituentRecord* constituent_record(Model&m,SemanticRef ref){
  if(!ref.serial||ref.region<2||ref.region>=N||ref.index<0||ref.index>=constituent_capacity(m))return nullptr;
  auto&node=constituent_slot(m,ref.index);
  return node.ref.serial==ref.serial?&node:nullptr;
}
inline void constituent_invalidate(Model&m){
  m.semantic_free.clear();m.semantic_free_valid=false;
  m.content_input_cache.clear();m.constituent_work.clear();m.constituent_roles.clear();
  for(auto&plan:m.constituent_plans){plan.root=SemanticRef{};plan.nodes.clear();}
}
// Plans contain handles, never owners or weight-dependent derivatives. Bound
// caching to 64 x 4096 handles; unusually deep roots use one reusable workspace.
inline const std::vector<SemanticRef>& constituent_schedule(Model&m,SemanticRef root,int receiver,uint64_t mark){
  auto&plan=m.constituent_plans[root.serial%m.constituent_plans.size()];
  bool cached=plan.root.serial==root.serial&&plan.root.index==root.index&&plan.root.region==root.region;
  if(cached)for(auto ref:plan.nodes)if(!constituent_record(m,ref)){cached=false;break;}
  if(cached){
    for(auto ref:plan.nodes){constituent_mark(m,ref.index)=mark;memset(constituent_gradient(m,ref.index),0,D*sizeof(float));}
    return plan.nodes;
  }
  auto&work=m.constituent_work;work.clear();bool complete=true;
  auto admit=[&](SemanticRef ref){
    if(!ref.serial)return;
    if(!constituent_record(m,ref)){m.composition.stream.completion_metrics[receiver][1]++;complete=false;return;}
    if(constituent_mark(m,ref.index)==mark)return;
    constituent_mark(m,ref.index)=mark;memset(constituent_gradient(m,ref.index),0,D*sizeof(float));work.push_back(ref);
  };
  admit(root);
  for(size_t i=0;i<work.size();i++){
    auto*node=constituent_record(m,work[i]);for(int e=0;e<node->count;e++)admit(node->edges[e].source);
  }
  std::sort(work.begin(),work.end(),[](SemanticRef a,SemanticRef b){return a.serial>b.serial;});
  if(complete&&work.size()<=4096){plan.root=root;plan.nodes=work;return plan.nodes;}
  plan.root=SemanticRef{};
  return work;
}
// A saved composition is a small DAG: multiple roles may use the same source.
// Accumulate those paths before updating its actual adapter. Reverse creation
// order is topological, including a sequence of locally encoded bound frames.
inline void constituent_update(Model&m,SemanticRef root,const float*gradient,float rate,int receiver,int teaching=0){
  if(!root.serial||rate==0)return;
  ContentCreditScope content_scope(m);
  auto&stream=m.composition.stream;
  uint64_t mark=++m.semantic_work_serial;
  const auto&work=constituent_schedule(m,root,receiver,mark);if(work.empty())return;
  memcpy(constituent_gradient(m,root.index),gradient,D*sizeof(float));
  for(size_t begin=0;begin<work.size();){
    // A contiguous ready batch ends before any of its direct children. Keep
    // all parameter/edge updates in reverse-serial order after the VJPs.
    size_t end=begin;uint64_t newest_child=0;
    while(end<work.size()&&end-begin<16&&(end==begin||work[end].serial>newest_child)){
      const auto&node=*constituent_record(m,work[end]);
      for(int e=0;e<node.count;e++)newest_child=std::max(newest_child,node.edges[e].source.serial);
      ++end;
    }
    float locals[16][D],payloads[16][D],derivatives[16][D];double norms[16]{};
    const ContentExpertRecord*contents[16]{};bool done[16]{};
    int count=int(end-begin);
    for(int a=0;a<count;a++){
      auto ref=work[begin+a];const auto&node=*constituent_record(m,ref);
      const float*incoming=constituent_gradient(m,ref.index);
      for(int j=0;j<D;j++){
        float value=incoming[j]*node.derivative[j];locals[a][j]=payloads[a][j]=value;
        norms[a]+=double(value)*value;
      }
      if(node.encoder)contents[a]=content_expert_record(m,ref);
    }
    for(int a=0;a<count;a++)if(contents[a]&&!done[a]){
      const ContentExpertRecord*records[16];const float*inputs[16];float*outputs[16];int n=0;
      for(int b=a;b<count;b++)if(contents[b]&&!done[b]&&
          contents[b]->ref.region==contents[a]->ref.region&&contents[b]->expert==contents[a]->expert){
        records[n]=contents[b];inputs[n]=locals[b];outputs[n]=derivatives[b];done[b]=true;n++;
      }
      content_expert_derivatives(m,records,inputs,outputs,n);
    }
    for(int a=0;a<count;a++){
    auto ref=work[begin+a];auto&node=*constituent_record(m,ref);int r=ref.region;
    float*local=locals[a],*payload_local=payloads[a];double norm2=norms[a];float gate_gradient=0;
    auto*content=contents[a];
    if(node.encoder){
      if(!content){m.content_expert_metrics[r][5]++;continue;}
      for(int j=0;j<D;j++)payload_local[j]+=derivatives[a][j];
    }
    for(int j=0;j<D;j++)gate_gradient+=payload_local[j]*node.payload[j]*node.gain_derivative;
    // Policy affects parameter updates, not the derivative of the saved code.
    if(!m.frozen_region[r]&&(r==receiver||m.teaching_scale[r][teaching]!=0)){
      float scale=rate*(r==receiver?1.f:m.teaching_scale[r][teaching]);
      for(int j=0;j<D;j++)for(int k=0;k<8;k++)
        m.composition.symbols[r-2][j][k]-=scale*local[j]*node.features[k];
      m.composition.payload_gates[r-2]-=scale*gate_gradient;
      m.composition.symbol_updates[r-2]++;
      if(node.encoder)content_expert_accumulate(m,*content,local,scale,teaching);
    }
    stream.completion_metrics[r][0]++;
    if(r!=receiver){stream.message_metrics[r][0]++;stream.message_metrics[r][1]+=std::sqrt(norm2);}
    for(int e=0;e<node.count;e++){
      const auto&edge=node.edges[e];auto child=edge.source;
      if(!child.serial||!constituent_record(m,child)||constituent_mark(m,child.index)!=mark)continue;
      float*out=constituent_gradient(m,child.index);
      const auto&role=constituent_role(m,edge.slot);
      for(int j=0;j<D;j++)out[role.coord[j]]+=edge.weight*role.sign[j]*node.payload_derivative[j]*payload_local[j];
    }
    }
    begin=end;
  }
}
inline void completion_semantic_update(Model&m,int r,LaceRecord&v,InputCredit&input,const float*du,const float*dcue,float rate,int teaching=0){
  float gradient[D];
  for(int j=0;j<D;j++)gradient[j]=std::clamp(input.message.qa*(input.message.alpha*du[j]+(dcue?dcue[j]:0.f)),-2.f,2.f);
  constituent_update(m,input.semantic,gradient,rate*m.message_config[3]/norm(v.u),r,teaching);
}
