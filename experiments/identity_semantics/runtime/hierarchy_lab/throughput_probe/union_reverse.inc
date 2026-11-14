// Policy 2: a weighted mean of root adjoints is differentiated once per unique
// node and regional teaching-policy group. Multiply by the group's absolute
// rate mass at the update. Clipping is applied to that mean and the combined
// router step, not separately to each root. This is an explicit learning rule;
// policy 1 remains the request-wise oracle and the policy of old checkpoints.
inline bool reverse_same_policy(const Model::ReverseRoot&a,const Model::ReverseRoot&b){
  return a.teaching==b.teaching&&!memcmp(a.scales,b.scales,sizeof(a.scales));
}
inline void constituent_union_flush(Model&m,bool force){
  if(m.reverse_flushing||m.reverse_roots.empty()||(!force&&!m.reverse_bank_version.empty()))return;
  m.reverse_flushing=true;m.reverse_metrics[4]++;
  auto&roots=m.reverse_roots;std::vector<bool> used(roots.size(),false);
  for(size_t first=0;first<roots.size();first++)if(!used[first]){
    const auto&policy=roots[first];std::vector<size_t> group;float mass=0;
    for(size_t q=first;q<roots.size();q++)if(!used[q]&&reverse_same_policy(policy,roots[q])){
      used[q]=true;group.push_back(q);mass+=std::abs(roots[q].rate);
    }
    m.reverse_metrics[5]++;
    // Keep the common singleton case in its original arithmetic order.
    if(group.size()==1){
      const auto&root=roots[first];
      constituent_update_reference(m,root.ref,root.gradient,root.rate,root.receiver,root.teaching);
      continue;
    }
    uint64_t mark=++m.semantic_work_serial;auto&work=m.reverse_work;work.clear();
    auto admit=[&](SemanticRef ref){
      if(!ref.serial)return;
      if(!constituent_record(m,ref)){m.composition.stream.completion_metrics[policy.receiver][1]++;return;}
      if(constituent_mark(m,ref.index)==mark)return;
      constituent_mark(m,ref.index)=mark;
      memset(constituent_gradient(m,ref.index),0,D*sizeof(float));work.push_back(ref);
    };
    for(auto q:group)admit(roots[q].ref);
    for(size_t i=0;i<work.size();i++){
      const auto&node=*constituent_record(m,work[i]);
      for(int e=0;e<node.count;e++)admit(node.edges[e].source);
    }
    std::sort(work.begin(),work.end(),[](SemanticRef a,SemanticRef b){return a.serial>b.serial;});
    for(auto q:group){const auto&root=roots[q];if(!constituent_record(m,root.ref))continue;
      float*out=constituent_gradient(m,root.ref.index),factor=root.rate/mass;
      for(int j=0;j<D;j++)out[j]+=factor*root.gradient[j];
    }
    m.reverse_metrics[2]+=work.size();
    // As in the reference pass, independent ready nodes share a bank tile.
    for(size_t begin=0;begin<work.size();){
      size_t end=begin;uint64_t newest_child=0;
      while(end<work.size()&&end-begin<16&&(end==begin||work[end].serial>newest_child)){
        const auto&node=*constituent_record(m,work[end]);
        for(int e=0;e<node.count;e++)newest_child=std::max(newest_child,node.edges[e].source.serial);
        ++end;
      }
      float locals[16][D],derivatives[16][D];
      const ContentExpertRecord*contents[16]{};bool done[16]{};int count=int(end-begin);
      for(int a=0;a<count;a++){
        auto ref=work[begin+a];const auto&node=*constituent_record(m,ref);
        const float*incoming=constituent_gradient(m,ref.index);
        for(int j=0;j<D;j++)locals[a][j]=incoming[j]*node.derivative[j];
        if(node.encoder)contents[a]=content_expert_record(m,ref);
      }
      for(int a=0;a<count;a++)if(contents[a]&&!done[a]){
        const ContentExpertRecord*records[16];const float*inputs[16];float*outputs[16];int n=0;
        for(int b=a;b<count;b++)if(contents[b]&&!done[b]&&
            contents[b]->ref.region==contents[a]->ref.region&&contents[b]->expert==contents[a]->expert){
          records[n]=contents[b];inputs[n]=locals[b];outputs[n]=derivatives[b];done[b]=true;n++;
        }
        content_expert_derivatives(m,records,inputs,outputs,n);m.reverse_metrics[3]+=n;
      }
      for(int a=0;a<count;a++){
        auto ref=work[begin+a];const auto&node=*constituent_record(m,ref);int r=ref.region;
        float*local=locals[a],payload[D],gate_gradient=0;double norm2=0;
        if(node.encoder&&!contents[a]){m.content_expert_metrics[r][5]++;continue;}
        for(int j=0;j<D;j++){
          payload[j]=local[j]+(node.encoder?derivatives[a][j]:0.f);
          gate_gradient+=payload[j]*node.payload[j]*node.gain_derivative;
          norm2+=double(local[j])*local[j];
        }
        float rate=mass*policy.scales[r];
        if(!m.frozen_region[r]&&rate!=0){
          for(int j=0;j<D;j++)for(int k=0;k<8;k++)
            m.composition.symbols[r-2][j][k]-=rate*local[j]*node.features[k];
          m.composition.payload_gates[r-2]-=rate*gate_gradient;
          m.composition.symbol_updates[r-2]++;
          if(node.encoder)content_expert_accumulate(m,*contents[a],local,rate,policy.teaching);
        }
        auto&stream=m.composition.stream;stream.completion_metrics[r][0]++;
        if(r!=policy.receiver){stream.message_metrics[r][0]++;stream.message_metrics[r][1]+=std::sqrt(norm2);}
        for(int e=0;e<node.count;e++){
          const auto&edge=node.edges[e];auto child=edge.source;
          if(!child.serial||!constituent_record(m,child)||constituent_mark(m,child.index)!=mark)continue;
          float*out=constituent_gradient(m,child.index);const auto&role=constituent_role(m,edge.slot);
          for(int j=0;j<D;j++)out[role.coord[j]]+=edge.weight*role.sign[j]*node.payload_derivative[j]*payload[j];
        }
      }
      begin=end;
    }
  }
  roots.clear();m.reverse_flushing=false;
}
inline void constituent_update(Model&m,SemanticRef root,const float*gradient,float rate,int receiver,int teaching=0){
  if(!root.serial||rate==0||m.reverse_update_scale==0)return;
  rate*=m.reverse_update_scale;if(rate==0)return;
  m.reverse_metrics[0]++;
  if(m.reverse_policy==1){constituent_update_reference(m,root,gradient,rate,receiver,teaching);return;}
  ContentCreditScope scope(m);m.reverse_roots.emplace_back();auto&request=m.reverse_roots.back();
  request.ref=root;request.rate=rate;request.receiver=receiver;request.teaching=teaching;
  memcpy(request.gradient,gradient,D*sizeof(float));
  for(int r=0;r<N;r++)request.scales[r]=r==receiver?1.f:m.teaching_scale[r][teaching];
}
inline void reverse_begin_epoch(Model&m){
  if(m.reverse_epoch_bytes<=1||!m.reverse_bank_version.empty())return;
  constexpr size_t weights=N*E*D*D,vectors=N*E*D;
  m.reverse_bank_version.resize(weights+2*vectors);
  memcpy(m.reverse_bank_version.data(),m.W,weights*sizeof(float));
  memcpy(m.reverse_bank_version.data()+weights,m.bias,vectors*sizeof(float));
  memcpy(m.reverse_bank_version.data()+weights+vectors,m.keys,vectors*sizeof(float));
  m.refresh_keys();
}
inline void reverse_finish_epoch(Model&m){
  bool active=!m.reverse_bank_version.empty();
  constituent_union_flush(m,true);content_expert_flush(m);
  m.reverse_bank_version.clear();m.reverse_epoch_position=0;
  if(active){m.reverse_version++;m.refresh_keys();}
}
