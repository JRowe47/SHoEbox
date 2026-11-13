inline LaceRecord* find_lace(Model&,int,int64_t,int);
inline int content_coord(int,int);
inline float content_sign(int,int);
inline bool message_enabled(const Model&m){
#ifdef SHOEBOX_LEARNED_COMPLETION
  return true; // The exported code itself is now the learned input channel.
#else
  return m.message_config[0]!=0||m.message_config[1]!=0||m.message_config[2]!=0;
#endif
}
inline bool message_feedback_enabled(const Model&m){return m.message_config[1]!=0||m.message_config[2]!=0;}
inline int message_coord(const Model&m,int slot,int j){return slot<=-100?content_coord(j,-100-slot):m.perm[slot*D+j];}
inline float message_sign(const Model&m,int slot,int j){return slot<=-100?content_sign(j,-100-slot):m.signs[slot*D+j];}

// Local code is structure + the originating region's semantic adapter. The
// received payload augments q, not that exported local code. Thus the recipient
// never rebrands an inherited vector as its own adapter output. No recursive
// context or hidden uncredited chain is embedded in a semantic publication.
inline void message_payload(Model&m,int r,const ContentEvent*parts,int n,float*q,MessageInput&saved){
#ifdef SHOEBOX_LEARNED_COMPLETION
  return; // Already transformed into the exported constituent; do not add twice.
#else
  if(!n||m.message_config[0]==0)return;
  float weight_sum=0;
  if(r>=3)for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2])weight_sum+=m.slots[4*k+3]*m.slots[4*k+3];
  float scale=1.f/std::sqrt(r>=3?std::max(1.f,weight_sum):float(n));
  float raw[D]{};
  for(int i=0;i<n;i++){
    const auto&child=parts[i];int slot=r>=3?child.ref.role:-100-i;
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
    if(contribution_apply(m,CHECK_PAYLOAD,child.ref.region))continue;
#endif
    float weight=scale*m.regional_gain[child.ref.region]*(r>=3?m.slots[4*slot+3]:1.f);
    for(int j=0;j<D;j++){int c=message_coord(m,slot,j);raw[j]+=weight*message_sign(m,slot,j)*(child.code[c]-child.structure[c]);}
    if(weight==0||child.origin.region<2||child.origin.region>=N)continue;
    auto&e=saved.semantic[saved.semantic_count++];e.producer=child.origin;e.occurrence=child.ref;
    e.slot=slot;e.coefficient=weight;
    memcpy(e.features,child.symbol_features,sizeof(e.features));
    memcpy(e.derivative,child.symbol_derivative,sizeof(e.derivative));
  }
  for(int j=0;j<D;j++){
    float value=std::tanh(raw[j]);q[j]+=m.message_config[0]*value;
    saved.payload_derivative[j]=m.message_config[0]*(1-value*value);
  }
#endif
}
inline void message_feedback(Model&m,const float*raw,float squared,float*out,MessageInput&saved){
  float scale=1.f/std::sqrt(std::max(1.f,squared));
  for(int j=0;j<D;j++){out[j]=std::tanh(scale*raw[j]);saved.feedback_derivative[j]=scale*(1-out[j]*out[j]);}
}
inline void message_route_gradient(Model&m,int r,LaceRecord&v,const float*g,float*out){
  if(r<2||!m.composition.enabled||!message_enabled(m))return;
  auto*saved_input=lace_input(m,r,v);if(!saved_input)return;
  const auto&input=*saved_input;if(input.tick!=v.tick)return;
  float dg=0;for(int j=0;j<D;j++)dg+=g[j]*(1-v.h[j]*v.h[j])*v.residual[j];
  // This is the smooth response-gate derivative with selection/mask held fixed.
  // No derivative is claimed for the hard choice or top-k boundary itself.
  for(int j=0;j<D;j++)out[j]=dg*input.message.route_derivative[j];
}
inline void message_semantic_update(Model&m,int r,LaceRecord&v,InputCredit&input,
                                    const float*du,const float*dcue,float rate,int teaching=0){
#ifdef SHOEBOX_LEARNED_COMPLETION
  completion_semantic_update(m,r,v,input,du,dcue,rate,teaching);return;
#else
  const auto&saved=input.message;float dq[D];
  for(int j=0;j<D;j++)dq[j]=saved.qa*(saved.alpha*du[j]+(dcue?dcue[j]:0.f));
  float scale=rate*m.message_config[3]/norm(v.u);
  if(scale==0)return;
  for(int j=0;j<D;j++){
    float gradient=std::clamp(dq[j],-2.f,2.f)*input.symbol_derivative[j]*scale;
    for(int k=0;k<8;k++)m.composition.symbols[r-2][j][k]-=gradient*input.symbol_features[k];
  }
  m.composition.symbol_updates[r-2]++;
  for(int e=0;e<saved.semantic_count;e++){
    const auto&source=saved.semantic[e];int producer=source.producer.region;
    if(producer<2||producer>=N||m.frozen_region[producer]||m.teaching_scale[producer][0]==0)continue;
    float source_scale=m.teaching_scale[producer][0]==1.f?scale:scale*m.teaching_scale[producer][0];
    float gradient[D]{};
    for(int j=0;j<D;j++)gradient[message_coord(m,source.slot,j)]+=source.coefficient*message_sign(m,source.slot,j)*saved.payload_derivative[j]*dq[j];
    double norm2=0;
    for(int j=0;j<D;j++){
      float g=std::clamp(gradient[j],-2.f,2.f)*source.derivative[j];norm2+=double(g)*g;
      for(int k=0;k<8;k++)m.composition.symbols[producer-2][j][k]-=source_scale*g*source.features[k];
    }
    m.composition.symbol_updates[producer-2]++;
    auto&metric=m.composition.stream.message_metrics[producer];metric[0]++;metric[1]+=std::sqrt(norm2);
  }
#endif
}
inline void message_feedback_credit(Model&m,int r,LaceRecord&v,const float*du,const float*dcue,int depth){
  if(r<2||!m.composition.enabled||!message_feedback_enabled(m))return;
  auto*saved_input=lace_input(m,r,v);if(!saved_input)return;
  const auto&input=*saved_input;if(input.tick!=v.tick)return;
  const auto&saved=input.message;float df[D];
  for(int j=0;j<D;j++)df[j]=saved.feedback_derivative[j]*(saved.alpha*m.message_config[1]*du[j]+(dcue?m.message_config[2]*dcue[j]:0.f));
  for(int e=0;e<saved.feedback_count;e++){
    const auto&source=saved.feedback[e];
    if(source.region<0||source.region>=N||!source.tick)continue;
    auto*p=find_lace(m,source.region,source.tick,source.index);
    if(!p){m.composition.stream.message_metrics[r][3]++;continue;}
    for(int j=0;j<D;j++)p->credit[depth][message_coord(m,source.slot,j)]+=source.coefficient*message_sign(m,source.slot,j)*df[j];
    // The legacy context edge and the new feedback branch are two Jacobian
    // terms for the same consumed occurrence. Count that use only once, so
    // reuse normalization cannot dilute it merely because a channel was split.
    bool existing=false;
    for(int k=0;k<v.nparents;k++){const auto&e=v.parents[k];existing|=e.region==source.region&&e.tick==source.tick&&e.slot==source.slot;}
    if(!existing){p->uses[depth]++;m.laces.metrics[source.region][1]++;m.laces.metrics[source.region][8+depth]++;}
#ifdef SHOEBOX_LEARNED_COMPLETION
    neural_pending(m,source.region,p->tick,depth);
#endif
    m.composition.stream.message_metrics[source.region][2]++;
  }
}
