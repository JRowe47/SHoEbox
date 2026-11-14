// Fixed orthonormal signed partitions: O(D) pooling and its exact transpose.
// These are response features, never content identities or directional teachers.
inline float prediction_sign(int j,int rank){return ((j/rank)&1)?-1.f:1.f;}
inline void prediction_pool(const float*input,float*out,int rank){
  memset(out,0,D*sizeof(float));float scale=std::sqrt(float(rank)/D);
  for(int j=0;j<D;j++)out[j%rank]+=scale*prediction_sign(j,rank)*input[j];
}
inline void prediction_expand(const float*input,float*out,int rank){
  float scale=std::sqrt(float(rank)/D);
  for(int j=0;j<D;j++)out[j]=scale*prediction_sign(j,rank)*input[j%rank];
}
inline int prediction_region(int r){return r==17?13:r>=2&&r<=14?r-2:-1;}
inline const CompactFeatureView& prediction_view(Model&m,int r,const LaceRecord&v){
  auto&z=m.prediction;auto&view=z.stream.current[r];
  // A publication remains immutable even if its producing bank later learns.
  if(view.tick==v.tick)return view;
  view=CompactFeatureView{};view.tick=v.tick;
  const auto&event=m.composition.stream.history[r][0];
  view.publication=event.ref.seq;view.semantic=event.semantic;
  view.bank_version=m.bank_pipeline.enabled?m.bank_pipeline.version:m.reverse_version;
  view.response_version=z.response_version[r];view.forecast_version=z.forecast_version[r];
  prediction_pool(v.h,view.response,z.rank);
  prediction_pool(event.values().code,view.content,z.rank);
  if(auto*saved=context_record(m,r,v))memcpy(view.context,saved->reads(),saved->heads()*MK_DIM*sizeof(float));
  return view;
}
inline void prediction_forecast(Model&m,int r,const LaceRecord&v,float*out){
  auto&z=m.prediction;const auto&view=prediction_view(m,r,v);
  z.stream.pending[r]=view;z.stream.pending[r].forecast_version=z.forecast_version[r];
  const float*w=z.forecast.data()+prediction_region(r)*D*z.rank;
  for(int i=0;i<D;i++){
    float value=m.pbias[r*D+i];
    for(int k=0;k<z.rank;k++)value+=w[i*z.rank+k]*view.response[k];
    out[i]=value;
  }
}
inline void prediction_gradient(Model&m,int r,const float*error,float*out){
  auto&z=m.prediction;float compact[D]{};
  const float*w=z.forecast.data()+prediction_region(r)*D*z.rank;
  for(int i=0;i<D;i++)for(int k=0;k<z.rank;k++)compact[k]+=w[i*z.rank+k]*error[i];
  prediction_expand(compact,out,z.rank);
}
inline void prediction_update(Model&m,int r,const float*error,float rate){
  auto&z=m.prediction;const auto&view=z.stream.pending[r];float denominator=1.f;
  for(int k=0;k<z.rank;k++)denominator+=view.response[k]*view.response[k];
  float a=rate/denominator;if(a==0)return;
  float*w=z.forecast.data()+prediction_region(r)*D*z.rank;
  for(int i=0;i<D;i++){
    m.pbias[r*D+i]-=a*error[i];
    for(int k=0;k<z.rank;k++)w[i*z.rank+k]-=a*error[i]*view.response[k];
  }
  ++z.forecast_version[r];
}
inline void prediction_slow_flush(Model&m){
  auto&z=m.prediction;auto&s=z.stream;if(!s.samples)return;
  auto*v=current_lace(m,17);
  // The current Y lace is pinned by the ordinary memory/history ownership.
  // Training samples accumulate against that one immutable slow publication.
  if(v&&v->tick==s.slow_tick&&s.rate!=0){
    float pooled[D],gradient[D];prediction_pool(s.gradient,pooled,z.rank);
    prediction_expand(pooled,gradient,z.rank);
    for(int j=0;j<D;j++)gradient[j]*=z.slow_scale/float(s.samples);
    if(m.lcfg[0])distribute_credit(m,17,*v,gradient,0);
    expert_update(m,17,gradient,s.rate,v);
  }
  s.samples=0;memset(s.gradient,0,sizeof(s.gradient));s.rate=0;s.mode=0;
}
inline bool prediction_slow_ready(const Model&m){
  const auto&s=m.prediction.stream;
  return !s.slow_tick||m.composition.stream.byte_position-s.slow_byte>=m.prediction.stride;
}
inline void prediction_begin_step(Model&m,int mode,float rate){
  auto&s=m.prediction.stream;
  // A change in teaching mode/rate closes the preceding learning interval.
  // Frozen copies have no pending samples; Python rejects an unsafe in-place
  // transition to frozen evaluation until finish_prediction() is called.
  if(mode&&s.samples&&(s.rate!=rate||s.mode!=mode)){
    prediction_slow_flush(m);s.slow_tick=0;
  }
}
inline void prediction_slow_publish(Model&m){
  auto*v=current_lace(m,17);if(!v)return;auto&z=m.prediction;auto&s=z.stream;
  const auto&view=prediction_view(m,17,*v);
  prediction_expand(view.response,s.slow,z.rank);
  s.slow_tick=v->tick;s.slow_byte=m.composition.stream.byte_position;
}
inline void prediction_fast_forward(Model&m,int byte,const float*const*snapshot,const int*valid){
  auto&z=m.prediction;auto&s=z.stream;memset(s.feedback,0,sizeof(s.feedback));
  // Required feedback always reads the previous-byte snapshot, even between
  // slow boundaries. The current recipe has none; custom recipes retain it.
  for(int k=m.starts[17];k<m.starts[18];k++)if(m.slots[4*k+2]){
    int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]);if(valid[src]<=age)continue;
    if((m.s.ablation==1&&src>=2&&src!=17)||(m.s.ablation>=10&&src==m.s.ablation-10))continue;
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
    if(contribution_apply(m,CHECK_SOURCE,src)||contribution_apply(m,CHECK_FEEDBACK,src))continue;
#endif
    float gain=m.slots[4*k+3]*m.regional_gain[src];
    for(int j=0;j<D;j++)s.feedback[j]+=gain*m.signs[k*D+j]*snapshot[src][m.perm[k*D+j]];
  }
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  // Whole-source interventions include the new direct A1 correction branch.
  const bool mute_a1=contribution_apply(m,CHECK_SOURCE,0);
  for(int j=0;j<D;j++)s.output[j]=std::tanh(s.slow[j]+z.fast[0][j]+
    z.fast[1][j]*m.emb[byte*D+j]+z.fast[2][j]*(mute_a1?0.f:m.s.h[0][j])+z.fast[3][j]*s.feedback[j]);
#else
  for(int j=0;j<D;j++)s.output[j]=std::tanh(s.slow[j]+z.fast[0][j]+
    z.fast[1][j]*m.emb[byte*D+j]+z.fast[2][j]*m.s.h[0][j]+z.fast[3][j]*s.feedback[j]);
#endif
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  if(contribution_apply(m,CHECK_NEURAL,17))memset(s.output,0,sizeof(s.output));
#endif
}
inline void prediction_fast_teach(Model&m,int byte,float*direct,int mode,float rate,
                                  const int64_t*ticks,const int*regions,const int*indices){
  if(mode==4||mode==5)return;
  auto&z=m.prediction;auto&s=z.stream;float pre[D],feedback_gradient[D];
  float denominator=1.f;
  for(int j=0;j<D;j++)denominator+=m.emb[byte*D+j]*m.emb[byte*D+j]+m.s.h[0][j]*m.s.h[0][j]+s.feedback[j]*s.feedback[j];
  for(int j=0;j<D;j++){
    pre[j]=direct[D+j]*(1-s.output[j]*s.output[j]);
    direct[j]+=z.fast[2][j]*pre[j];feedback_gradient[j]=z.fast[3][j]*pre[j];
  }
  // Preserve the declared lace sampling clock for fast feedback as well.
  if(m.lcfg[0]&&m.s.tick%int(m.lcfg[7])==0)for(int k=m.starts[17];k<m.starts[18];k++)if(m.slots[4*k+2]){
    int src=int(m.slots[4*k]);if(!ticks[src])continue;
    if((m.s.ablation==1&&src>=2&&src!=17)||(m.s.ablation>=10&&src==m.s.ablation-10))continue;
    auto*v=find_lace(m,regions[src],ticks[src],indices[src]);if(!v)continue;
    float*pending=v->credit[0],gain=m.slots[4*k+3]*m.regional_gain[src];
    for(int j=0;j<D;j++)pending[m.perm[k*D+j]]+=gain*m.signs[k*D+j]*feedback_gradient[j];
    ++v->uses[0];neural_pending(m,regions[src],v->tick,0);
  }
  if(s.slow_tick){
    if(!s.samples){s.rate=rate;s.mode=mode;}
    ++s.samples;for(int j=0;j<D;j++)s.gradient[j]+=pre[j];
  }
  if(m.frozen_region[17])return;
  float a=rate*m.teaching_scale[17][0]/denominator;
  for(int j=0;j<D;j++){
    float step=a*std::clamp(pre[j],-2.f,2.f);
    z.fast[0][j]-=step;z.fast[1][j]-=step*m.emb[byte*D+j];
    z.fast[2][j]-=step*m.s.h[0][j];z.fast[3][j]-=step*s.feedback[j];
  }
}
inline void prediction_reset_stream(Model&m){m.prediction.stream=PredictionStream{};}
