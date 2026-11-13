// A local forecast owns its learning rate and objective. Finish its bounded
// reverse pass synchronously: never enqueue it in the language/DFA buffers,
// whose eventual flush has a different rate and teaching policy. No work from
// this pass survives an API call, so the retained-event checkpoint ABI is intact.
using LocalKey=std::pair<int,int64_t>;
struct LocalSignal {
  std::array<float,D> neural{},input{};
  int neural_uses=0,input_uses=0;
};
struct LocalCredit {
  std::map<LocalKey,LocalSignal> pending[LACE_DEPTH],updates;
};
inline void local_deliver_input(Model&,int,LaceRecord&,const float*,const float*,int,LocalCredit&);
inline void local_transform(Model&m,const LaceParent&edge,const float*gradient,float*out){
  if(edge.slot<=-100){
    int position=-100-edge.slot;
    for(int j=0;j<D;j++)out[content_coord(j,position)]+=edge.coefficient*content_sign(j,position)*gradient[j];
  }else if(edge.slot<0){
    for(int j=0;j<D;j++)out[j]+=edge.coefficient*gradient[j];
  }else for(int j=0;j<D;j++)out[m.perm[edge.slot*D+j]]+=edge.coefficient*m.signs[edge.slot*D+j]*gradient[j];
}
inline void local_deliver_input(Model&m,int r,LaceRecord&v,const float*du,const float*dcue,int depth,LocalCredit&work){
  if(depth>=std::clamp(int(m.lcfg[0]),0,LACE_DEPTH))return;
  // Coalesce the context and feedback Jacobians of each consumed role before
  // delivery. Splitting a channel must not increase its reuse denominator.
  auto deliver=[&](const LaceParent&edge,const float*gradient,int uses){
    auto*p=find_lace(m,edge.region,edge.tick,edge.index);
    if(!p){m.local_credit_metrics[r][7]++;return;}
    if(p->expert<0){
      local_deliver_input(m,edge.region,*p,gradient,nullptr,depth,work);
      m.local_credit_metrics[edge.region][6]++;return;
    }
    auto&target=work.pending[depth][{edge.region,edge.tick}];
    for(int j=0;j<D;j++)target.neural[j]+=gradient[j];
    target.neural_uses+=uses;m.local_credit_metrics[edge.region][1]+=uses;
  };
  auto*saved=lace_input(m,r,v);
  const MessageInput*message=saved&&saved->tick==v.tick&&r>=2&&message_feedback_enabled(m)?&saved->message:nullptr;
  float feedback[D]{};
  if(message)for(int j=0;j<D;j++)feedback[j]=message->feedback_derivative[j]*(message->alpha*m.message_config[1]*du[j]+(dcue?m.message_config[2]*dcue[j]:0.f));
  bool matched[COMP_MAX]{};
  for(int k=0;k<v.nparents;k++){
    const auto&edge=v.parents[k];float gradient[D]{};local_transform(m,edge,du,gradient);
    if(message)for(int e=0;e<message->feedback_count;e++){
      const auto&source=message->feedback[e];
      if(!matched[e]&&edge.region==source.region&&edge.tick==source.tick&&edge.slot==source.slot){
        local_transform(m,source,feedback,gradient);matched[e]=true;
        m.local_credit_metrics[source.region][2]++;
      }
    }
    deliver(edge,gradient,1);
  }
  if(message)for(int e=0;e<message->feedback_count;e++)if(!matched[e]){
    const auto&source=message->feedback[e];
    if(source.region<0||source.region>=N||!source.tick)continue;
    float gradient[D]{};local_transform(m,source,feedback,gradient);
    deliver(source,gradient,1);m.local_credit_metrics[source.region][2]++;
  }
  // Retained u is an input dependency, not the old expert's output h.
  if(r>=2&&saved&&saved->tick==v.tick&&saved->previous_tick&&saved->retention!=0){
    auto*previous=find_lace(m,r,saved->previous_tick,saved->previous_index);
    auto*input=previous?lace_input(m,r,*previous):nullptr;
    if(!input||input->tick!=previous->tick){m.local_credit_metrics[r][7]++;return;}
    auto&target=work.pending[depth][{r,previous->tick}];
    for(int j=0;j<D;j++)target.input[j]+=saved->retention*du[j];
    target.input_uses++;m.local_credit_metrics[r][3]++;
  }
}
inline void local_distribute(Model&m,int r,LaceRecord&v,const float*g,int depth,LocalCredit&work){
  if(depth>=std::clamp(int(m.lcfg[0]),0,LACE_DEPTH))return;
  if(v.expert<0){local_deliver_input(m,r,v,g,nullptr,depth,work);return;}
  float du[D],dcue[D]{};
  neural_input_gradient(m,r,v,g,du);message_route_gradient(m,r,v,g,dcue);
  local_deliver_input(m,r,v,du,dcue,depth,work);
}
inline void local_prediction_update(Model&m,int r,LaceRecord&v,const float*g,float rate){
  const float issuer_scale=m.teaching_scale[r][1];
  if(rate==0||issuer_scale==0||m.frozen_region[r])return;
  m.local_credit_metrics[r][0]++;
  LocalCredit work;float upstream[D];
  for(int j=0;j<D;j++)upstream[j]=issuer_scale*g[j];
  local_distribute(m,r,v,upstream,0,work);
  // Match lace reuse policy: mean derivative for onward neural propagation,
  // uses^-reuse_exponent for parameter updates and retained-input propagation.
  // Frozen recipients still transmit derivatives to their causal ancestors.
  for(int d=0;d<std::clamp(int(m.lcfg[0]),0,LACE_DEPTH);d++){
    for(auto&entry:work.pending[d]){
      int source=entry.first.first;auto*producer=find_lace(m,source,entry.first.second);
      if(!producer){m.local_credit_metrics[source][7]++;continue;}
      auto&signal=entry.second;auto&total=work.updates[entry.first];
      if(signal.neural_uses){
        float mean[D],scale=1.f/std::pow(float(signal.neural_uses),m.lcfg[5]);
        for(int j=0;j<D;j++){
          mean[j]=signal.neural[j]/signal.neural_uses;total.neural[j]+=signal.neural[j]*scale;
        }
        total.neural_uses++;local_distribute(m,source,*producer,mean,d+1,work);
      }
      if(signal.input_uses){
        float gradient[D],scale=1.f/std::pow(float(signal.input_uses),m.lcfg[5]);
        for(int j=0;j<D;j++)total.input[j]+=(gradient[j]=signal.input[j]*scale);
        total.input_uses++;local_deliver_input(m,source,*producer,gradient,nullptr,d+1,work);
      }
    }
  }
  // Calculate semantic derivatives before any expert/router mutation too.
  // This replaces expert_update's semantic side effect exactly once; neural
  // transport itself never adds a second copy of the constituent-code branch.
  semantic_expert_update(m,r,&v,g,rate*issuer_scale,1);
  for(auto&entry:work.updates){
    int source=entry.first.first;auto*producer=find_lace(m,source,entry.first.second);
    auto&total=entry.second;float scale=rate*m.lcfg[1]*m.teaching_scale[source][1];
    if(total.neural_uses)semantic_expert_update(m,source,producer,total.neural.data(),scale,1);
    if(total.input_uses)semantic_input_update(m,source,*producer,total.input.data(),scale,nullptr,1);
  }
  expert_update(m,r,g,rate,&v,1,false);
  for(auto&entry:work.updates){
    int source=entry.first.first;auto*producer=find_lace(m,source,entry.first.second);auto&total=entry.second;
    if(total.neural_uses){
      expert_update(m,source,total.neural.data(),rate*m.lcfg[1],producer,1,false);
      if(!m.frozen_region[source]&&m.teaching_scale[source][1]!=0&&m.lcfg[1]!=0)m.local_credit_metrics[source][4]++;
    }
    if(total.input_uses)m.local_credit_metrics[source][5]++;
  }
}
