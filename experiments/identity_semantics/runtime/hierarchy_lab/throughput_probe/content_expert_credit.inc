// Content experts share the regional parameter banks, but own a separate
// constituent-only input and saved routing eligibility. No recurrent h/u edge
// is invented for this transform.
inline ContentExpertRecord* content_expert_record(Model&m,SemanticRef ref){
  if(!ref.serial||ref.index<0||size_t(ref.index)>=m.content_records.size())return nullptr;
  auto&record=m.content_records[ref.index];
  return record.ref.serial==ref.serial&&record.ref.region==ref.region?&record:nullptr;
}
inline void content_expert_flush(Model&m){
  for(auto&entry:m.content_gradients){
    int r=entry.first.first,e=entry.first.second;auto&g=entry.second;
    float*w=m.W+(r*E+e)*D*D,*b=m.bias+(r*E+e)*D;
    for(int j=0;j<D*D;j++)w[j]-=g.weights[j];
    for(int j=0;j<D;j++)b[j]-=g.bias[j];
    if(g.route){
      float*k=m.keys+(r*E+e)*D,n=1e-9f;
      for(int j=0;j<D;j++){k[j]-=g.keys[j];n+=k[j]*k[j];}
      n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;m.refresh_key(r,e);
    }
  }
  m.content_gradients.clear();
}
struct ContentCreditScope {
  Model&m;
  explicit ContentCreditScope(Model&model):m(model){m.content_credit_depth++;}
  ~ContentCreditScope(){if(--m.content_credit_depth==0)content_expert_flush(m);}
  ContentCreditScope(const ContentCreditScope&)=delete;
};
inline void content_expert_derivative(Model&m,const ContentExpertRecord&v,const float*local,float*dx){
  const float*w=m.W+(v.ref.region*E+v.expert)*D*D;
  float dg=0;for(int j=0;j<D;j++)dg+=local[j]*v.residual[j];
  for(int j=0;j<D;j++)dx[j]=dg*v.route_derivative[j];
  for(int i=0;i<D;i++)for(int j=0;j<D;j++)dx[j]+=v.gate*w[i*D+j]*local[i];
}
inline void content_expert_accumulate(Model&m,const ContentExpertRecord&v,const float*local,float rate,int teaching){
  int r=v.ref.region;if(rate==0||m.frozen_region[r])return;
  auto&gradient=m.content_gradients[{r,v.expert}];float a=rate/norm(v.input);
  if(r>=6&&r<=14&&m.utility.config[7]>0&&m.utility.config[7]!=1.f)a*=m.utility.config[7];
  for(int i=0;i<D;i++){
    float d=std::clamp(local[i],-2.f,2.f);gradient.bias[i]+=a*d;
    for(int j=0;j<D;j++)if(m.lcfg[9]>=D||(v.input_mask&(uint64_t(1)<<j)))
      gradient.weights[i*D+j]+=a*d*v.gate*v.input[j];
  }
  m.content_expert_metrics[r][1]++;m.content_expert_metrics[r][teaching?2:3]++;
  if(m.lcfg[3]<=0||v.gate_derivative==0||v.expert==v.runner)return;
  float dg=0;for(int j=0;j<D;j++)dg+=local[j]*v.residual[j];
  float step=std::clamp(rate*m.task_route_lr_scale*m.lcfg[3]*dg*v.gate_derivative,-.05f,.05f);
  if(step==0)return;
  auto&runner=m.content_gradients[{r,v.runner}];gradient.route=runner.route=true;
  for(int j=0;j<D;j++)if(v.input_mask&(uint64_t(1)<<j)){
    float value=step*v.input[j];gradient.keys[j]+=value;runner.keys[j]-=value;
  }
  m.content_expert_metrics[r][4]++;
}
