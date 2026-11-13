// Content experts share the regional parameter banks, but own a separate
// constituent-only input and saved routing eligibility. No recurrent h/u edge
// is invented for this transform.
inline ContentExpertRecord* content_expert_record(Model&m,SemanticRef ref){
  if(!ref.serial||ref.index<0||size_t(ref.index)>=m.content_records.size())return nullptr;
  auto&record=m.content_records[ref.index];
  return record.ref.serial==ref.serial&&record.ref.region==ref.region?&record:nullptr;
}
inline Model::ContentBankWorkspace& content_bank(Model&m,int r,int e){
  int index=r*E+e;auto&slot=m.content_banks[index];
  if(!slot)slot=std::make_unique<Model::ContentBankWorkspace>();
  if(!slot->active){
    slot->gradient=ContentExpertGradient{};slot->requests.clear();
    slot->active=true;m.content_dirty.push_back(index);
  }
  return *slot;
}
inline float content_input_norm(Model&m,const ContentExpertRecord&v){
  if(size_t(v.ref.index)>=m.content_input_cache.size())m.content_input_cache.resize(size_t(v.ref.index)+1);
  auto&cache=m.content_input_cache[v.ref.index];
  if(cache.serial!=v.ref.serial){cache.serial=v.ref.serial;cache.norm=norm(v.input);}
  return cache.norm;
}
// Tile the outer products by bank/row. Each matrix element sees the original
// request order; clipping, rates and saved gates have already been applied to
// each request separately. A bounded queue reuses storage even for huge DAGs.
struct ContentRows {Model::ContentBankWorkspace*bank;const float*inputs[64];};
inline void content_rows(void*context,int first,int last){
  auto&job=*static_cast<ContentRows*>(context);auto&bank=*job.bank;
  const auto&requests=bank.requests;const auto&inputs=job.inputs;
  for(int i=first;i<last;i++){
    float*row=bank.gradient.weights+i*D;size_t q=0;
    for(;q+4<=requests.size();q+=4){
      const auto&a=requests[q];const auto&b=requests[q+1];
      const auto&c=requests[q+2];const auto&d=requests[q+3];
      if((a.mask&b.mask&c.mask&d.mask)==~uint64_t(0)){
        float fa=a.factor[i],fb=b.factor[i],fc=c.factor[i],fd=d.factor[i];
        for(int j=0;j<D;j++){
          float value=row[j];value+=fa*inputs[q][j];value+=fb*inputs[q+1][j];
          value+=fc*inputs[q+2][j];value+=fd*inputs[q+3][j];row[j]=value;
        }
      }else for(size_t k=q;k<q+4;k++){
        float factor=requests[k].factor[i];uint64_t mask=requests[k].mask;
        while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;row[j]+=factor*inputs[k][j];}
      }
    }
    for(;q<requests.size();q++){
      float factor=requests[q].factor[i];
      if(requests[q].mask==~uint64_t(0))for(int j=0;j<D;j++)row[j]+=factor*inputs[q][j];
      else {uint64_t mask=requests[q].mask;
        while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;row[j]+=factor*inputs[q][j];}
      }
    }
  }
}
inline void content_materialize(Model&m,Model::ContentBankWorkspace&bank){
  if(bank.requests.empty())return;
  ContentRows job{&bank,{}};
  for(size_t q=0;q<bank.requests.size();q++)job.inputs[q]=m.content_records[bank.requests[q].ref.index].input;
  // Dispatch only a substantial ready tile; each worker owns complete rows.
  if(m.content_workers&&bank.requests.size()>=32)m.content_workers->run(content_rows,&job,D);
  else content_rows(&job,0,D);
  bank.requests.clear();
}
inline void content_expert_flush(Model&m){
  std::sort(m.content_dirty.begin(),m.content_dirty.end());
  for(int index:m.content_dirty){
    int r=index/E,e=index%E;auto&bank=*m.content_banks[index];
    content_materialize(m,bank);auto&g=bank.gradient;
    float*w=m.W+index*D*D,*b=m.bias+index*D;
    for(int j=0;j<D*D;j++)w[j]-=g.weights[j];
    for(int j=0;j<D;j++)b[j]-=g.bias[j];
    if(g.route){
      float*k=m.keys+index*D,n=1e-9f;
      for(int j=0;j<D;j++){k[j]-=g.keys[j];n+=k[j]*k[j];}
      n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;m.refresh_key(r,e);
    }
    bank.active=false;
  }
  m.content_dirty.clear();
}
struct ContentWorkerScope {
  Model&m;
  explicit ContentWorkerScope(Model&model):m(model){if(m.content_worker_depth++==0&&m.content_workers)m.content_workers->begin();}
  ~ContentWorkerScope(){if(--m.content_worker_depth==0&&m.content_workers)m.content_workers->end();}
};
struct ContentCreditScope {
  Model&m;
  ContentWorkerScope workers;
  explicit ContentCreditScope(Model&model):m(model),workers(model){m.content_credit_depth++;}
  ~ContentCreditScope(){if(--m.content_credit_depth==0)content_expert_flush(m);}
  ContentCreditScope(const ContentCreditScope&)=delete;
};
inline void content_expert_derivative(Model&m,const ContentExpertRecord&v,const float*local,float*dx){
  const float*w=m.W+(v.ref.region*E+v.expert)*D*D;
  float dg=0;for(int j=0;j<D;j++)dg+=local[j]*v.residual[j];
  for(int j=0;j<D;j++)dx[j]=dg*v.route_derivative[j];
  for(int i=0;i<D;i++)for(int j=0;j<D;j++)dx[j]+=v.gate*w[i*D+j]*local[i];
}
// Ready requests share a bank but retain separate reductions and saved gates.
// Each output coordinate accumulates rows in exactly the scalar VJP order.
inline void content_expert_derivatives(Model&m,const ContentExpertRecord*const*v,
                                       const float*const*local,float*const*dx,int count){
  int q=0;
  for(;q+4<=count;q+=4){
    for(int a=q;a<q+4;a++){
      float dg=0;for(int j=0;j<D;j++)dg+=local[a][j]*v[a]->residual[j];
      for(int j=0;j<D;j++)dx[a][j]=dg*v[a]->route_derivative[j];
    }
    const float*w=m.W+(v[q]->ref.region*E+v[q]->expert)*D*D;
    float g0=v[q]->gate,g1=v[q+1]->gate,g2=v[q+2]->gate,g3=v[q+3]->gate;
    for(int i=0;i<D;i++){
      float l0=local[q][i],l1=local[q+1][i],l2=local[q+2][i],l3=local[q+3][i];
      for(int j=0;j<D;j++){
        float weight=w[i*D+j];
        dx[q][j]+=g0*weight*l0;dx[q+1][j]+=g1*weight*l1;
        dx[q+2][j]+=g2*weight*l2;dx[q+3][j]+=g3*weight*l3;
      }
    }
  }
  for(;q<count;q++)content_expert_derivative(m,*v[q],local[q],dx[q]);
}
inline void content_expert_accumulate(Model&m,const ContentExpertRecord&v,const float*local,float rate,int teaching){
  int r=v.ref.region;if(rate==0||m.frozen_region[r])return;
  auto&bank=content_bank(m,r,v.expert);auto&gradient=bank.gradient;float a=rate/content_input_norm(m,v);
  if(r>=6&&r<=14&&m.utility.config[7]>0&&m.utility.config[7]!=1.f)a*=m.utility.config[7];
  if(bank.requests.size()==64)content_materialize(m,bank);
  bank.requests.emplace_back();auto&request=bank.requests.back();
  request.ref=v.ref;request.mask=m.lcfg[9]>=D?~uint64_t(0):v.input_mask;
  for(int i=0;i<D;i++){
    float d=std::clamp(local[i],-2.f,2.f);gradient.bias[i]+=a*d;
    request.factor[i]=a*d*v.gate;
  }
  m.content_expert_metrics[r][1]++;m.content_expert_metrics[r][teaching?2:3]++;
  if(m.lcfg[3]<=0||v.gate_derivative==0||v.expert==v.runner)return;
  float dg=0;for(int j=0;j<D;j++)dg+=local[j]*v.residual[j];
  float step=std::clamp(rate*m.task_route_lr_scale*m.lcfg[3]*dg*v.gate_derivative,-.05f,.05f);
  if(step==0)return;
  auto&runner=content_bank(m,r,v.runner).gradient;gradient.route=runner.route=true;
  for(int j=0;j<D;j++)if(v.input_mask&(uint64_t(1)<<j)){
    float value=step*v.input[j];gradient.keys[j]+=value;runner.keys[j]-=value;
  }
  m.content_expert_metrics[r][4]++;
}
