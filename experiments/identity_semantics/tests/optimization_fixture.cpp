// Independent scalar/full-scan oracles retained from a798c92.
#include "neural_fixture.cpp"
inline void reference_flush_credit(Model&m,int r,float lr){
  if(!m.lcfg[0])return;
  auto records=lace_records(m,r);
  if(m.composition.enabled&&r>=2){
    auto&z=m.composition.stream;
    // Newest to oldest: a bounded depth budget, no spatial expert update for
    // input-only state dependence, and no gradient through a fictitious h edge.
    for(auto it=records.rbegin();it!=records.rend();++it){
      auto&v=**it;auto*saved=lace_input(m,r,v);if(!saved)continue;
      auto&state=*saved;
      if(!v.tick||state.tick!=v.tick)continue;
      for(int d=0;d<int(m.lcfg[0]);d++)if(state.uses[d]){
        float gradient[D];float scale=1.f/std::pow(float(state.uses[d]),m.lcfg[5]);
        for(int j=0;j<D;j++)gradient[j]=state.gradient[d][j]*scale;
        memset(state.gradient[d],0,D*sizeof(float));state.uses[d]=0;
#ifdef SHOEBOX_IDENTITY_SEMANTICS
        semantic_input_update(m,r,v,gradient,lr*m.lcfg[1]*m.teaching_scale[r][0]);
#endif
        deliver_input_credit(m,r,v,gradient,d+1);z.metrics[r][9]++;
      }
    }
  }
  for(auto*record:records){
    auto&v=*record;
    if(!v.tick)continue;
    float total[D]{};bool any=false;
    for(int d=0;d<int(m.lcfg[0]);d++)if(v.uses[d]){
      float mean[D],scale=1.f/std::pow(float(v.uses[d]),m.lcfg[5]);
      for(int j=0;j<D;j++){mean[j]=v.credit[d][j]/v.uses[d];total[j]+=v.credit[d][j]*scale;}
      // Clear before propagation so delayed cycles cannot consume new credit
      // recursively in this same depth or bypass the explicit depth budget.
      memset(v.credit[d],0,D*sizeof(float));v.uses[d]=0;
      distribute_credit(m,r,v,mean,d+1);any=true;
    }
    if(any){
      double normg=0;for(float z:total)normg+=z*z;
      m.laces.metrics[r][2]++;m.laces.metrics[r][5]+=std::sqrt(normg);
      expert_update(m,r,total,lr*m.lcfg[1],&v);
    }
  }
}

extern "C" {
void optimization_vjp(void*p,int count,float*out){
  auto&m=*static_cast<Model*>(p);ContentExpertRecord records[16];float local[16][D];
  const ContentExpertRecord*values[16];const float*inputs[16];float*outputs[16];
  for(int q=0;q<count;q++){
    auto&v=records[q];v.ref.region=3;v.expert=2;v.gate=.6f+.07f*q;
    for(int j=0;j<D;j++){
      v.residual[j]=std::sin(float(q*19+j))*.3f;
      v.route_derivative[j]=std::cos(float(q*3+j))*.02f;
      local[q][j]=std::sin(float(q*5+j*7))*3.f;
    }
    values[q]=&v;inputs[q]=local[q];outputs[q]=out+(count+q)*D;
    content_expert_derivative(m,v,local[q],out+q*D);
  }
  content_expert_derivatives(m,values,inputs,outputs,count);
}
void optimization_gradients(void*p,int optimized,int sparse){
  auto&m=*static_cast<Model*>(p);m.lcfg[9]=sparse?16:D;
  m.content_records.resize(137);std::map<std::pair<int,int>,ContentExpertGradient> expected;
  ContentCreditScope scope(m);
  for(int q=0;q<137;q++){
    auto&v=m.content_records[q];v.ref={uint64_t(q+1),3,q};v.expert=2;v.runner=3;
    v.input_mask=sparse?0x8181818181818181ull:~uint64_t(0);
    v.gate=.7f+.03f*(q%7);v.gate_derivative=.4f;
    float local[D];float rate=q%2?.04f:.000001f;
    for(int j=0;j<D;j++){
      v.input[j]=std::sin(float(q*7+j))*.3f;v.residual[j]=std::cos(float(q*13+j))*.2f;
      local[j]=std::sin(float(q*5+j*3))*(q%3?8.f:.001f);
    }
    if(optimized){content_expert_accumulate(m,v,local,rate,q%2);continue;}
    auto&g=expected[{3,2}];float a=rate/norm(v.input);
    for(int i=0;i<D;i++){
      float d=std::clamp(local[i],-2.f,2.f);g.bias[i]+=a*d;
      for(int j=0;j<D;j++)if(m.lcfg[9]>=D||(v.input_mask&(uint64_t(1)<<j)))
        g.weights[i*D+j]+=a*d*v.gate*v.input[j];
    }
    float dg=0;for(int j=0;j<D;j++)dg+=local[j]*v.residual[j];
    float step=std::clamp(rate*m.task_route_lr_scale*m.lcfg[3]*dg*v.gate_derivative,-.05f,.05f);
    if(step!=0){auto&runner=expected[{3,3}];g.route=runner.route=true;
      for(int j=0;j<D;j++)if(v.input_mask&(uint64_t(1)<<j)){
        float value=step*v.input[j];g.keys[j]+=value;runner.keys[j]-=value;
      }
    }
  }
  if(!optimized)for(auto&entry:expected){
    int r=entry.first.first,e=entry.first.second;auto&g=entry.second;
    float*w=m.W+(r*E+e)*D*D,*b=m.bias+(r*E+e)*D;
    for(int j=0;j<D*D;j++)w[j]-=g.weights[j];
    for(int j=0;j<D;j++)b[j]-=g.bias[j];
    if(g.route){float*k=m.keys+(r*E+e)*D,n=1e-9f;
      for(int j=0;j<D;j++){k[j]-=g.keys[j];n+=k[j]*k[j];}
      n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;m.refresh_key(r,e);
    }
  }
}
void optimization_credit_order(void*p,int optimized,int frozen){
  auto&m=*static_cast<Model*>(p);reset_context(p);m.s.tick=100;m.lcfg[0]=3;m.frozen_region[3]=frozen;
  for(int k=0;k<6;k++)fixture_execute(m,3,2+k%3);
  for(int k=0;k<6;k++){
    auto*v=find_lace(m,3,101+k);auto&input=*lace_input(m,3,*v);
    v->nparents=2;
    v->parents[0]={3,-1,101+(k+2)%6,.2f,-1};
    v->parents[1]={3,-1,v->tick,.1f,-1};
    if(k){input.previous_tick=100+k;input.retention=.3f;}
    input.message.feedback_count=1;input.message.feedback[0]={3,m.starts[3],101+(k+4)%6,.1f,-1};
    input.message.alpha=.4f;
    for(int j=0;j<D;j++)input.message.feedback_derivative[j]=.7f;
    if(k==2||k==4){
      v->uses[0]=2;input.uses[0]=1;
      for(int j=0;j<D;j++){v->credit[0][j]=.01f*(j%5-2);input.gradient[0][j]=.02f*(j%3-1);}
    }
  }
  neural_invalidate(m);
  for(int pass=0;pass<3;pass++){
    if(optimized)flush_credit(m,3,.02f);else reference_flush_credit(m,3,.02f);
  }
}
void optimization_cache_status(void*p,uint64_t*out){
  auto&m=*static_cast<Model*>(p);out[0]=out[1]=0;
  for(auto&plan:m.constituent_plans)out[0]+=plan.root.serial!=0;
  for(auto&cache:m.content_input_cache)out[1]+=cache.serial!=0;
  out[2]=m.semantic_free.size();out[3]=m.neural_index_valid;
}
}
extern "C" void optimization_allocate(void*p,int count,uint64_t*out){
  auto&m=*static_cast<Model*>(p);auto&s=m.composition.stream;
  for(int i=0;i<count;i++){
    ContentEvent event{};event.ref.region=3;event.ref.id={s.semantic_serial+1,73};
    semantic_encode(m,3,event);constituent_encode(m,3,event,nullptr,0,false);
    s.history[3][0]=event;
  }
  out[0]=s.semantic_serial;out[1]=constituent_capacity(m);
  out[2]=m.semantic_free.size();out[3]=s.semantic_cursor;
}
