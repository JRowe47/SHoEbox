// Scalar request-level oracle: deliberately repeats handles across former
// 64-request tile boundaries, banks, masks and nested/outer credit scopes.
#include "optimization_fixture.cpp"
#include <cassert>
extern "C" void factor_fixture(void*p,int optimized,int scopes,int sparse,int frozen){
  auto&m=*static_cast<Model*>(p);m.content_records.resize(145);m.content_input_cache.clear();
  m.lcfg[3]=.1f;m.utility.config[7]=.3f;m.frozen_region[7]=frozen;
  for(int q=0;q<145;q++){
    auto&v=m.content_records[q];v=ContentExpertRecord{};
    v.ref={uint64_t(q+1),q%2?3:7,q};v.expert=1+q%3;v.runner=q%13?v.expert+1:v.expert;
    v.input_mask=q%5?0x8181818181818181ull:0;
    v.gate=.7f+.03f*(q%7);v.gate_derivative=q%17?.4f:0.f;
    for(int j=0;j<D;j++){
      v.input[j]=std::sin(float(q*7+j))*.3f;v.residual[j]=std::cos(float(q*13+j))*.2f;
    }
  }
  for(int scope=0;scope<scopes;scope++){
    ContentCreditScope outer(m);
    std::map<std::pair<int,int>,ContentExpertGradient> expected;
    for(int pass=0;pass<5;pass++){
      ContentCreditScope nested(m);
      m.lcfg[9]=sparse&&pass%2?16:D;
      for(int q=0;q<145;q++){
        auto&v=m.content_records[q];int r=v.ref.region;
        float rate=q%11?(pass%2?.04f:.000001f):0.f,local[D];
        for(int j=0;j<D;j++)local[j]=std::sin(float(q*5+j*3+pass*7))*(pass%3?800.f:.001f);
        if(optimized){content_expert_accumulate(m,v,local,rate,pass%2);continue;}
        if(rate==0||m.frozen_region[r])continue;
        auto&g=expected[{r,v.expert}];float a=rate/norm(v.input);
        if(r>=6&&r<=14&&m.utility.config[7]>0&&m.utility.config[7]!=1.f)a*=m.utility.config[7];
        for(int i=0;i<D;i++){
          float d=std::clamp(local[i],-2.f,2.f);g.bias[i]+=a*d;
          for(int j=0;j<D;j++)if(m.lcfg[9]>=D||(v.input_mask&(uint64_t(1)<<j)))
            g.weights[i*D+j]+=a*d*v.gate*v.input[j];
        }
        if(m.lcfg[3]<=0||v.gate_derivative==0||v.expert==v.runner)continue;
        float dg=0;for(int j=0;j<D;j++)dg+=local[j]*v.residual[j];
        float step=std::clamp(rate*m.task_route_lr_scale*m.lcfg[3]*dg*v.gate_derivative,-.05f,.05f);
        if(step!=0){auto&runner=expected[{r,v.runner}];g.route=runner.route=true;
          for(int j=0;j<D;j++)if(v.input_mask&(uint64_t(1)<<j)){
            float value=step*v.input[j];g.keys[j]+=value;runner.keys[j]-=value;
          }
        }
      }
      if(optimized)assert(m.content_factor_metrics[4]==uint64_t(scope));
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
  assert(m.content_credit_depth==0&&m.content_dirty.empty());
  for(auto&bank:m.content_banks)if(bank){
    assert(!bank->active&&bank->requests.empty()&&bank->routes.empty());
  }
}

extern "C" void factor_cancellation(void*p){
  auto&m=*static_cast<Model*>(p);m.content_records.resize(1);m.content_input_cache.clear();auto&v=m.content_records[0];
  v=ContentExpertRecord{};v.ref={1,3,0};v.expert=2;v.runner=3;v.gate=1;v.gate_derivative=1;
  v.input_mask=~uint64_t(0);v.input[0]=1;v.residual[0]=1;m.lcfg[3]=1;m.lcfg[9]=D;
  float before=m.W[(3*E+2)*D*D],local[D]{};
  {ContentCreditScope scope(m);
    local[0]=100;content_expert_accumulate(m,v,local,.04f,0);
    local[0]=-100;content_expert_accumulate(m,v,local,.04f,1);
    assert(m.W[(3*E+2)*D*D]==before);
    assert(m.content_banks[3*E+2]->gradient.route&&m.content_banks[3*E+3]->gradient.route);
  }
  assert(m.W[(3*E+2)*D*D]==before);
  assert(m.content_factor_metrics[0]==2&&m.content_factor_metrics[1]==1);
  assert(m.content_factor_metrics[2]==4&&m.content_factor_metrics[3]==2);
  // A recycled slot has a new handle and must start a fresh factor and norm.
  v.ref.serial=2;v.input[0]=2;
  {ContentCreditScope scope(m);local[0]=3;content_expert_accumulate(m,v,local,.04f,0);}
  assert(m.content_input_cache[0].serial==2);
  assert(m.content_factor_metrics[1]==2&&m.content_factor_metrics[4]==2);
}
