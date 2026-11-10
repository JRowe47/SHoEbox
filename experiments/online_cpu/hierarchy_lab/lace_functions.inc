inline void sparse_weight_update(float*w,const float*u,float factor,uint64_t mask){
#ifdef __AVX512F__
  __m512 f=_mm512_set1_ps(factor);
  for(int j=0;j<D;j+=16){__mmask16 k=__mmask16(mask>>j);if(!k)continue;
    __m512 current=_mm512_loadu_ps(w+j),input=_mm512_loadu_ps(u+j);
    _mm512_storeu_ps(w+j,_mm512_mask3_fnmadd_ps(f,input,current,k));
  }
#else
  while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;w[j]-=factor*u[j];}
#endif
}
inline uint64_t feature_mask(const float*u,int k){
  if(k>=D)return ~uint64_t(0);
  int ids[D];for(int j=0;j<D;j++)ids[j]=j;
  std::nth_element(ids,ids+k,ids+D,[&](int a,int b){
    float x=std::abs(u[a]),y=std::abs(u[b]);return x==y?a<b:x>y;
  });
  uint64_t mask=0;for(int j=0;j<k;j++)mask|=uint64_t(1)<<ids[j];return mask;
}
inline LaceRecord* current_lace(Model&m,int r){
  auto&v=m.laces.records[r][m.laces.cursor[r]];return v.tick?&v:nullptr;
}
inline LaceRecord* find_lace(Model&m,int r,int64_t tick,int index=-1){
  if(!tick)return nullptr;
  if(index>=0&&index<LACE_RING&&m.laces.records[r][index].tick==tick)return &m.laces.records[r][index];
  for(int a=0;a<LACE_RING;a++){
    auto&v=m.laces.records[r][(m.laces.cursor[r]-a)&(LACE_RING-1)];
    if(v.tick==tick)return &v;
  }return nullptr;
}
inline void route_task_update(Model&m,int r,LaceRecord&v,const float*g,float rate){
  if(m.lcfg[3]<=0||v.gate_derivative==0)return;
  float dg=0;for(int j=0;j<D;j++)dg+=g[j]*(1-v.h[j]*v.h[j])*v.residual[j];
  float step=std::clamp(rate*m.lcfg[3]*dg*v.gate_derivative,-.05f,.05f);
  float *chosen=m.keys+(r*E+v.expert)*D,*runner=m.keys+(r*E+v.runner)*D;
  uint64_t mask=v.input_mask;
  while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;float d=step*v.u[j];chosen[j]-=d;runner[j]+=d;}
  for(float*k:{chosen,runner}){float n=1e-9f;for(int j=0;j<D;j++)n+=k[j]*k[j];n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;}
  m.laces.metrics[r][4]++;
}
inline void distribute_credit(Model&m,int r,LaceRecord&v,const float*g,int depth){
  if(depth>=int(m.lcfg[0]))return;
  if(depth>=2&&m.lcfg[10]>1&&(v.tick/m.cadence[r])%int(m.lcfg[10]))return;
  float pre[D],du[D];
  bool sparse=m.lcfg[9]<D;
  for(int i=0;i<D;i++){pre[i]=g[i]*(1-v.h[i]*v.h[i]);du[i]=(!sparse||(v.input_mask&(uint64_t(1)<<i)))?pre[i]:0;}
  float*w=m.W+(r*E+v.expert)*D*D;
  if(sparse){
    // Input SDR coordinates admit synapses for both upstream task credit and
    // plasticity. The forward expert still receives the complete payload.
#ifdef __AVX512F__
    __m512 gate=_mm512_set1_ps(v.gate);
    for(int i=0;i<D;i++)if(pre[i]!=0){__m512 error=_mm512_set1_ps(pre[i]);
      for(int j=0;j<D;j+=16){__mmask16 k=__mmask16(v.input_mask>>j);if(!k)continue;
        __m512 scaled=_mm512_mul_ps(gate,_mm512_loadu_ps(w+i*D+j));
        __m512 current=_mm512_loadu_ps(du+j);
        _mm512_storeu_ps(du+j,_mm512_mask3_fmadd_ps(scaled,error,current,k));
      }
    }
#else
    int ids[D],count=0;uint64_t mask=v.input_mask;
    while(mask){ids[count++]=__builtin_ctzll(mask);mask&=mask-1;}
    for(int i=0;i<D;i++)if(pre[i]!=0)for(int a=0;a<count;a++){int j=ids[a];du[j]+=v.gate*w[i*D+j]*pre[i];}
#endif
  }else for(int i=0;i<D;i++)for(int j=0;j<D;j++)du[j]+=v.gate*w[i*D+j]*pre[i];
  for(int k=0;k<v.nparents;k++){
    const auto&edge=v.parents[k];
    // A1 has a direct byte loss; avoid duplicating its dominant shortcut.
    if(edge.region==0)continue;
    LaceRecord*p=find_lace(m,edge.region,edge.tick,edge.index);
    if(!p){m.laces.metrics[r][3]++;continue;}
    if(edge.slot<0){for(int j=0;j<D;j++)p->credit[depth][j]+=edge.coefficient*du[j];}
    else for(int j=0;j<D;j++)p->credit[depth][m.perm[edge.slot*D+j]]+=edge.coefficient*m.signs[edge.slot*D+j]*du[j];
    p->uses[depth]++;m.laces.metrics[edge.region][1]++;m.laces.metrics[edge.region][8+depth]++;
  }
}
// Apply accumulated downstream teaching only when this region next runs.
// Local Jacobians use current weights and saved publication activations:
// this is bounded delayed local credit, not an exact temporal gradient.
inline void flush_credit(Model&m,int r,float lr){
  if(!m.lcfg[0])return;
  for(int a=LACE_RING-1;a>=0;a--){
    auto&v=m.laces.records[r][(m.laces.cursor[r]-a)&(LACE_RING-1)];
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
