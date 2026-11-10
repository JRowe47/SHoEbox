inline float* state_parameter(Model&m,int r,int field){return m.state_parameters+(r*3+field)*D;}
inline float* prototype(Model&m,int r,int e){return m.state_parameters+6*D+(r*E+e)*D;}
inline float* task_offset(Model&m,int r,int e){return m.state_parameters+6*D+2*E*D+(r*E+e)*D;}
inline TemporalRecord* temporal_record(Model&m,int r,int64_t tick){
  if(tick<=0)return nullptr;
  auto&v=m.learned.records[r][tick&(TEMP_RING-1)];return v.tick==tick?&v:nullptr;
}
inline bool bottom_enabled(Model&m){return m.scfg[0]>0||m.scfg[8]>0||m.scfg[11]>0;}
inline void materialize_key(Model&m,int r,int e){
  float*p=prototype(m,r,e),*o=task_offset(m,r,e),*k=m.keys+(r*E+e)*D;
  float n=1e-12f;for(int j=0;j<D;j++){k[j]=p[j]+o[j];n+=k[j]*k[j];}
  n=1/std::sqrt(n);for(int j=0;j<D;j++)k[j]*=n;
}
inline void initialize_learned(Model&m){
  memset(&m.learned,0,sizeof(m.learned));
  if(!m.state_parameters)return;
  for(int r=0;r<2;r++)for(int e=0;e<E;e++){
    m.learned.mass[r][e]=m.scfg[12];
    for(int j=0;j<D;j++)m.learned.sums[r][e][j]=m.scfg[12]*prototype(m,r,e)[j];
  }
}
inline void publish_state(Model&m,int r,float response_gate,int runner=0){
  if(r>=2||!bottom_enabled(m))return;
  auto&s=m.s;auto&z=m.learned;auto&v=z.records[r][s.tick&(TEMP_RING-1)];
  memset(&v,0,sizeof(v));v.tick=s.tick;v.expert=s.chosen[r];v.generation=z.generation[r][v.expert];
  v.runner=runner;v.runner_generation=z.generation[r][runner];
  v.response_gate=response_gate;v.input_mask=feature_mask(s.u[r],int(m.scfg[7]));
  memcpy(v.u,s.u[r],D*sizeof(float));memcpy(v.h,s.h[r],D*sizeof(float));
  if(m.scfg[0]<=0)return;
  float*read=state_parameter(m,r,0),*wa=state_parameter(m,r,1),*wb=state_parameter(m,r,2);
  for(int j=0;j<D;j++){
    float old=z.memory[r][j],a=std::tanh(read[j]);
    float gate=1.f/(1.f+std::exp(-std::clamp(wa[j]*s.mem[r][j]+wb[j],-8.f,8.f)));
    v.before[j]=old;v.read[j]=m.scfg[2]*a;v.read_derivative[j]=m.scfg[2]*(1-a*a);
    v.eligibility_a[j]=z.eligibility_a[r][j];v.eligibility_b[j]=z.eligibility_b[r][j];v.write[j]=gate;
    // Exact diagonal retention eligibility with the expert candidate detached.
    // Cross-coordinate and hard-route effects are deliberately not a dense RTRL tape.
    float d=gate*(1-gate)*(s.h[r][j]-old);
    z.eligibility_a[r][j]=(1-gate)*z.eligibility_a[r][j]+d*s.mem[r][j];
    z.eligibility_b[r][j]=(1-gate)*z.eligibility_b[r][j]+d;
    z.memory[r][j]=(1-gate)*old+gate*s.h[r][j];
    z.metrics[r][7]+=std::abs(v.read[j]*old);z.metrics[r][8]+=gate;
  }
  z.metrics[r][9]++;
}
inline bool valid_bottom_publication(Model&m,int r,LaceRecord*v){
  if(r>=2||!v||!bottom_enabled(m))return true;
  auto*p=temporal_record(m,r,v->tick);
  return p&&p->expert==v->expert&&p->generation==m.learned.generation[r][v->expert];
}
inline void learn_bottom_state(Model&m,int r,LaceRecord*v,const float*g,float rate){
  if(r>=2||m.scfg[0]<=0||!v||v->tick%int(m.scfg[6]))return;
  auto*record=temporal_record(m,r,v->tick);if(!record)return;
  auto&z=m.learned;
  if(record->generation!=z.generation[r][record->expert]){z.metrics[r][2]++;return;}
  float pre[D],du[D];for(int i=0;i<D;i++)du[i]=pre[i]=g[i]*(1-v->h[i]*v->h[i]);
  float*w=m.W+(r*E+v->expert)*D*D;
  for(int i=0;i<D;i++)for(int j=0;j<D;j++)du[j]+=v->gate*w[i*D+j]*pre[i];
  float step=rate*m.scfg[1]/norm(v->u);
  float*read=state_parameter(m,r,0),*wa=state_parameter(m,r,1),*wb=state_parameter(m,r,2);
  for(int j=0;j<D;j++){
    float gradient=std::clamp(du[j],-4.f,4.f);
    read[j]=std::clamp(read[j]-step*gradient*record->before[j]*record->read_derivative[j],-3.f,3.f);
    float ds=gradient*record->read[j];
    wa[j]=std::clamp(wa[j]-step*ds*record->eligibility_a[j],-3.f,3.f);
    wb[j]=std::clamp(wb[j]-step*ds*record->eligibility_b[j],-7.f,2.f);
    z.metrics[r][6]+=std::abs(gradient*record->before[j]*record->read_derivative[j]);
  }
  z.metrics[r][0]++;
  if(m.scfg[5]<=0)return;
  // One sampled temporal lace per teaching event; six geometrically spaced ages.
  // This is bounded local credit through the retained-state path. It does not
  // differentiate through every intervening candidate or discrete route.
  const int ages[6]{1,2,4,8,16,32};int age=ages[(v->tick/int(m.scfg[6]))%6];
  auto*parent=temporal_record(m,r,v->tick-age);
  if(!parent||parent->generation!=z.generation[r][parent->expert]){z.metrics[r][2]++;return;}
  float retention[D];for(int j=0;j<D;j++)retention[j]=parent->write[j];
  for(int k=1;k<age;k++){
    auto*middle=temporal_record(m,r,v->tick-k);if(!middle){z.metrics[r][2]++;return;}
    for(int j=0;j<D;j++)retention[j]*=1-middle->write[j];
  }
  float a=rate*m.scfg[5]/norm(parent->u);
  float*pw=m.W+(r*E+parent->expert)*D*D,*pb=m.bias+(r*E+parent->expert)*D;
  for(int i=0;i<D;i++){
    float error=std::clamp(du[i]*record->read[i]*retention[i]*(1-parent->h[i]*parent->h[i]),-2.f,2.f);
    pb[i]-=a*error;
    if(error!=0)sparse_weight_update(pw+i*D,parent->u,a*error*parent->response_gate,parent->input_mask);
  }
  z.metrics[r][1]++;z.metrics[r][10]+=age;
}
inline void bottom_task_update(Model&m,int r,LaceRecord&v,const float*g,float rate){
  if(m.scfg[8]<=0||v.gate_derivative==0)return;
  if(!valid_bottom_publication(m,r,&v)){m.learned.metrics[r][2]++;return;}
  auto*saved=temporal_record(m,r,v.tick);
  if(!saved||saved->runner!=v.runner||saved->runner_generation!=m.learned.generation[r][v.runner]){m.learned.metrics[r][2]++;return;}
  float dg=0;for(int j=0;j<D;j++)dg+=g[j]*(1-v.h[j]*v.h[j])*v.residual[j];
  float step=std::clamp(rate*m.scfg[8]*dg*v.gate_derivative,-.025f,.025f);
  float*chosen=task_offset(m,r,v.expert),*runner=task_offset(m,r,v.runner);
  uint64_t mask=v.input_mask;while(mask){int j=__builtin_ctzll(mask);mask&=mask-1;float d=step*v.u[j];chosen[j]-=d;runner[j]+=d;}
  for(float*o:{chosen,runner}){
    float n=0;for(int j=0;j<D;j++)n+=o[j]*o[j];
    if(n>m.scfg[21]*m.scfg[21]){float a=m.scfg[21]/std::sqrt(n);for(int j=0;j<D;j++)o[j]*=a;}
  }
  auto&u=m.learned.utility[r][v.expert];u+=.01f*(std::clamp(-dg,-1.f,1.f)-u);
  materialize_key(m,r,v.expert);materialize_key(m,r,v.runner);m.learned.metrics[r][3]++;
}
inline bool revive_expert(Model&m,int r,int dead,int donor){
  if(dead==donor)return false;
  auto&z=m.learned;
  memcpy(m.W+(r*E+dead)*D*D,m.W+(r*E+donor)*D*D,D*D*sizeof(float));
  memcpy(m.bias+(r*E+dead)*D,m.bias+(r*E+donor)*D,D*sizeof(float));
  memcpy(task_offset(m,r,dead),task_offset(m,r,donor),D*sizeof(float));
  float*pa=prototype(m,r,donor),*pb=prototype(m,r,dead);float na=0,nb=0;
  for(int j=0;j<D;j++){float old=pa[j],d=(j&1)?-.01f:.01f;pa[j]=old*(1+d);pb[j]=old*(1-d);na+=pa[j]*pa[j];nb+=pb[j]*pb[j];}
  na=1/std::sqrt(na+1e-12f);nb=1/std::sqrt(nb+1e-12f);
  float mass=.5f*z.mass[r][donor];z.mass[r][dead]=z.mass[r][donor]=mass;
  for(int j=0;j<D;j++){pa[j]*=na;pb[j]*=nb;z.sums[r][donor][j]=pa[j]*mass;z.sums[r][dead][j]=pb[j]*mass;}
  z.last_decay[r][dead]=z.last_decay[r][donor]=z.decay_clock[r];
  z.last_used[r][dead]=z.events[r];z.utility[r][dead]=0;
  z.generation[r][dead]++; // Outstanding laces to the old occupant are rejected.
  materialize_key(m,r,dead);materialize_key(m,r,donor);z.metrics[r][5]++;return true;
}
inline void update_bottom_prototype(Model&m,int r,int best,uint64_t mask,float route){
  auto&z=m.learned;z.events[r]++;z.last_used[r][best]=z.events[r];
  float cue[D]{},n=0;for(int j=0;j<D;j++)if(mask&(uint64_t(1)<<j)){cue[j]=m.s.u[r][j];n+=cue[j]*cue[j];}
  n=1/std::sqrt(n+1e-6f);for(float&x:cue)x*=n;
  float*p=prototype(m,r,best);
  if(m.scfg[11]>0){
    double progress=std::min(1.,double(z.events[r])/m.scfg[14]);
    float target=m.scfg[13]+.5f*(m.scfg[12]-m.scfg[13])*(1+std::cos(3.141592653589793*progress));
    z.decay_clock[r]+=std::log1p(-1./(E*target));
    float decay=std::exp(z.decay_clock[r]-z.last_decay[r][best]);z.last_decay[r][best]=z.decay_clock[r];
    z.mass[r][best]=decay*z.mass[r][best]+1;
    for(int j=0;j<D;j++){z.sums[r][best][j]=decay*z.sums[r][best][j]+cue[j];p[j]=z.sums[r][best][j]/z.mass[r][best];}
    z.metrics[r][4]++;
  }else for(int j=0;j<D;j++)p[j]+=route*(cue[j]-p[j]);
  float pn=1e-9f;for(int j=0;j<D;j++)pn+=p[j]*p[j];pn=1/std::sqrt(pn);for(int j=0;j<D;j++)p[j]*=pn;
  materialize_key(m,r,best);
  if(m.scfg[11]<=0||m.scfg[15]<=0||z.events[r]<m.scfg[16]||z.events[r]%int(m.scfg[15]))return;
  // Maintenance is periodic. Lazy decay avoids touching every bank entry per byte.
  float total=0;int donor=0;float largest=-1;
  for(int e=0;e<E;e++){
    float decay=std::exp(z.decay_clock[r]-z.last_decay[r][e]);z.mass[r][e]*=decay;
    for(int j=0;j<D;j++)z.sums[r][e][j]*=decay;z.last_decay[r][e]=z.decay_clock[r];total+=z.mass[r][e];
    float score=z.mass[r][e]*(1+std::max(0.f,z.utility[r][e]));if(score>largest){largest=score;donor=e;}
  }
  for(int e=0;e<E;e++)if(e!=donor&&z.mass[r][e]<m.scfg[17]*total/E&&z.events[r]-z.last_used[r][e]>=m.scfg[18]&&z.utility[r][e]<=m.scfg[23]){
    revive_expert(m,r,e,donor);break; // At most one split per region per maintenance event.
  }
}
