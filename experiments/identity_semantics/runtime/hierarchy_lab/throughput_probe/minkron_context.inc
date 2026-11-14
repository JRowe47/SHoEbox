inline bool context_enabled(const Model&m,int r){return r>=2&&m.event.heads>0&&m.composition.enabled;}
// Fixed signed partitions reuse the 64 input features. No dense Q/K/V maps.
inline int context_coordinate(int head,int a,int b){return head?((a+8*b)*17+11)%D:a*8+b;}
inline float context_sign(int head,int a,int b){return ((a+b+head)%3==0)?-1.f:1.f;}
inline ContextEligibility* context_record(Model&m,int r,const LaceRecord&v){
  for(int i=0;i<LACE_RING;i++)if(m.event.context[r][i].tick==v.tick)return &m.event.context[r][i];
  auto it=m.event.archive[r].find(v.tick);return it==m.event.archive[r].end()?nullptr:&it->second;
}
inline void context_forward(Model&m,int r,const float*u,float*out,ContextEligibility&saved){
  saved=ContextEligibility{};saved.tick=m.s.tick;
  const int heads=m.event.heads;const float scale=1/std::sqrt(float(heads));
  for(int h=0;h<heads;h++){
    float key[MK_DIM],value[MK_DIM];float normk=1.f;
    auto&memory=m.event.memory[r][h];memcpy(saved.before[h],memory,sizeof(memory));
    for(int a=0;a<MK_DIM;a++){
      float q=0,v=0;
      for(int b=0;b<8;b++){
        int j=context_coordinate(h,a,b);float sign=context_sign(h,a,b);
        q+=sign*u[j]*.3535533905932738f;
        v+=sign*u[(j+19)%D]*.3535533905932738f;
      }
      key[a]=std::tanh(q);value[a]=std::tanh(v);normk+=key[a]*key[a];
      saved.query_derivative[h][a]=1-key[a]*key[a];
    }
    for(int b=0;b<MK_DIM;b++){
      float read=0;for(int a=0;a<MK_DIM;a++)read+=memory[a][b]*key[a];
      saved.read[h*MK_DIM+b]=scale*read;
      // Read before write: the response sees prior event memory. The current
      // observed input writes a normalized delta for subsequent publications.
      float error=value[b]-.98f*read;
      for(int a=0;a<MK_DIM;a++)memory[a][b]=.98f*memory[a][b]+.5f*key[a]*error/normk;
    }
  }
  for(int i=0;i<D;i++){
    float z=m.event.parameters.bias[r][i];
    for(int k=0;k<heads*MK_DIM;k++)z+=m.event.parameters.readout[r][i][k]*saved.read[k];
    out[i]=z;
  }
  SHOEBOX_DIAGNOSTIC(m.event.metrics[r][0]++;)
}
// Exact local VJP through the saved read and fixed pooling. Older memory
// influence is represented by the recurrence, not an invented temporal tape.
inline bool context_input_gradient(Model&m,int r,const LaceRecord&v,const float*g,float*du){
  if(!context_enabled(m,r))return false;
  auto*saved=context_record(m,r,v);
  if(!saved){SHOEBOX_DIAGNOSTIC(m.event.metrics[r][9]++;)memset(du,0,D*sizeof(float));return true;}
  float dr[MK_FEATURES]{};int heads=m.event.heads;
  for(int i=0;i<D;i++){
    float d=g[i]*(1-v.h[i]*v.h[i]);du[i]=d;
    for(int k=0;k<heads*MK_DIM;k++)dr[k]+=m.event.parameters.readout[r][i][k]*d;
  }
  float scale=1/std::sqrt(float(heads));
  for(int h=0;h<heads;h++)for(int a=0;a<MK_DIM;a++){
    float dq=0;for(int b=0;b<MK_DIM;b++)dq+=saved->before[h][a][b]*dr[h*MK_DIM+b]*scale;
    dq*=saved->query_derivative[h][a];
    for(int b=0;b<8;b++)du[context_coordinate(h,a,b)]+=context_sign(h,a,b)*.3535533905932738f*dq;
  }
  return true;
}
inline void context_update(Model&m,int r,LaceRecord&v,const float*g,float rate){
  auto*saved=context_record(m,r,v);if(!saved){SHOEBOX_DIAGNOSTIC(m.event.metrics[r][9]++;)return;}
  float denominator=1.f;for(int k=0;k<m.event.heads*MK_DIM;k++)denominator+=saved->read[k]*saved->read[k];
  float a=rate/denominator;
  if(r>=6&&r<=14&&m.utility.config[7]>0&&m.utility.config[7]!=1.f)a*=m.utility.config[7];
  for(int i=0;i<D;i++){
    float d=std::clamp(g[i]*(1-v.h[i]*v.h[i]),-2.f,2.f);
    m.event.parameters.bias[r][i]-=a*d;
    for(int k=0;k<m.event.heads*MK_DIM;k++)m.event.parameters.readout[r][i][k]-=a*d*saved->read[k];
    if(m.cfg[10]&&(r==15||r==16)){
      float*port=m.P+r*D*D;float b=rate*d/norm(v.u);
      if(int(m.cfg[10])==2)port[i*D+i]-=b*v.returned[i];
      else for(int j=0;j<D;j++)port[i*D+j]-=b*v.returned[j];
    }
  }
  SHOEBOX_DIAGNOSTIC(m.event.metrics[r][1]++;)
}
