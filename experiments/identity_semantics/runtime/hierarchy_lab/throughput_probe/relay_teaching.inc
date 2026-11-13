// A forecast targets the next completed content event in this region, including
// an unchanged singleton. Idle ticks are not targets. Relays never schedule a
// forecast or claim a local expert execution.
inline void relay_schedule(Model&m,int r,const float*q){
  auto&z=m.composition.stream;auto&t=z.teaching[r];
  if(t.tick){z.teaching_metrics[r][5]++;return;}
  auto*v=current_lace(m,r);if(!v||v->expert<0)return;
  t.tick=v->tick;t.index=m.laces.cursor[r];
  memcpy(t.input,q,sizeof(t.input));
  for(int i=0;i<D;i++){
    float value=m.pbias[r*D+i];
    for(int j=0;j<D;j++)value+=m.P[(r*D+i)*D+j]*v->h[j];
    t.prediction[i]=value;
  }
  z.teaching_metrics[r][0]++;
}
inline void relay_teach(Model&m,int r,const float*q,int mode,float lp,bool relay){
  auto&z=m.composition.stream;auto&t=z.teaching[r];
  if(!t.tick)return;
  const int64_t tick=t.tick;t.tick=0; // exactly one target, including zero-rate probes
  if((mode!=1&&mode!=3&&mode!=6)||m.frozen_region[r]||
     m.utility.config[3]<=0||(m.cfg[10]&&(r==15||r==16))){
    z.teaching_metrics[r][3]++;return;
  }
  auto*producer=find_lace(m,r,tick,t.index);
  if(!producer||producer->expert<0){z.teaching_metrics[r][4]++;return;}
  auto&v=*producer;
  float err[D],g[D]{};double mse=0;
  for(int i=0;i<D;i++){
    const float target=q[i]-(m.utility.config[4]>0?t.input[i]:0.f);
    err[i]=std::clamp(t.prediction[i]-target,-1.f,1.f);mse+=err[i]*err[i];
  }
  float*P=m.P+r*D*D,*pb=m.pbias+r*D;
  for(int i=0;i<D;i++)for(int j=0;j<D;j++)g[j]+=P[i*D+j]*err[i];
  float rate=lp*m.utility.config[3];
  if(m.utility_observe){double norm2=0;for(int j=0;j<D;j++)norm2+=double(g[j])*g[j];
    m.utility_metrics[r][2]+=std::sqrt(norm2);m.utility_metrics[r][3]++;}
  // The saved record supplies h/u/expert/gate, router cue, semantic eligibility
  // and producer parents. Neither the relay's h nor the current expert is used.
  expert_update(m,r,g,rate,&v,1);
  float a=rate*m.teaching_scale[r][1]/norm(v.h);
  if(a!=0)for(int i=0;i<D;i++){
    pb[i]-=a*err[i];for(int j=0;j<D;j++)P[i*D+j]-=a*err[i]*v.h[j];
  }
  m.s.mse[r]+=mse/D;z.teaching_metrics[r][relay?2:1]++;
}
