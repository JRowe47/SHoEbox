// Every request in this packet belongs to one local objective and is evaluated
// before its ordered writer runs. Context/sparse kernels keep their own exact
// reductions; dense requests to the same bank share its row loads.
inline void local_adjoint_packet(Model&m,LocalAdjoint*const*jobs,int count){
  int q=0;
  for(;q+4<=count;q+=4){
    float pre[4][D];
    for(int a=0;a<4;a++){
      auto&job=*jobs[q+a];const auto&v=*job.producer;
      for(int i=0;i<D;i++)job.du[i]=pre[a][i]=job.mean[i]*(1-v.h[i]*v.h[i]);
    }
    const float*w=m.weight_bank(jobs[q]->region*E+jobs[q]->producer->expert);
    float g0=jobs[q]->producer->gate,g1=jobs[q+1]->producer->gate;
    float g2=jobs[q+2]->producer->gate,g3=jobs[q+3]->producer->gate;
    for(int i=0;i<D;i++)for(int j=0;j<D;j++){
      float weight=w[i*D+j];
      jobs[q]->du[j]+=g0*weight*pre[0][i];jobs[q+1]->du[j]+=g1*weight*pre[1][i];
      jobs[q+2]->du[j]+=g2*weight*pre[2][i];jobs[q+3]->du[j]+=g3*weight*pre[3][i];
    }
  }
  for(;q<count;q++){auto&job=*jobs[q];neural_input_gradient(m,job.region,*job.producer,job.mean,job.du);}
}
inline void local_prepare_adjoints(Model&m,LocalCredit&work,int depth){
  auto&signals=work.pending[depth];work.adjoints.resize(signals.order.size());
  int limit=std::clamp(int(m.lcfg[0]),0,LACE_DEPTH);
  for(size_t i=0;i<signals.order.size();i++){
    const auto&entry=signals.entries[signals.order[i]];auto&job=work.adjoints[i];job=LocalAdjoint{};
    job.region=entry.first.first;job.producer=find_lace(m,job.region,entry.first.second);
    if(entry.second.neural_uses)for(int j=0;j<D;j++)job.mean[j]=entry.second.neural[j]/entry.second.neural_uses;
  }
  if(depth+1>=limit)return;
  for(size_t begin=0;begin<work.adjoints.size();begin+=16){
    size_t end=std::min(begin+16,work.adjoints.size());bool done[16]{};
    for(size_t a=begin;a<end;a++)if(!done[a-begin]){
      auto&job=work.adjoints[a];const auto&signal=signals.entries[signals.order[a]].second;
      if(!job.producer||!signal.neural_uses||job.producer->expert<0)continue;
      if(m.lcfg[9]<D||context_enabled(m,job.region)){
        neural_input_gradient(m,job.region,*job.producer,job.mean,job.du);done[a-begin]=true;
      }else{
        LocalAdjoint*packet[16];int n=0;
        for(size_t b=a;b<end;b++){
          auto&other=work.adjoints[b];const auto&candidate=signals.entries[signals.order[b]].second;
          if(!done[b-begin]&&other.producer&&candidate.neural_uses&&other.region==job.region&&
             other.producer->expert==job.producer->expert){packet[n++]=&other;done[b-begin]=true;}
        }
        local_adjoint_packet(m,packet,n);
      }
    }
  }
  for(auto&job:work.adjoints)if(job.producer&&job.producer->expert>=0)
    message_route_gradient(m,job.region,*job.producer,job.mean,job.dcue);
}
