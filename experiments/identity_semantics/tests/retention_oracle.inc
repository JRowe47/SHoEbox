// Independent full-mark oracle copied from Action 1. Test-only.
struct ReferenceNeuralMarks {std::map<int64_t,int> marks[N];};
inline ReferenceNeuralMarks reference_neural_marks(Model&m){
  neural_ensure_index(m);
  struct Work {int region;int64_t tick;int budget;};
  std::vector<Work> work;
  ReferenceNeuralMarks result;auto&marks=result.marks;
  const int depth=std::clamp(int(m.lcfg[0]),0,LACE_DEPTH);
  auto add=[&](int r,int64_t tick,int index,int budget){
    if(!tick||r<0||r>=N)return;
    auto found=marks[r].find(tick);
    if(found!=marks[r].end()&&found->second>=budget)return;
    if(!find_lace(m,r,tick,index)){SHOEBOX_DIAGNOSTIC(m.neural_metrics[r][3]++;)return;}
    marks[r][tick]=budget;work.push_back({r,tick,budget});
  };
  auto event=[&](const ContentEvent&e){
    add(e.origin.region,e.origin.tick,e.origin.index,depth);
    for(int k=0;k<e.context_data().context_count;k++){const auto&edge=e.context_data().context_edges[k];
      add(edge.region,edge.tick,edge.index,depth);
    }
  };
  auto&s=m.composition.stream;
  for(int r=0;r<N;r++){
    // Cache entries may receive direct teaching and cover the recent histories.
    for(int k=0;k<LACE_RING;k++)add(r,m.laces.records[r][k].tick,k,depth);
    for(int k=0;k<2;k++)event(s.history[r][k]);
    for(int k=0;k<s.length[r];k++)event(s.pending[r][k]);
    add(r,s.memory_tick[r],s.memory_index[r],depth);
    add(r,s.teaching[r].tick,s.teaching[r].index,depth);
    // Sparse pending indices are the roots of unresolved work. Do not scan all
    // immutable archived producers just to rediscover their empty gradients.
    for(const auto*pending:{&m.pending_neural[r],&m.pending_input[r]})
      for(const auto&entry:*pending)for(int d=0;d<LACE_DEPTH;d++)if(entry.second&(1u<<d))
        add(r,entry.first,-1,std::max(0,depth-d-1));
  }
  for(size_t i=0;i<work.size();i++){
    const auto item=work[i];int r=item.region;
    auto*v=find_lace(m,r,item.tick,-1);if(!v)continue;
    // -1 means a temporal Jacobian only. Do not recursively sample an earlier
    // temporal parent for such an intermediate record.
    if(item.budget<0)continue;
    if(r<2&&m.scfg[0]>0&&m.scfg[5]>0&&item.tick%std::max(1,int(m.scfg[6]))==0){
      const int ages[]{1,2,4,8,16,32};int age=ages[(item.tick/std::max(1,int(m.scfg[6])))%6];
      for(int k=1;k<=age;k++){
        // A document boundary can legitimately end this sampled history.
        if(find_lace(m,r,item.tick-k,-1))add(r,item.tick-k,-1,-1);
      }
    }
    if(item.budget==0)continue;
    auto edge=[&](const LaceParent&p){
      auto*producer=find_lace(m,p.region,p.tick,p.index);
      add(p.region,p.tick,p.index,item.budget-(producer&&producer->expert<0?0:1));
    };
    for(int k=0;k<v->nparents;k++)edge(v->parents[k]);
    auto*input=lace_input(m,r,*v);
    if(input&&input->tick==v->tick){
      for(int k=0;k<input->message.feedback_count;k++)edge(input->message.feedback[k]);
      if(input->retention!=0)add(r,input->previous_tick,input->previous_index,item.budget-1);
    }
  }
  return result;
}
