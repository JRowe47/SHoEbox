inline LaceRecord* find_lace(Model&,int,int64_t,int);
// Eviction from the hot cache never destroys a producer. Copy all eligibility
// before publish_state can overwrite its matching bottom temporal record.
inline void neural_archive_slot(Model&m,int r,int index){
  neural_ensure_index(m);
  auto&v=m.laces.records[r][index];if(!v.tick)return;
  auto&saved=m.neural_archive[r][v.tick];
  saved.lace=v;saved.input=m.composition.stream.input[r][index];
  if(saved.input.tick!=v.tick){saved.input=InputCredit{};saved.input.tick=v.tick;}
  memcpy(saved.cue,m.probe.cues[r][index],sizeof(saved.cue));
  m.neural_index[r][v.tick]=&saved.lace;
  if(r<2){const auto&temporal=m.learned.records[r][v.tick&(TEMP_RING-1)];
    if(temporal.tick==v.tick)m.neural_temporal[r][v.tick]=temporal;
  }
  if(r>=2&&m.event.heads>0&&m.event.context[r][index].tick==v.tick)m.event.archive[r][v.tick]=m.event.context[r][index];
  m.event.context[r][index]=ContextEligibility{};
  auto&metrics=m.neural_metrics[r];metrics[0]++;
  metrics[2]=std::max(metrics[2],uint64_t(m.neural_archive[r].size()));
}
// Mark live owners, then follow only dependencies reachable within the explicit
// neural credit depth. This retains held events indefinitely without turning
// retained recurrent input into an unbounded temporal tape.
inline void neural_collect(Model&m){
  neural_ensure_index(m);
  struct Work {int region;int64_t tick;int budget;};
  std::vector<Work> work;
  std::map<int64_t,int> marks[N];
  const int depth=std::clamp(int(m.lcfg[0]),0,LACE_DEPTH);
  auto add=[&](int r,int64_t tick,int index,int budget){
    if(!tick||r<0||r>=N)return;
    auto found=marks[r].find(tick);
    if(found!=marks[r].end()&&found->second>=budget)return;
    if(!find_lace(m,r,tick,index)){m.neural_metrics[r][3]++;return;}
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
  for(int r=0;r<N;r++)for(auto it=m.neural_archive[r].begin();it!=m.neural_archive[r].end();){
    if(!marks[r].count(it->first)){
      m.event.archive[r].erase(it->first);
      m.neural_index[r].erase(it->first);
      m.pending_neural[r].erase(it->first);m.pending_input[r].erase(it->first);
      if(r<STATE_REGIONS)m.neural_temporal[r].erase(it->first);
      it=m.neural_archive[r].erase(it);m.neural_metrics[r][1]++;
    }
    else ++it;
  }
}
