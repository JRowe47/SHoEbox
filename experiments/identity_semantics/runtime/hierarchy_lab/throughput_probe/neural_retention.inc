inline LaceRecord* find_lace(Model&,int,int64_t,int);
inline int retention_node(Model&m,int r,int64_t tick){
  auto&z=m.neural_retention;auto found=z.index[r].find(tick);
  if(found!=z.index[r].end())return found->second;
  int id;
  if(z.free.empty()){id=int(z.nodes.size());z.nodes.emplace_back();}
  else {id=z.free.back();z.free.pop_back();}
  auto&v=z.nodes[id];v.region=r;v.tick=tick;v.root=v.next_root=v.active=-2;
  v.generation=0;v.queued=false;v.edges.clear();memset(v.incoming,0,sizeof(v.incoming));
  z.index[r].emplace(tick,id);z.candidates.push_back(id);return id;
}
inline void retention_queue(NeuralRetention&z,int id){
  if(!z.nodes[id].queued){z.nodes[id].queued=true;z.work.push_back(id);}
}
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
  m.event.context[r][index].clear();
  if(m.neural_retention.initialized)retention_node(m,r,v.tick);
  auto&metrics=m.neural_metrics[r];SHOEBOX_DIAGNOSTIC(SHOEBOX_DIAGNOSTIC(metrics[0]++;))
  SHOEBOX_DIAGNOSTIC(SHOEBOX_DIAGNOSTIC(metrics[2]=std::max(metrics[2],uint64_t(m.neural_archive[r].size()));))
}
// Reconcile changed roots, then propagate only changed maximum budgets. Every
// edge owns an exact budget count. Temporal Jacobians use -1 (retain without
// expansion), and transparent relays consume zero neural depth.
inline void neural_collect(Model&m){
  neural_ensure_index(m);auto&z=m.neural_retention;
  int depth=std::clamp(int(m.lcfg[0]),0,LACE_DEPTH);
  int temporal=m.scfg[0]>0&&m.scfg[5]>0,stride=std::max(1,int(m.scfg[6]));
  if(z.initialized&&(z.depth!=depth||z.temporal!=temporal||z.stride!=stride))z.invalidate();
  if(!z.initialized){
    z.depth=depth;z.temporal=temporal;z.stride=stride;z.initialized=true;
    for(int r=0;r<N;r++)for(const auto&entry:m.neural_archive[r])retention_node(m,r,entry.first);
  }
  if(++z.generation==0){for(auto&v:z.nodes)v.generation=0;z.generation=1;}
  z.next_roots.clear();z.work.clear();
  auto add=[&](int r,int64_t tick,int index,int budget){
    if(!tick||r<0||r>=N)return;
    if(!find_lace(m,r,tick,index)){SHOEBOX_DIAGNOSTIC(m.neural_metrics[r][3]++;)return;}
    int id=retention_node(m,r,tick);auto&v=z.nodes[id];
    if(v.generation!=z.generation){v.generation=z.generation;v.next_root=budget;z.next_roots.push_back(id);}
    else v.next_root=std::max(v.next_root,budget);
  };
  auto event=[&](const ContentEvent&e){
    add(e.origin.region,e.origin.tick,e.origin.index,depth);
    for(int k=0;k<e.context_data().context_count;k++){const auto&p=e.context_data().context_edges[k];add(p.region,p.tick,p.index,depth);}
  };
  auto&s=m.composition.stream;
  for(int r=0;r<N;r++){
    for(int k=0;k<LACE_RING;k++)add(r,m.laces.records[r][k].tick,k,depth);
    for(const auto&e:s.history[r])event(e);
    for(int k=0;k<s.length[r];k++)event(s.pending[r][k]);
    add(r,s.memory_tick[r],s.memory_index[r],depth);add(r,s.teaching[r].tick,s.teaching[r].index,depth);
    for(const auto*pending:{&m.pending_neural[r],&m.pending_input[r]})
      for(const auto&entry:*pending)for(int d=0;d<LACE_DEPTH;d++)if(entry.second&(1u<<d))
        add(r,entry.first,-1,std::max(0,depth-d-1));
  }
  for(int id:z.roots)if(z.nodes[id].generation!=z.generation){z.nodes[id].root=-2;retention_queue(z,id);}
  for(int id:z.next_roots)if(z.nodes[id].root!=z.nodes[id].next_root){z.nodes[id].root=z.nodes[id].next_root;retention_queue(z,id);}
  z.roots.swap(z.next_roots);
  for(size_t i=0;i<z.work.size();i++){
    int id=z.work[i];z.nodes[id].queued=false;
    int budget=z.nodes[id].root;
    for(int b=LACE_DEPTH;b>=-1;b--)if(z.nodes[id].incoming[b+1]){budget=std::max(budget,b);break;}
    if(budget==z.nodes[id].active)continue;
    z.nodes[id].active=budget;
    // Release the previous expansion before replacing it. Nodes are not
    // recycled until the entire propagation queue is quiescent.
    for(const auto&e:z.nodes[id].edges){--z.nodes[e.node].incoming[e.budget+1];retention_queue(z,e.node);}
    z.nodes[id].edges.clear();
    if(budget==-2){z.candidates.push_back(id);continue;}
    if(budget==-1)continue;
    int r=z.nodes[id].region;int64_t tick=z.nodes[id].tick;
    auto*v=find_lace(m,r,tick,-1);if(!v)continue;
    auto edge=[&](int region,int64_t at,int hint,int remaining){
      if(!at||region<0||region>=N)return;
      if(!find_lace(m,region,at,hint)){SHOEBOX_DIAGNOSTIC(m.neural_metrics[region][3]++;)return;}
      int target=retention_node(m,region,at);
      ++z.nodes[target].incoming[remaining+1];retention_queue(z,target);
      z.nodes[id].edges.push_back({target,remaining});
    };
    if(r<2&&temporal&&tick%stride==0){
      const int ages[]{1,2,4,8,16,32};int age=ages[(tick/stride)%6];
      for(int k=1;k<=age;k++)if(find_lace(m,r,tick-k,-1))edge(r,tick-k,-1,-1);
    }
    if(budget==0)continue;
    auto parent=[&](const LaceParent&p){
      auto*producer=find_lace(m,p.region,p.tick,p.index);
      edge(p.region,p.tick,p.index,budget-(producer&&producer->expert<0?0:1));
    };
    for(int k=0;k<v->nparents;k++)parent(v->parents[k]);
    auto*input=lace_input(m,r,*v);
    if(input&&input->tick==tick){
      for(int k=0;k<input->message.feedback_count;k++)parent(input->message.feedback[k]);
      if(input->retention!=0)edge(r,input->previous_tick,input->previous_index,budget-1);
    }
  }
  for(int id:z.candidates){
    auto&v=z.nodes[id];if(v.region<0||v.active!=-2)continue;
    int r=v.region;auto tick=v.tick;
    if(m.neural_archive[r].erase(tick)){
      m.event.archive[r].erase(tick);m.neural_index[r].erase(tick);
      m.pending_neural[r].erase(tick);m.pending_input[r].erase(tick);
      if(r<STATE_REGIONS)m.neural_temporal[r].erase(tick);
      SHOEBOX_DIAGNOSTIC(m.neural_metrics[r][1]++;)
    }
    z.index[r].erase(tick);v.region=-1;z.free.push_back(id);
  }
  z.candidates.clear();z.work.clear();
}
