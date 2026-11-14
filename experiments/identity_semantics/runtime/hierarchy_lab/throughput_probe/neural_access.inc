inline InputCredit* lace_input(Model&,int,const LaceRecord&);
#ifdef SHOEBOX_LEARNED_COMPLETION
inline void neural_invalidate(Model&m){
  m.neural_index_valid=false;
  for(int r=0;r<N;r++){
    m.neural_index[r].clear();m.pending_neural[r].clear();m.pending_input[r].clear();
  }
}
inline void neural_ensure_index(Model&m){
  if(m.neural_index_valid)return;
  for(int r=0;r<N;r++){
    auto&index=m.neural_index[r];index.clear();
    m.pending_neural[r].clear();m.pending_input[r].clear();
    index.reserve(m.neural_archive[r].size()+LACE_RING);
    auto add=[&](LaceRecord&v,const InputCredit&input){
      if(!v.tick)return;
      index[v.tick]=&v;
      for(int d=0;d<LACE_DEPTH;d++){
        if(v.uses[d])m.pending_neural[r][v.tick]|=1u<<d;
        if(input.tick==v.tick&&input.uses[d])m.pending_input[r][v.tick]|=1u<<d;
      }
    };
    for(auto&entry:m.neural_archive[r])add(entry.second.lace,entry.second.input);
    for(int k=0;k<LACE_RING;k++)add(m.laces.records[r][k],m.composition.stream.input[r][k]);
  }
  m.neural_index_valid=true;
}
inline void neural_pending(Model&m,int r,int64_t tick,int depth,bool input=false){
  neural_ensure_index(m);
  (input?m.pending_input[r]:m.pending_neural[r])[tick]|=1u<<depth;
}
inline void neural_consumed(Model&m,int r,int64_t tick,int depth,bool input=false){
  auto&pending=input?m.pending_input[r]:m.pending_neural[r];
  auto it=pending.find(tick);if(it==pending.end())return;
  it->second&=~(1u<<depth);
  // Release through the indexed owner. No iterators/pointers to gradient rows
  // survive this call; propagation may recreate work at the same tick.
  if(!it->second){
    auto found=m.neural_index[r].find(tick);
    if(found!=m.neural_index[r].end()){
      if(!input)found->second->credit.release_if_zero();
      else if(auto*saved=lace_input(m,r,*found->second))saved->gradient.release_if_zero();
    }
    pending.erase(it);
  }
}
#endif
// Locate a cache entry without subtracting pointers from unrelated allocations.
inline int lace_cache_index(Model&m,int r,const LaceRecord&v){
  uintptr_t address=reinterpret_cast<uintptr_t>(&v),base=reinterpret_cast<uintptr_t>(&m.laces.records[r][0]);
  if(address<base||address-base>=sizeof(m.laces.records[r])||(address-base)%sizeof(LaceRecord))return -1;
  return int((address-base)/sizeof(LaceRecord));
}
inline InputCredit* lace_input(Model&m,int r,const LaceRecord&v){
  int index=lace_cache_index(m,r,v);
  if(index>=0)return &m.composition.stream.input[r][index];
#ifdef SHOEBOX_LEARNED_COMPLETION
  auto found=m.neural_archive[r].find(v.tick);
  if(found!=m.neural_archive[r].end()&&found->second.lace.tick==v.tick)return &found->second.input;
#endif
  return nullptr;
}
inline const float* lace_cue(Model&m,int r,const LaceRecord&v){
  int index=lace_cache_index(m,r,v);
  if(index>=0)return m.probe.ticks[r][index]==v.tick?m.probe.cues[r][index]:nullptr;
#ifdef SHOEBOX_LEARNED_COMPLETION
  auto found=m.neural_archive[r].find(v.tick);
  if(found!=m.neural_archive[r].end()&&found->second.lace.tick==v.tick)return found->second.cue;
#endif
  return nullptr;
}
// Chronological order is independent of cache wrap, checkpoint and chunking.
inline std::vector<LaceRecord*> lace_records(Model&m,int r){
  std::vector<LaceRecord*> records;
#ifdef SHOEBOX_LEARNED_COMPLETION
  records.reserve(LACE_RING+m.neural_archive[r].size());
  for(auto&entry:m.neural_archive[r])records.push_back(&entry.second.lace);
#else
  records.reserve(LACE_RING);
#endif
  for(int a=LACE_RING-1;a>=0;a--){
    auto&v=m.laces.records[r][(m.laces.cursor[r]-a)&(LACE_RING-1)];
    if(v.tick)records.push_back(&v);
  }
  return records;
}
