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
