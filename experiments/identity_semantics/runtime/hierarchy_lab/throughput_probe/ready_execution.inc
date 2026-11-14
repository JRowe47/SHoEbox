// Pack fixed recipes together, including the S2/R2/F2 preparation family.
// Numerical encoding/teaching stays at its original regional boundary: a
// publication can be ready while its parameter bank still has an online writer.
inline void execution_rebuild(Model&m){
  auto&x=m.execution;memset(x.listeners,0,sizeof(x.listeners));
  for(int r=0;r<N;r++){
    auto&recipe=x.recipe[r];recipe.immediate.clear();recipe.squared=0;recipe.multiplicity=0;
    recipe.driver=m.starts[r]<m.starts[r+1]?int(m.slots[4*m.starts[r]]):0;
    for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
      int source=int(m.slots[4*k]);float weight=m.slots[4*k+3];
      recipe.immediate.push_back(k);recipe.squared+=weight*weight;
      recipe.multiplicity+=source==recipe.driver;
      if(r>=2&&(r!=2||source==recipe.driver))x.listeners[source]|=uint32_t(1)<<r;
    }
    recipe.scale=1.f/std::sqrt(std::max(1.f,recipe.squared));
  }
  x.family_compatible=true;
  for(int r=3;r<=5;r++)for(int k:x.recipe[r].immediate)
    if(int(m.slots[4*k])>=3)x.family_compatible=false;
  x.valid=true;
}
inline const RegionalExecution::Recipe& execution_recipe(Model&m,int r){
  if(!m.execution.valid)execution_rebuild(m);return m.execution.recipe[r];
}
inline void execution_begin_run(Model&m){
  execution_rebuild(m);
  // Reconstruct readiness at every API entry, including checkpoint imports.
  // False candidates have no operational side effects in content_prepare.
  m.execution.pending=(uint32_t(1)<<N)-1;
  m.execution.family_tick=-1;
}
inline void execution_begin_step(Model&m,int byte){
  m.execution.family_tick=-1;
  for(auto&frame:m.execution.bottom_family){
    for(int i=0;i<frame.count;i++)frame.children[i]=ContentEvent{};
    frame.count=0;frame.ready=false;frame.fresh=0;
  }
  if(m.composition.stream.draining)m.execution.pending=(uint32_t(1)<<N)-4;
  else {m.execution.pending|=3u|(1u<<17);memcpy(m.execution.byte,m.emb+byte*D,D*sizeof(float));}
}
inline int execution_next(Model&m,int first){
  if(!m.composition.enabled)return first;
#ifdef SHOEBOX_CONTRIBUTION_CHECKS
  // This dedicated intervention oracle records attempted source reads too.
  // Preserve its exhaustive probe schedule; production uses the ready bitset.
  return first;
#endif
  uint32_t candidates=m.execution.pending&(~uint32_t(0)<<first);
  int r=candidates?__builtin_ctz(candidates):N;
  SHOEBOX_DIAGNOSTIC(for(int skipped=first;skipped<r;skipped++)
    if(skipped>=2)m.composition.stream.metrics[skipped][5]++;)
  if(r<N)m.execution.pending&=~(1u<<r);return r;
}
inline void execution_publish(Model&m,int source){
  if(!m.execution.valid)execution_rebuild(m);
  // A lower-index listener published after its turn remains pending until the
  // next step. The ascending drain therefore executes each region at most once.
  m.execution.pending|=m.execution.listeners[source];
}
