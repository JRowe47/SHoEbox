inline int completion_driver(Model&m,int r,int&multiplicity){
  int driver=int(m.slots[4*m.starts[r]]);multiplicity=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]&&int(m.slots[4*k])==driver)multiplicity++;
  return driver;
}
inline bool completion_frame_ready(Model&m,int r){
  auto&s=m.composition.stream;int required=0,driver=completion_driver(m,r,required);
  if(binding_source_removed(m,driver))return false;
  auto seq=s.history[driver][0].ref.seq;
  if(seq-s.seen[r]<(s.draining?1:required))return false;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
    int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]);
    if(!s.draining&&!binding_source_removed(m,src)&&!s.history[src][age].ref.seq)return false;
  }
  return true;
}
inline bool completion_ready(Model&m,int r){
  auto&s=m.composition.stream;
  return completion_frame_ready(m,r)||(s.draining&&s.length[r]);
}
inline void completion_emit(Model&m,int r,bool learn,ContentEvent&event,ContentEvent*parts,int&n){
  auto&s=m.composition.stream;n=s.length[r];
  for(int i=0;i<n;i++)parts[i]=s.pending[r][i];s.length[r]=0;
  if(n==1){event=parts[0];event.ref.seq=0;}
  else{
    event=content_compose(m,r,parts,n,1,learn);
    event.clock_begin=parts[0].clock_begin;event.clock_end=parts[n-1].clock_end;
    s.completion_metrics[r][3]++;
  }
}
inline bool completion_prepare(Model&m,int r,bool learn,ContentEvent&event,ContentEvent*parts,int&n){
  auto&s=m.composition.stream;n=0;
  if(!completion_frame_ready(m,r)){
    if(s.draining&&s.length[r]){completion_emit(m,r,learn,event,parts,n);return true;}
    s.metrics[r][5]++;return false;
  }
  ContentEvent frame{},children[COMP_MAX];int count=0;
  int required=0,driver=completion_driver(m,r,required);
  uint32_t fresh=0;
  for(int k=m.starts[r];k<m.starts[r+1];k++)if(!m.slots[4*k+2]){
    int src=int(m.slots[4*k]),age=int(m.slots[4*k+1]),role=k-m.starts[r];
    const auto&child=s.history[src][age];
    if(binding_source_removed(m,src)||!child.ref.seq||(src==driver&&child.ref.seq<=s.seen[r]))continue;
    children[count]=child;children[count++].ref.role=k;
    if(s.fusion_seen[r][role]!=child.ref.seq)fresh|=1u<<role;
  }
  if(!count)return false;
  frame=binding_compose(m,r,children,count,learn,fresh);
  m.semantic_inflight=frame.semantic;
  for(int i=0;i<count;i++)s.fusion_seen[r][children[i].ref.role-m.starts[r]]=children[i].ref.seq;
  int fresh_count=__builtin_popcount(fresh);
  s.metrics[r][0]+=fresh_count;s.metrics[r][14]++;s.metrics[r][15]+=count-fresh_count;s.metrics[r][16]+=fresh_count;
  int available=std::min<int64_t>(required,s.history[driver][0].ref.seq-s.seen[r]);
  s.seen[r]=s.history[driver][0].ref.seq;
  frame.clock_begin=s.history[driver][available-1].ref.begin;
  frame.clock_end=s.history[driver][0].ref.end;
  frame.ref.seq=++s.frame_serial[r];frame.ref.tick=m.s.tick;
  float q[D]{},qw=0,cw=0;int parents=0;LaceParent links[LACE_LINKS];
  binding_bundle(m,r,frame,children,count,q,frame.context,qw,cw,links,parents);
  frame.context_count=parents;for(int i=0;i<parents;i++)frame.context_edges[i]=links[i];
  int length=s.length[r];bool boundary=false;
  if(length){
    const auto&last=s.pending[r][length-1];
    float score=content_transition(m,r,last.ref.id,frame.ref.id,learn);
    bool adjacent=last.clock_end==frame.clock_begin;
    boundary=score<m.composition.boundary_threshold||!adjacent;
    if(length==1&&last.ref.recognized&&frame.ref.recognized&&adjacent)boundary=false;
  }
  if(boundary)completion_emit(m,r,learn,event,parts,n);
  s.pending[r][s.length[r]++]=frame;
  m.semantic_inflight=SemanticRef{};
  if(!n&&s.length[r]>=int(m.composition.max_children))completion_emit(m,r,learn,event,parts,n);
  if(!n){s.metrics[r][5]++;return false;}
  return true;
}
inline void completion_bundle(Model&m,int r,const ContentEvent&event,const ContentEvent*parts,int n,
                              float*q,float*c,float&qw,float&cw,LaceParent*parents,int&nparents){
  if(!n)return;
  memcpy(q,event.code,D*sizeof(float));qw=1;cw=1;
  float scale=1.f/std::sqrt(float(n));
  for(int i=0;i<n;i++){
    for(int j=0;j<D;j++)c[j]+=scale*parts[i].context[j];
    for(int k=0;k<parts[i].context_count;k++){
      auto edge=parts[i].context_edges[k];edge.coefficient*=scale;
      if(nparents<LACE_LINKS)parents[nparents++]=edge;
    }
  }
}
