// Per-instance controls, outside every serialized state. All interventions are
// evaluation-only. Exact IDs, boundaries, recognition and provenance survive
// every intervention except the explicitly destructive whole-source control.
enum ContributionKind { CHECK_NONE, CHECK_NEURAL, CHECK_DESCRIPTOR,
                        CHECK_RELAY, CHECK_RETAINED, CHECK_SEMANTIC, CHECK_SOURCE,
                        CHECK_EXPERT };
inline bool contribution_selected(const Model&m,int kind,int r){
  return m.contribution_kind==kind&&m.contribution_region==r;
}
inline bool contribution_apply(Model&m,int kind,int r){
  if(!contribution_selected(m,kind,r))return false;
  m.contribution_counts[r][4]++;return true;
}
inline uint64_t contribution_hash(uint64_t h,uint64_t value){
  for(int i=0;i<8;i++){h^=value&255;h*=1099511628211ULL;value>>=8;}return h;
}
inline void contribution_event(Model&m,int r,const ContentEvent&e){
  auto&v=m.contribution_counts[r];v[0]++;
  const uint64_t span=e.ref.end-e.ref.begin;v[2]+=span;v[3]=std::max(v[3],span);
  uint64_t h=m.contribution_hashes[r];
  auto add=[&](uint64_t x){h=contribution_hash(h,x);};
  auto ref=[&](const ChildEvent&x){add(x.id.a);add(x.id.b);add(x.seq);add(x.begin);add(x.end);
    add(x.tick);add(x.region);add(x.role);add(x.recognized);add(x.index);};
  ref(e.ref);ref(e.origin);add(e.count);add(e.kind);add(e.local_arity);add(e.relayed);
  add(e.present_mask);add(e.fresh_mask);
  for(int i=0;i<e.count;i++)ref(e.children[i]);
  m.contribution_hashes[r]=h;
}
