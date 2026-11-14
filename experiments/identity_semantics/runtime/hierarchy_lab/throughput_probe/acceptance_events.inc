inline void acceptance_publish(Model&m,int r,const ContentEvent&e,bool relay,bool formed){
  auto&z=m.acceptance;auto&v=z.events[r];
  bool no_primary=false;
  if(r>=15&&e.kind==4&&e.clock_begin==0&&e.clock_end==0){
    int driver=int(m.slots[4*m.starts[r]]);no_primary=true;
    for(int i=0;i<e.count&&i<COMP_MAX;i++)if(e.values().children[i].region==driver&&e.values().children[i].seq)no_primary=false;
  }
  bool primary=e.clock_begin>=0&&e.clock_end>e.clock_begin;
  bool coverage=e.ref.begin>=0&&e.ref.end>e.ref.begin;
  uint64_t span=primary?uint64_t(e.clock_end-e.clock_begin):0;
  uint64_t covered=coverage?uint64_t(e.ref.end-e.ref.begin):0;
  v[0]++;v[1]+=span;v[2]=std::max(v[2],span);v[3]+=covered;v[4]=std::max(v[4],covered);
  v[5]+=formed;v[6]+=formed&&e.ref.recognized;v[7]+=formed&&!e.ref.recognized;
  v[8]+=relay;v[9]+=!primary&&!no_primary;v[10]+=!coverage;v[12]+=no_primary;
  v[11]+=primary&&coverage&&(e.clock_begin<e.ref.begin||e.clock_end>e.ref.end);
  z.primary_hist[r][span?64-__builtin_clzll(span):0]++;
  z.coverage_hist[r][covered?64-__builtin_clzll(covered):0]++;
  if(r>=2&&r<=14&&!relay){
    int frames=e.kind==1?e.count:1;
    // Bin zero exposes malformed sequence arity without an out-of-bounds write.
    if(frames<1||frames>COMP_MAX)frames=0;
    z.lengths[r][frames]++;
    z.recognized_lengths[r][frames]+=formed&&e.ref.recognized;
  }
}
