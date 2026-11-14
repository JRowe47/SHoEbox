inline std::vector<bool> reference_semantic_marks(Model&m,const ContentEvent*parts,int n){
  auto&s=m.composition.stream;
  std::vector<SemanticRef> work;std::vector<bool> marks(constituent_capacity(m));
  auto add=[&](SemanticRef ref){
    if(!constituent_record(m,ref)||marks[ref.index])return;
    marks[ref.index]=true;work.push_back(ref);
  };
  for(int a=0;a<N;a++){
    for(int k=0;k<2;k++)add(s.history[a][k].semantic);
    for(int k=0;k<s.length[a];k++)add(s.pending[a][k].semantic);
    for(int k=0;k<LACE_RING;k++)add(s.input[a][k].semantic);
    for(const auto&entry:m.neural_archive[a])add(entry.second.input.semantic);
  }
  for(int k=0;k<n;k++)add(parts[k].semantic);
  add(m.semantic_inflight);
  for(const auto&root:m.reverse_roots)add(root.ref);
  for(auto ref:m.event.new_events)add(ref);
  for(int a=2;a<N;a++)for(int t=0;t<2;t++)add(m.event.replay[a][t].request.ref);
  for(size_t i=0;i<work.size();i++){
    auto&node=*constituent_record(m,work[i]);for(int k=0;k<node.count;k++)if(event_follow(m,node.ref,k))add(node.edges[k].source);
  }
  return marks;
}
