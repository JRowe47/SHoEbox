inline void constituent_retain(Model&m,const ContentEvent*parts,int n){
  auto&z=m.semantic_retention;auto&s=m.composition.stream;
  if(z.local!=m.event.local||z.frontier!=m.event.frontier||z.local_only!=m.event.local_only)z.invalidate();
  z.local=m.event.local;z.frontier=m.event.frontier;z.local_only=m.event.local_only;
  z.nodes.resize(constituent_capacity(m));
  if(++z.generation==0){for(auto&v:z.nodes)v.root_generation=v.expiry_generation=0;z.generation=1;}
  z.work.clear();z.next_roots.clear();z.next_expiring.clear();
  auto prepare=[&](SemanticRef ref){
    auto&v=z.nodes[ref.index];if(v.serial!=ref.serial){v=SemanticRetention::Node{};v.serial=ref.serial;}
  };
  auto queue=[&](int id){if(!z.nodes[id].queued){z.nodes[id].queued=true;z.work.push_back(id);}};
  auto add=[&](SemanticRef ref){
    if(!constituent_record(m,ref))return;prepare(ref);auto&v=z.nodes[ref.index];
    if(v.root_generation!=z.generation){v.root_generation=z.generation;z.next_roots.push_back(ref.index);}
  };
  for(int r=0;r<N;r++){
    for(const auto&e:s.history[r])add(e.semantic);
    for(int k=0;k<s.length[r];k++)add(s.pending[r][k].semantic);
    for(const auto&input:s.input[r])add(input.semantic);
    for(const auto&entry:m.neural_archive[r])add(entry.second.input.semantic);
  }
  for(int k=0;k<n;k++)add(parts[k].semantic);
  add(m.semantic_inflight);
  for(const auto&root:m.reverse_roots)add(root.ref);
  for(auto ref:m.event.new_events)add(ref);
  for(int r=2;r<N;r++)for(int t=0;t<2;t++)add(m.event.replay[r][t].request.ref);
  for(int id:z.roots)if(z.nodes[id].root_generation!=z.generation){z.nodes[id].root=false;queue(id);}
  for(int id:z.next_roots)if(!z.nodes[id].root){z.nodes[id].root=true;queue(id);}
  for(int id:z.expiring)if(z.nodes[id].active)queue(id);
  z.roots.swap(z.next_roots);
  for(size_t i=0;i<z.work.size();i++){
    int id=z.work[i];auto&saved=z.nodes[id];saved.queued=false;
    bool live=saved.root||saved.incoming>0;
    auto&node=constituent_slot(m,id);unsigned edges=0;
    if(live)for(int k=0;k<node.count;k++)
      if(event_follow(m,node.ref,k)&&constituent_record(m,node.edges[k].source))edges|=1u<<k;
    unsigned changed=edges^saved.edges;
    for(int k=0;k<node.count;k++)if(changed&(1u<<k)){
      auto child=node.edges[k].source;
      if(edges&(1u<<k)){prepare(child);++z.nodes[child.index].incoming;}
      else --z.nodes[child.index].incoming;
      queue(child.index);
    }
    saved.active=live;saved.edges=edges;
    if(m.event.local&&edges&&saved.expiry_generation!=z.generation){
      saved.expiry_generation=z.generation;z.next_expiring.push_back(id);
    }
  }
  z.expiring.swap(z.next_expiring);z.work.clear();
}
inline bool constituent_retained(Model&m,int index){
  auto&z=m.semantic_retention;
  return size_t(index)<z.nodes.size()&&z.nodes[index].active&&
    z.nodes[index].serial==constituent_slot(m,index).ref.serial;
}
